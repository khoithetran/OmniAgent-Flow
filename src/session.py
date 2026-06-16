"""Redis-backed session state for the Telegram chatbot.

This module owns two pieces of per-conversation state:

1. **Chat history** - a sliding window of the last 10 messages exchanged
   with a given sender. Stored in a Redis list under the
   ``session:{sender_id}`` key with a 30-minute TTL that refreshes on
   every write. This gives the LLM short-term context without
   committing the full history to disk.

2. **Pending crawl** - when the LLM detects a company/org mention
   without an accompanying URL, we set a short-lived marker under
   ``pending_crawl:{sender_id}`` and ask the user to send a URL.
   When the next URL arrives, the worker reads the marker, crawls
   the URL, then deletes the marker. The TTL (5 minutes by default)
   keeps stale markers from accumulating.

The session module never calls OpenAI or crawls anything itself. It
only deals with Redis. Other layers (chat, rag, crawler) call into
it to read/write state.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.config import get_settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chat_key(sender_id: str) -> str:
    return f"session:{sender_id}"


def _pending_crawl_key(sender_id: str) -> str:
    return f"pending_crawl:{sender_id}"


async def _get_client() -> Redis:
    """Resolve a Redis client from the global pool initialised in main.

    Imported lazily to avoid a circular import between this module and
    ``src.main`` (which uses the same global client).
    """
    from src.main import redis_client

    if redis_client is None:  # type: ignore[truthy-bool]
        raise RuntimeError(
            "Redis client has not been initialised. "
            "Make sure the FastAPI lifespan started before the first request."
        )
    return redis_client  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------


async def get_history(sender_id: str) -> list[dict[str, Any]]:
    """Return the saved messages for a sender, oldest first.

    Invalid JSON entries are skipped silently and logged so one bad
    write does not break the whole session.
    """
    key = _chat_key(sender_id)
    try:
        client = await _get_client()
        raw_items = await client.lrange(key, 0, -1)
    except RedisError:
        logger.exception("Failed to read chat history from Redis", sender_id=sender_id)
        raise

    messages: list[dict[str, Any]] = []
    for raw in raw_items:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.exception(
                "Skipping invalid JSON in session history",
                sender_id=sender_id,
                raw=raw,
            )
            continue
        if isinstance(payload, dict) and "role" in payload and "content" in payload:
            messages.append(payload)
    return messages


async def save_message(
    sender_id: str,
    role: str,
    content: str,
    max_messages: int = 10,
) -> None:
    """Append a message to the sliding window.

    The window uses ``RPUSH`` + ``LTRIM -N -1`` so only the newest N
    entries are kept. ``EXPIRE`` is called on every write so the
    session lives for ``SESSION_TTL_SECONDS`` from the last message,
    not from the first.
    """
    if max_messages < 1:
        raise ValueError("max_messages must be greater than 0")
    if role not in {"user", "assistant", "system"}:
        raise ValueError(f"role must be one of user/assistant/system, got {role!r}")

    settings = get_settings()
    key = _chat_key(sender_id)
    payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)

    try:
        client = await _get_client()
        await client.rpush(key, payload)
        await client.ltrim(key, -max_messages, -1)
        await client.expire(key, settings.session_ttl_seconds)
        logger.info(
            "Saved chat message to session",
            sender_id=sender_id,
            role=role,
            max_messages=max_messages,
            ttl_seconds=settings.session_ttl_seconds,
        )
    except RedisError:
        logger.exception("Failed to save chat message to Redis", sender_id=sender_id)
        raise


async def clear_history(sender_id: str) -> None:
    """Delete the chat history for a sender. Used by admin/debug paths."""
    try:
        client = await _get_client()
        await client.delete(_chat_key(sender_id))
    except RedisError:
        logger.exception("Failed to clear chat history", sender_id=sender_id)
        raise


# ---------------------------------------------------------------------------
# Pending crawl marker
# ---------------------------------------------------------------------------


async def set_pending_crawl(sender_id: str, company: str) -> None:
    """Mark a sender as waiting for a URL about a specific company/org.

    The marker auto-expires after ``pending_crawl_ttl_seconds`` so we
    never accumulate stale requests.
    """
    settings = get_settings()
    key = _pending_crawl_key(sender_id)
    payload = json.dumps({"company": company, "created_at": _now_iso()}, ensure_ascii=False)
    try:
        client = await _get_client()
        await client.set(key, payload, ex=settings.pending_crawl_ttl_seconds)
        logger.info(
            "Set pending_crawl marker",
            sender_id=sender_id,
            company=company,
            ttl_seconds=settings.pending_crawl_ttl_seconds,
        )
    except RedisError:
        logger.exception("Failed to set pending_crawl marker", sender_id=sender_id)
        raise


async def pop_pending_crawl(sender_id: str) -> dict[str, Any] | None:
    """Atomically read-and-delete the pending_crawl marker.

    Returns the stored payload (or None if no marker exists) so the
    worker can include the previously-detected company name in its
    confirmation reply after the user finally sends a URL.
    """
    key = _pending_crawl_key(sender_id)
    try:
        client = await _get_client()
        # GETDEL is supported on Redis 6.2+. Fall back to a
        # get-then-delete pair if the deployment is older.
        try:
            raw = await client.getdel(key)
        except AttributeError:
            raw = await client.get(key)
            if raw is not None:
                await client.delete(key)
    except RedisError:
        logger.exception("Failed to read pending_crawl marker", sender_id=sender_id)
        raise

    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.exception(
            "Invalid JSON in pending_crawl marker; deleting",
            sender_id=sender_id,
        )
        return None


async def peek_pending_crawl(sender_id: str) -> dict[str, Any] | None:
    """Read the marker without deleting it. Used for read-only checks."""
    key = _pending_crawl_key(sender_id)
    try:
        client = await _get_client()
        raw = await client.get(key)
    except RedisError:
        logger.exception("Failed to peek pending_crawl marker", sender_id=sender_id)
        raise
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
