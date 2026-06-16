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
    """Return a Redis client, reusing the FastAPI lifespan one when set.

    When the lifespan has not been started (e.g. unit tests, one-off
    scripts), we open a transient client from the configured URL and
    cache it on this module so the rest of the chat flow works.
    """
    global _fallback_client

    # Fast path: reuse the lifespan client when available.
    try:
        from src.main import redis_client as lifespan_client

        if lifespan_client is not None:
            return lifespan_client
    except Exception:  # noqa: BLE001
        # Import failure (e.g. src.main not yet loaded) - fall through
        # to the lazy fallback below.
        pass

    if _fallback_client is None:
        from src.config import get_settings

        settings = get_settings()
        _fallback_client = Redis.from_url(
            f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
            decode_responses=True,
        )
        await _fallback_client.ping()
    return _fallback_client


_fallback_client: Redis | None = None


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


# ---------------------------------------------------------------------------
# Email capture
# ---------------------------------------------------------------------------


#: How long we keep a captured email. We want a long window so a user
#: can come back to ask more questions without losing their contact
#: information, but not so long that Redis fills up with stale data.
EMAIL_TTL_SECONDS = 30 * 24 * 3600  # 30 days


def _email_key(sender_id: str) -> str:
    return f"email:{sender_id}"


def _pending_email_key(sender_id: str) -> str:
    return f"pending_email:{sender_id}"


async def set_pending_email(sender_id: str) -> None:
    """Mark a sender as waiting for an email (after a fall-back reply).

    The marker auto-expires after ``pending_crawl_ttl_seconds`` so
    short conversations do not pollute Redis.
    """
    settings = get_settings()
    key = _pending_email_key(sender_id)
    try:
        client = await _get_client()
        await client.set(
            key,
            json.dumps({"created_at": _now_iso()}, ensure_ascii=False),
            ex=settings.pending_crawl_ttl_seconds,
        )
        logger.info(
            "Set pending_email marker",
            sender_id=sender_id,
            ttl_seconds=settings.pending_crawl_ttl_seconds,
        )
    except RedisError:
        logger.exception("Failed to set pending_email marker", sender_id=sender_id)
        raise


async def pop_pending_email(sender_id: str) -> dict[str, Any] | None:
    """Atomically read-and-delete the pending_email marker."""
    key = _pending_email_key(sender_id)
    try:
        client = await _get_client()
        try:
            raw = await client.getdel(key)
        except AttributeError:
            raw = await client.get(key)
            if raw is not None:
                await client.delete(key)
    except RedisError:
        logger.exception("Failed to read pending_email marker", sender_id=sender_id)
        raise

    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def peek_pending_email(sender_id: str) -> dict[str, Any] | None:
    """Read the pending_email marker without deleting it."""
    key = _pending_email_key(sender_id)
    try:
        client = await _get_client()
        raw = await client.get(key)
    except RedisError:
        logger.exception("Failed to peek pending_email marker", sender_id=sender_id)
        raise
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def save_email(sender_id: str, email: str) -> None:
    """Persist a captured email for ``sender_id`` with a long TTL.

    We store the value as a JSON object so the admin can later see
    when it was captured. A subsequent call overwrites the previous
    email; we do not keep a history to keep the schema simple.
    """
    payload = json.dumps(
        {
            "email": email,
            "sender_id": sender_id,
            "captured_at": _now_iso(),
        },
        ensure_ascii=False,
    )
    try:
        client = await _get_client()
        await client.set(_email_key(sender_id), payload, ex=EMAIL_TTL_SECONDS)
        logger.info(
            "Saved email",
            sender_id=sender_id,
            email=email,
            ttl_seconds=EMAIL_TTL_SECONDS,
        )
    except RedisError:
        logger.exception("Failed to save email", sender_id=sender_id)
        raise


async def get_email(sender_id: str) -> str | None:
    """Return the email we have on file for ``sender_id``, if any."""
    try:
        client = await _get_client()
        raw = await client.get(_email_key(sender_id))
    except RedisError:
        logger.exception("Failed to read email", sender_id=sender_id)
        raise
    if raw is None:
        return None
    try:
        return json.loads(raw).get("email")
    except (json.JSONDecodeError, AttributeError):
        return None


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# LLM response cache
# ---------------------------------------------------------------------------


def _cache_key(normalized_question: str) -> str:
    return f"cache:{normalized_question}"


def _normalize_for_cache(text: str) -> str:
    """Lowercase, strip, and collapse whitespace for cache-key consistency."""
    import re

    return re.sub(r"\s+", " ", text.strip().lower())


async def cache_get(user_message: str) -> str | None:
    """Return a cached LLM reply for an identical question, or None.

    Cache hits are logged; misses are silent.
    The key is derived from a normalized (lowercased, whitespace-collapsed)
    version of the user message, so "Công ty A" and "công ty a" share a slot.
    """
    settings = get_settings()
    key = _cache_key(_normalize_for_cache(user_message))
    try:
        client = await _get_client()
        raw = await client.get(key)
    except RedisError:
        logger.exception("Failed to read response cache from Redis")
        return None

    if raw is None:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Cache entry has invalid JSON; skipping", key=key)
        return None

    reply = payload.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        return None

    logger.info("Cache HIT", key=key, ttl_remaining=settings.cache_ttl_seconds)
    return reply


async def cache_set(user_message: str, reply_text: str) -> None:
    """Store an LLM reply under a normalized question key.

    If another coroutine races to write the same key simultaneously,
    the winner just overwrites with an identical value — that is harmless.
    """
    settings = get_settings()
    key = _cache_key(_normalize_for_cache(user_message))
    payload = json.dumps(
        {
            "reply": reply_text,
            "created_at": _now_iso(),
        },
        ensure_ascii=False,
    )
    try:
        client = await _get_client()
        await client.set(key, payload, ex=settings.cache_ttl_seconds)
        logger.info(
            "Cached LLM reply",
            key=key,
            ttl_seconds=settings.cache_ttl_seconds,
        )
    except RedisError:
        # Cache write failures are non-fatal — the user still got a reply.
        logger.exception("Failed to write response cache to Redis")
