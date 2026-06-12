import json
from json import JSONDecodeError
from typing import Any

from loguru import logger
from redis.exceptions import RedisError

from src.config import get_settings
from src.database import get_redis_client


def _session_key(sender_id: str) -> str:
    return f"session:{sender_id}"


async def get_session_history(sender_id: str) -> list[dict[str, Any]]:
    key = _session_key(sender_id)

    try:
        redis_client = await get_redis_client()
        raw_messages = await redis_client.lrange(key, 0, -1)
    except RedisError:
        logger.exception("Failed to read session history from Redis", sender_id=sender_id)
        raise

    messages: list[dict[str, Any]] = []
    for raw_message in raw_messages:
        try:
            message = json.loads(raw_message)
        except JSONDecodeError:
            logger.exception(
                "Invalid JSON message found in Redis session history",
                sender_id=sender_id,
                raw_message=raw_message,
            )
            continue

        if isinstance(message, dict):
            messages.append(message)

    return messages


async def save_session_message(
    sender_id: str,
    role: str,
    content: str,
    max_messages: int = 10,
) -> None:
    if max_messages < 1:
        raise ValueError("max_messages must be greater than 0")

    settings = get_settings()
    key = _session_key(sender_id)
    message = {"role": role, "content": content}

    try:
        redis_client = await get_redis_client()
        await redis_client.rpush(key, json.dumps(message, ensure_ascii=False))
        await redis_client.ltrim(key, -max_messages, -1)
        await redis_client.expire(key, settings.session_ttl_seconds)
        logger.info(
            "Saved session message to Redis",
            sender_id=sender_id,
            role=role,
            max_messages=max_messages,
            ttl_seconds=settings.session_ttl_seconds,
        )
    except RedisError:
        logger.exception("Failed to save session message to Redis", sender_id=sender_id)
        raise
