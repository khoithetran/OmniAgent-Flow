import asyncio
from typing import Any

from loguru import logger
from redis.exceptions import RedisError

from src.database import close_redis, get_redis_client, init_redis
from src.services.session_service import get_session_history, save_session_message


SENDER_ID = "user_test"
SESSION_KEY = f"session:{SENDER_ID}"


async def main() -> None:
    try:
        await init_redis()
        redis_client = await get_redis_client()
        await redis_client.delete(SESSION_KEY)

        await save_session_message(SENDER_ID, role="user", content="Xin chào")
        await save_session_message(
            SENDER_ID,
            role="assistant",
            content="Chào bạn, tôi là AI",
        )

        history: list[dict[str, Any]] = await get_session_history(SENDER_ID)
        ttl_seconds = await redis_client.ttl(SESSION_KEY)

        logger.info("Session key: {}", SESSION_KEY)
        logger.info("Session history: {}", history)
        logger.info("Session TTL seconds: {}", ttl_seconds)
    except RedisError:
        logger.exception("Redis test failed. Hay kiem tra Redis da duoc bat chua.")
    finally:
        await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
