from typing import Optional

import asyncpg
from loguru import logger
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.config import get_settings


postgres_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[Redis] = None


async def connect_postgres() -> asyncpg.Pool:
    global postgres_pool

    if postgres_pool is None:
        settings = get_settings()
        try:
            postgres_pool = await asyncpg.create_pool(dsn=settings.postgres_dsn)
            logger.info("Connected to PostgreSQL")
        except Exception:
            logger.exception("Failed to connect to PostgreSQL")
            raise

    return postgres_pool


async def close_postgres() -> None:
    global postgres_pool

    if postgres_pool is not None:
        await postgres_pool.close()
        postgres_pool = None
        logger.info("PostgreSQL connection pool closed")


async def init_redis() -> Redis:
    global redis_client

    if redis_client is None:
        settings = get_settings()
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info(
                "Connected to Redis",
                redis_host=settings.redis_host,
                redis_port=settings.redis_port,
                redis_db=settings.redis_db,
            )
        except RedisError:
            logger.exception("Failed to connect to Redis")
            raise

    return redis_client


async def get_redis_client() -> Redis:
    if redis_client is None:
        return await init_redis()

    return redis_client


async def close_redis() -> None:
    global redis_client

    if redis_client is not None:
        await redis_client.aclose()
        redis_client = None
        logger.info("Redis connection closed")
