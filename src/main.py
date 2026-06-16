from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import sys
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from redis.asyncio import Redis

from src.api.telegram_webhook import router as telegram_router
from src.config import get_settings


# ---------------------------------------------------------------------------
# Module-level resources
# ---------------------------------------------------------------------------
#
# A single Redis client and a single Qdrant client are created at startup
# and reused for the lifetime of the process. Other modules import the
# globals below instead of opening new connections on every call.

redis_client: Optional[Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global redis_client

    settings = get_settings()
    redis_client = Redis.from_url(
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
        decode_responses=True,
    )
    await redis_client.ping()
    logger.info(
        "Connected to Redis",
        redis_host=settings.redis_host,
        redis_port=settings.redis_port,
        redis_db=settings.redis_db,
    )

    try:
        yield
    finally:
        if redis_client is not None:
            await redis_client.aclose()
            redis_client = None
            logger.info("Redis connection closed")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

settings = get_settings()

logger.remove()
logger.add(
    sys.stdout,
    backtrace=True,
    diagnose=settings.app_env == "development",
    enqueue=True,
    serialize=True,
)

app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(telegram_router, prefix="/api")


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
