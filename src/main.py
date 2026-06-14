from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.webhook import router as webhook_router
from src.config import get_settings
from src.database import close_postgres, close_redis, init_redis
from src.services.conversation_service import init_conversation_schema


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_redis()
    await init_conversation_schema()
    yield
    await close_postgres()
    await close_redis()


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
app.include_router(webhook_router, prefix="/api")


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
