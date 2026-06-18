"""FastAPI app factory and HTTP routes for the chatbot.

Endpoints
---------
- ``GET  /health``                      - liveness check
- ``GET  /api/telegram/webhook``         - Telegram webhook setup handshake
- ``POST /api/telegram/webhook``         - incoming user message
- ``POST /api/crawl``                    - admin: crawl + index a website
- ``DELETE /api/crawl``                  - admin: drop the indexed KB
- ``GET  /api/crawl/status``            - admin: KB stats

Lifespan
--------
The FastAPI lifespan initialises two long-lived clients:

- ``redis_client`` (module-level) used by ``src.session``
- ``qdrant_client`` and ``openai_client`` (module-level) used by
  ``src.rag``

Other modules import those globals instead of opening new
connections on every call.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import sys
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from redis.asyncio import Redis

from src.api.telegram_webhook import router as telegram_router
from src.config import get_settings
from src.crawler import crawl_full_website, CrawlResult
from src.rag import (
    init_openai,
    init_qdrant,
    index_crawl_results,
    reset_collection,
)


# ---------------------------------------------------------------------------
# Module-level resources
# ---------------------------------------------------------------------------

redis_client: Optional[Redis] = None
qdrant_client: Optional[QdrantClient] = None
openai_client: Optional[Any] = None  # AsyncOpenAI; avoid module-level import


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global redis_client, qdrant_client, openai_client

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

    # RAG clients are idempotent; safe to call on every startup.
    qdrant_client = init_qdrant()
    openai_client = init_openai()

    try:
        yield
    finally:
        if redis_client is not None:
            await redis_client.aclose()
            redis_client = None
        qdrant_client = None
        openai_client = None
        logger.info("Resources released")


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
    format="{time:HH:mm:ss} | {level} | {message}",
    colorize=False,
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


# ---------------------------------------------------------------------------
# Pydantic request/response models for the admin endpoints
# ---------------------------------------------------------------------------


class CrawlRequest(BaseModel):
    url: str = Field(..., description="Absolute URL of the website to crawl")
    max_pages: int = Field(default=20, ge=1, le=200)
    replace: bool = Field(default=True, description="Drop the existing KB first")


class CrawlResponse(BaseModel):
    pages_crawled: int
    pages_indexed: int
    chunks_indexed: int
    failures: int
    max_pages: int
    url: str


class CrawlStatusResponse(BaseModel):
    collection: str
    points_count: int
    indexed_vectors: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.post("/api/crawl", response_model=CrawlResponse, tags=["admin"])
async def crawl_and_index(request: CrawlRequest) -> CrawlResponse:
    """Crawl a website and index the result into the RAG KB."""
    if request.max_pages > 200:
        raise HTTPException(
            status_code=400, detail="max_pages must be <= 200"
        )

    logger.info(
        "Crawl request",
        url=request.url,
        max_pages=request.max_pages,
        replace=request.replace,
    )

    results: list[CrawlResult] = await crawl_full_website(
        request.url,
        max_pages=request.max_pages,
    )
    summary = await index_crawl_results(results, replace=request.replace)

    pages_crawled = len(results)
    pages_indexed = summary["pages"]
    chunks_indexed = summary["chunks"]
    failures = summary["failures"]

    return CrawlResponse(
        pages_crawled=pages_crawled,
        pages_indexed=pages_indexed,
        chunks_indexed=chunks_indexed,
        failures=failures,
        max_pages=request.max_pages,
        url=request.url,
    )


@app.delete("/api/crawl", tags=["admin"])
async def clear_index() -> dict[str, str]:
    """Drop the RAG knowledge base. Used for re-crawls and debugging."""
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant not ready")
    reset_collection(qdrant_client)
    return {"status": "cleared"}


@app.get("/api/crawl/status", response_model=CrawlStatusResponse, tags=["admin"])
async def crawl_status() -> CrawlStatusResponse:
    """Return the current size of the RAG knowledge base."""
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant not ready")
    info = qdrant_client.get_collection(collection_name=settings.qdrant_collection)
    return CrawlStatusResponse(
        collection=settings.qdrant_collection,
        points_count=info.points_count,
        indexed_vectors=info.indexed_vectors_count or 0,
    )
