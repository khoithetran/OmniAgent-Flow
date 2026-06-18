"""Retrieval-Augmented Generation (RAG) layer for the Telegram chatbot.

This module owns the Qdrant vector store and the OpenAI embedding
client. It exposes three high-level operations:

1. ``index_markdown()`` - chunk a markdown document, embed each chunk
   with ``text-embedding-3-small``, upsert into Qdrant. Used by the
   crawler flow.
2. ``index_crawl_results()`` - convenience wrapper for ``CrawlResult``
   batches produced by ``src.crawler``.
3. ``search()`` - embed a query and retrieve the top-k most relevant
   chunks. Used by the chat flow to ground the LLM.

Design notes
------------
- The Qdrant client is module-level. ``src.main`` owns it via the
  FastAPI lifespan; other modules import the global to avoid opening
  a new client per request.
- We use ``text-embedding-3-small`` (1536 dims) by default. The
  collection vector size is configured in ``Settings.rag_embedding_size``
  so it stays in sync with the embedding model.
- Metadata is stored as a payload so the chat layer can show
  citations (URL + title) without a second lookup.
- We always re-create the collection on ``index_markdown(..., replace=True)``
  so re-crawling a site does not leave stale chunks. The default is
  to upsert incrementally when ``replace=False``.
"""

from __future__ import annotations

import uuid
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import get_settings
from src.simple_crawler import CrawlResult, chunk_markdown


# ---------------------------------------------------------------------------
# Module-level resources
# ---------------------------------------------------------------------------
#
# Both clients are created once at startup. ``src.main`` assigns them
# during the FastAPI lifespan; importers use the getters below which
# raise a clear error if the lifespan never ran.

qdrant_client: QdrantClient | None = None
openai_client: AsyncOpenAI | None = None
qdrant_available: bool = False


def _get_qdrant() -> QdrantClient:
    if qdrant_client is None:
        raise RuntimeError(
            "Qdrant client has not been initialised. "
            "Call init_qdrant() during FastAPI startup."
        )
    return qdrant_client


def _get_openai() -> AsyncOpenAI:
    if openai_client is None:
        raise RuntimeError(
            "OpenAI client has not been initialised. "
            "Call init_openai() during FastAPI startup."
        )
    return openai_client


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_qdrant() -> QdrantClient | None:
    """Build a Qdrant client from settings. Idempotent.

    Returns None when Qdrant is unreachable so the app still works
    in general-LLM mode (e.g. HF Spaces without a Qdrant instance).
    """
    global qdrant_client, qdrant_available
    if qdrant_client is None:
        settings = get_settings()
        try:
            qdrant_client = QdrantClient(url=settings.qdrant_url, timeout=3)
            # Quick health check
            qdrant_client.get_collections()
            qdrant_available = True
            logger.info("Qdrant client initialised", url=settings.qdrant_url)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Qdrant unavailable; RAG features disabled (general LLM mode)"
            )
            qdrant_client = None
            qdrant_available = False
    return qdrant_client


def init_openai() -> AsyncOpenAI | None:
    """Build an OpenAI client. Returns None when the key is missing.

    The chat/crawler callers must check for None and bail out
    gracefully - the RAG layer is only one of two paths in the chat
    flow, the other being a general-knowledge LLM reply.
    """
    global openai_client
    if openai_client is None:
        settings = get_settings()
        api_key = settings.openai_api_key_value
        if api_key is None:
            logger.warning("OpenAI key not configured; RAG layer disabled")
            return None
        openai_client = AsyncOpenAI(api_key=api_key)
        logger.info("OpenAI client initialised", model=settings.openai_model)
    return openai_client


def close_clients() -> None:
    """Close module-level clients. Called from the FastAPI shutdown."""
    global qdrant_client, openai_client, qdrant_available
    qdrant_client = None
    openai_client = None
    qdrant_available = False


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


def _ensure_collection(client: QdrantClient, *, vector_size: int) -> None:
    """Create the Qdrant collection if it does not exist yet.

    Uses cosine similarity because OpenAI embeddings are normalised.
    """
    settings = get_settings()
    name = settings.qdrant_collection

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return

    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
        ),
    )
    logger.info("Created Qdrant collection", collection=name, size=vector_size)


def reset_collection(client: QdrantClient) -> None:
    """Drop the configured collection so the next index starts clean."""
    settings = get_settings()
    client.delete_collection(collection_name=settings.qdrant_collection)
    logger.info("Deleted Qdrant collection", collection=settings.qdrant_collection)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings using ``text-embedding-3-small``."""
    if not texts:
        return []
    client = _get_openai()
    settings = get_settings()
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    vectors = await embed_texts([query])
    if not vectors:
        raise RuntimeError("Failed to embed query (empty response from OpenAI)")
    return vectors[0]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def _chunk_payload(
    *,
    chunk: str,
    url: str,
    title: str,
    page_chunk_index: int,
    page_chunk_total: int,
) -> dict[str, Any]:
    return {
        "text": chunk,
        "url": url,
        "title": title,
        "page_chunk_index": page_chunk_index,
        "page_chunk_total": page_chunk_total,
    }


def _upsert_points(
    client: QdrantClient,
    *,
    ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
) -> None:
    settings = get_settings()
    points = [
        qmodels.PointStruct(id=point_id, vector=vec, payload=payload)
        for point_id, vec, payload in zip(ids, vectors, payloads)
    ]
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,
    )


async def _embed_in_batches(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed ``texts`` in fixed-size batches to respect OpenAI rate limits.

    Any input that would exceed the model's per-request token limit
    is split into smaller sub-chunks first. This protects the flow
    against an over-aggressive ``chunk_size`` config or unusually
    long paragraphs that survived ``chunk_markdown``.
    """
    safe_texts: list[str] = []
    safe_index: list[int] = []
    for index, text in enumerate(texts):
        for sub in _split_oversize_text(text):
            safe_texts.append(sub)
            safe_index.append(index)

    all_vectors: list[list[float]] = []
    result: list[list[float] | None] = [None] * len(texts)
    for start in range(0, len(safe_texts), batch_size):
        batch = safe_texts[start : start + batch_size]
        batch_vectors = await embed_texts(batch)
        for vector, original_index in zip(batch_vectors, safe_index[start : start + batch_size]):
            # When a chunk is split into multiple sub-chunks, average
            # their vectors so the final representation is still a
            # single 1536-dim point. (For non-split inputs, this loop
            # runs once and the average is the vector itself.)
            existing = result[original_index]
            if existing is None:
                result[original_index] = vector
            else:
                result[original_index] = [
                    (a + b) / 2 for a, b in zip(existing, vector)
                ]
        all_vectors.extend(batch_vectors)

    return [r if r is not None else [] for r in result]


def _split_oversize_text(
    text: str,
    *,
    max_chars: int = 7000,
) -> list[str]:
    """Split ``text`` so every sub-chunk fits in one OpenAI embed call.

    The default ``max_chars`` is conservative (``text-embedding-3-small``
    accepts 8192 tokens; a 7000-char chunk is roughly 1500-2000 tokens
    for English/Vietnamese mixed text). We also keep a small overlap
    so retrieval still has context near the cut.
    """
    if len(text) <= max_chars:
        return [text]
    overlap = 200
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


async def index_markdown(
    markdown: str,
    *,
    url: str,
    title: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    replace: bool = False,
) -> int:
    """Chunk a markdown document and index it into Qdrant.

    Parameters
    ----------
    markdown:
        The full markdown body to index.
    url, title:
        Stored as payload so we can build a citation when the chat
        layer shows a chunk to the user.
    chunk_size, chunk_overlap:
        Override ``Settings.rag_chunk_size``. Overlap defaults to 10%
        of ``chunk_size``.
    replace:
        When True, the configured collection is dropped and recreated
        before indexing. Use this for ``POST /api/crawl`` to make
        re-crawls idempotent.

    Returns
    -------
    int
        Number of chunks indexed.
    """
    settings = get_settings()
    client = _get_qdrant()

    effective_chunk_size = chunk_size or settings.rag_chunk_size
    effective_overlap = chunk_overlap or max(1, effective_chunk_size // 10)

    if replace:
        reset_collection(client)
    _ensure_collection(client, vector_size=settings.rag_embedding_size)

    chunks = chunk_markdown(
        markdown,
        chunk_size=effective_chunk_size,
        overlap=effective_overlap,
    )
    if not chunks:
        logger.info("No chunks produced for indexing", url=url)
        return 0

    vectors = await _embed_in_batches(chunks)
    ids = [uuid.uuid4().hex for _ in chunks]
    payloads = [
        _chunk_payload(
            chunk=chunk,
            url=url,
            title=title,
            page_chunk_index=i,
            page_chunk_total=len(chunks),
        )
        for i, chunk in enumerate(chunks)
    ]
    _upsert_points(client, ids=ids, vectors=vectors, payloads=payloads)
    logger.info(
        "Indexed markdown into Qdrant",
        url=url,
        chunks=len(chunks),
        chunk_size=effective_chunk_size,
    )
    return len(chunks)


async def index_crawl_results(
    results: list[CrawlResult],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    replace: bool = False,
) -> dict[str, int]:
    """Index a batch of ``CrawlResult`` objects.

    Returns a summary dict shaped like
    ``{"pages": N, "chunks": M, "failures": F}`` so the caller can
    report progress back to the user.
    """
    settings = get_settings()
    client = _get_qdrant()

    if replace:
        reset_collection(client)
    _ensure_collection(client, vector_size=settings.rag_embedding_size)

    all_chunks: list[str] = []
    all_payloads: list[dict[str, Any]] = []
    failures = 0

    for result in results:
        if not result.success or not result.markdown.strip():
            failures += 1
            continue
        page_chunks = chunk_markdown(
            result.markdown,
            chunk_size=chunk_size or settings.rag_chunk_size,
            overlap=chunk_overlap or max(1, (chunk_size or settings.rag_chunk_size) // 10),
        )
        for index, chunk_text in enumerate(page_chunks):
            all_chunks.append(chunk_text)
            all_payloads.append(
                _chunk_payload(
                    chunk=chunk_text,
                    url=result.url,
                    title=result.title,
                    page_chunk_index=index,
                    page_chunk_total=len(page_chunks),
                )
            )

    if not all_chunks:
        logger.info("No chunks to index from crawl batch")
        return {"pages": 0, "chunks": 0, "failures": failures}

    vectors = await _embed_in_batches(all_chunks)
    ids = [uuid.uuid4().hex for _ in all_chunks]
    _upsert_points(client, ids=ids, vectors=vectors, payloads=all_payloads)

    summary = {
        "pages": len(results) - failures,
        "chunks": len(all_chunks),
        "failures": failures,
    }
    logger.info("Indexed crawl batch into Qdrant", **summary)
    return summary


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class RagSearchResult:
    """One hit returned by ``search()``."""

    __slots__ = ("text", "url", "title", "score", "page_chunk_index", "page_chunk_total")

    def __init__(
        self,
        text: str,
        url: str,
        title: str,
        score: float,
        page_chunk_index: int = 0,
        page_chunk_total: int = 0,
    ) -> None:
        self.text = text
        self.url = url
        self.title = title
        self.score = score
        self.page_chunk_index = page_chunk_index
        self.page_chunk_total = page_chunk_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "url": self.url,
            "title": self.title,
            "score": self.score,
            "page_chunk_index": self.page_chunk_index,
            "page_chunk_total": self.page_chunk_total,
        }


async def search(query: str, *, top_k: int | None = None) -> list[RagSearchResult]:
    """Embed ``query`` and return the top-k most similar chunks.

    Empty result is returned when the collection is empty or the
    OpenAI key is missing - both are recoverable from the chat layer.
    """
    settings = get_settings()
    client = _get_qdrant()
    k = top_k or settings.rag_top_k

    # Skip search gracefully if the collection has no points yet.
    try:
        info = client.get_collection(collection_name=settings.qdrant_collection)
    except Exception:  # noqa: BLE001
        logger.info("Qdrant collection not initialised; search returns empty")
        return []
    if info.points_count == 0:
        return []

    if openai_client is None:
        logger.warning("OpenAI key missing; search returns empty")
        return []

    if qdrant_client is None:
        logger.info("Qdrant not available; search returns empty")
        return []

    query_vector = await embed_query(query)
    raw_hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=k,
        with_payload=True,
    )

    results: list[RagSearchResult] = []
    for point in raw_hits.points:
        payload = point.payload or {}
        results.append(
            RagSearchResult(
                text=str(payload.get("text", "")),
                url=str(payload.get("url", "")),
                title=str(payload.get("title", "")),
                score=float(point.score or 0.0),
                page_chunk_index=int(payload.get("page_chunk_index", 0)),
                page_chunk_total=int(payload.get("page_chunk_total", 0)),
            )
        )
    return results


def format_context(results: list[RagSearchResult], *, max_chars: int = 6000) -> str:
    """Turn search hits into a single string for the LLM system prompt.

    Each hit is prefixed by its source URL and title so the LLM can
    attribute claims and so the user can ask for citations. The
    output is truncated to ``max_chars`` to keep the prompt under
    control on long pages.
    """
    if not results:
        return ""

    blocks: list[str] = []
    running = 0
    for index, hit in enumerate(results, start=1):
        block = (
            f"[{index}] {hit.title} ({hit.url})\n"
            f"{hit.text.strip()}"
        )
        if running + len(block) > max_chars:
            remaining = max_chars - running
            if remaining <= 0:
                break
            block = block[:remaining].rsplit(" ", 1)[0] + "..."
        blocks.append(block)
        running += len(block)
        if running >= max_chars:
            break
    return "\n\n---\n\n".join(blocks)
