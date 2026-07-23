"""Cross-Encoder Reranking module for two-stage retrieval.

Stage 1 (Vector/BM25 Search) retrieves a broad candidate pool (e.g. Top 20).
Stage 2 (Cross-Encoder) computes deep cross-attention scores between the query
and each candidate chunk to select the most relevant Top K (e.g. Top 3).

Model used:
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``
    - Lightweight (~22MB)
    - Runs fast on CPU
    - High accuracy for passage ranking

Design Notes:
- The CrossEncoder model is lazy-loaded on first call to avoid slowing down
  application startup.
- Fallback mechanism: If sentence-transformers is unavailable, a lightweight
  keyword overlap fallback scorer is used so the RAG pipeline never crashes.
"""

from __future__ import annotations

import re
from typing import Any
from loguru import logger

_RERANKER_MODEL = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_IS_DISABLED = False


def _get_model():
    """Lazy load the CrossEncoder model."""
    global _RERANKER_MODEL, _IS_DISABLED
    if _IS_DISABLED:
        return None
    if _RERANKER_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder reranker model", model=_MODEL_NAME)
            _RERANKER_MODEL = CrossEncoder(_MODEL_NAME, max_length=512)
            logger.info("CrossEncoder reranker loaded successfully")
        except Exception as exc:
            logger.warning(
                "Failed to load CrossEncoder model, fallback scorer will be used",
                error=str(exc),
            )
            _IS_DISABLED = True
            return None
    return _RERANKER_MODEL


def _fallback_keyword_score(query: str, text: str) -> float:
    """Fallback scoring based on keyword overlap ratio when model is unavailable."""
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    if not query_words:
        return 0.0
    overlap = query_words.intersection(text_words)
    return len(overlap) / len(query_words)


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rerank candidates based on semantic relevance to query.

    Parameters
    ----------
    query:
        The search query string.
    candidates:
        List of dicts representing candidate chunks. Must contain a 'text' key.
        Supported keys: 'text', 'score', 'url', 'title', etc.
    top_k:
        Number of top reranked items to return.

    Returns
    -------
    list[dict[str, Any]]
        Re-ordered candidate dicts with added 'rerank_score'.
    """
    if not candidates:
        return []

    if len(candidates) <= 1:
        for c in candidates:
            c["rerank_score"] = float(c.get("score", 1.0))
        return candidates[:top_k]

    model = _get_model()

    if model is not None:
        try:
            pairs = [[query, item.get("text", "")] for item in candidates]
            scores = model.predict(pairs)
            
            # Attach rerank score and sort
            scored_candidates = []
            for item, score in zip(candidates, scores):
                item_copy = dict(item)
                item_copy["rerank_score"] = float(score)
                scored_candidates.append(item_copy)

            scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            logger.info(
                "Rerank completed using CrossEncoder",
                input_count=len(candidates),
                output_count=min(top_k, len(scored_candidates)),
            )
            return scored_candidates[:top_k]
        except Exception as exc:
            logger.warning("CrossEncoder predict failed, falling back", error=str(exc))

    # Fallback path
    scored_candidates = []
    for item in candidates:
        item_copy = dict(item)
        fallback_s = _fallback_keyword_score(query, item_copy.get("text", ""))
        item_copy["rerank_score"] = float(item_copy.get("score", 0.0)) + fallback_s
        scored_candidates.append(item_copy)

    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_candidates[:top_k]
