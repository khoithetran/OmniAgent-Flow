"""Hybrid Search & Reciprocal Rank Fusion (RRF) module.

Combines Dense Vector Search (semantic similarity) with Sparse BM25 Search
(exact keyword match) to improve retrieval recall and precision.

Mechanics:
1. Sparse BM25 Search: Tokens are lowercased and split. BM25 Okapi scores candidate chunks.
2. Reciprocal Rank Fusion (RRF): Merges dense and sparse rank lists using the formula:
       RRF_Score(d) = sum( 1 / (k + rank(d)) )
   where k = 60 (standard constant).

This ensures exact matches (e.g. product IDs, names, codes) and semantic matches
both receive high priority in the candidate pool.
"""

from __future__ import annotations

import re
from typing import Any
from loguru import logger


def _tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def bm25_search(
    query: str,
    corpus_chunks: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Perform BM25 keyword search over in-memory corpus chunks.

    Parameters
    ----------
    query:
        Search query string.
    corpus_chunks:
        List of candidate dicts. Each must contain a 'text' key.
    top_k:
        Number of top BM25 results to return.

    Returns
    -------
    list[dict[str, Any]]
        Top matching dicts with added 'bm25_score'.
    """
    if not corpus_chunks or not query.strip():
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    tokenized_corpus = [_tokenize(chunk.get("text", "")) for chunk in corpus_chunks]

    try:
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        scored_results = []
        for chunk, score in zip(corpus_chunks, scores):
            if score > 0:
                item = dict(chunk)
                item["bm25_score"] = float(score)
                scored_results.append(item)

        scored_results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_results[:top_k]

    except ImportError:
        logger.warning("rank_bm25 not installed, using term frequency fallback")
        # Simple term frequency fallback
        scored_results = []
        query_set = set(tokenized_query)
        for chunk, doc_tokens in zip(corpus_chunks, tokenized_corpus):
            if not doc_tokens:
                continue
            tf = sum(1 for t in doc_tokens if t in query_set) / len(doc_tokens)
            if tf > 0:
                item = dict(chunk)
                item["bm25_score"] = float(tf)
                scored_results.append(item)

        scored_results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_results[:top_k]


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    dense_results:
        Ranked list of dicts from vector search.
    sparse_results:
        Ranked list of dicts from BM25 search.
    k:
        RRF smoothing constant (default 60).
    top_k:
        Maximum number of fused items to return.

    Returns
    -------
    list[dict[str, Any]]
        Fused dicts sorted by RRF score (highest first), with added 'rrf_score'.
    """
    scores: dict[str, float] = {}
    item_map: dict[str, dict[str, Any]] = {}

    def _get_key(item: dict[str, Any]) -> str:
        # Deduplicate based on text content hash or snippet
        text = item.get("text", "").strip()
        return text[:100]  # use first 100 chars as key

    # Process dense results
    for rank, item in enumerate(dense_results, start=1):
        key = _get_key(item)
        rrf_val = 1.0 / (k + rank)
        scores[key] = scores.get(key, 0.0) + rrf_val
        item_map[key] = item

    # Process sparse results
    for rank, item in enumerate(sparse_results, start=1):
        key = _get_key(item)
        rrf_val = 1.0 / (k + rank)
        scores[key] = scores.get(key, 0.0) + rrf_val
        if key not in item_map:
            item_map[key] = item

    # Build final list
    fused = []
    for key, rrf_score in scores.items():
        entry = dict(item_map[key])
        entry["rrf_score"] = rrf_score
        fused.append(entry)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    logger.info(
        "RRF Fusion completed",
        dense_count=len(dense_results),
        sparse_count=len(sparse_results),
        fused_total=len(fused),
        returning=min(top_k, len(fused)),
    )
    return fused[:top_k]
