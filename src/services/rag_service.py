from collections import Counter
from math import log, sqrt
from re import findall
from typing import Any
import hashlib
import uuid

from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_settings


class KnowledgeDocument(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagSearchResult(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0


def tokenize_text(text: str) -> list[str]:
    return findall(r"[\w]+", text.casefold())


def create_hash_embedding(text: str, size: int) -> list[float]:
    if size < 1:
        raise ValueError("embedding size must be greater than 0")

    vector = [0.0] * size
    for token in tokenize_text(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % size
        vector[index] += 1.0

    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def calculate_bm25_scores(
    query: str,
    documents: list[KnowledgeDocument],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    query_tokens = tokenize_text(query)
    corpus_tokens = [tokenize_text(document.content) for document in documents]

    if not query_tokens or not corpus_tokens:
        return {document.id: 0.0 for document in documents}

    doc_count = len(corpus_tokens)
    average_length = sum(len(tokens) for tokens in corpus_tokens) / doc_count or 1.0
    document_frequency: Counter[str] = Counter()

    for tokens in corpus_tokens:
        document_frequency.update(set(tokens))

    scores: dict[str, float] = {}
    for document, tokens in zip(documents, corpus_tokens):
        token_counts = Counter(tokens)
        doc_length = len(tokens) or 1
        score = 0.0

        for token in query_tokens:
            frequency = token_counts[token]
            if frequency == 0:
                continue

            idf = log(
                (doc_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
                + 1
            )
            denominator = frequency + k1 * (1 - b + b * doc_length / average_length)
            score += idf * (frequency * (k1 + 1)) / denominator

        scores[document.id] = score

    return scores


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {key: 1.0 if value > 0 else 0.0 for key, value in scores.items()}

    return {
        key: (value - min_score) / (max_score - min_score)
        for key, value in scores.items()
    }


def hybrid_rank_documents(
    query: str,
    documents: list[KnowledgeDocument],
    dense_scores: dict[str, float] | None = None,
    limit: int = 5,
    dense_weight: float | None = None,
    bm25_weight: float | None = None,
) -> list[RagSearchResult]:
    settings = get_settings()
    resolved_dense_weight = (
        dense_weight if dense_weight is not None else settings.rag_dense_weight
    )
    resolved_bm25_weight = (
        bm25_weight if bm25_weight is not None else settings.rag_bm25_weight
    )
    resolved_dense_scores = dense_scores or {}
    bm25_scores = calculate_bm25_scores(query, documents)
    normalized_dense = normalize_scores(resolved_dense_scores)
    normalized_bm25 = normalize_scores(bm25_scores)

    results: list[RagSearchResult] = []
    for document in documents:
        dense_score = normalized_dense.get(document.id, 0.0)
        bm25_score = normalized_bm25.get(document.id, 0.0)
        hybrid_score = (
            resolved_dense_weight * dense_score
            + resolved_bm25_weight * bm25_score
        )
        results.append(
            RagSearchResult(
                id=document.id,
                content=document.content,
                metadata=document.metadata,
                dense_score=dense_score,
                bm25_score=bm25_score,
                hybrid_score=hybrid_score,
                final_score=hybrid_score,
            )
        )

    results.sort(key=lambda result: result.final_score, reverse=True)
    return results[:limit]


def rerank_results(
    query: str,
    results: list[RagSearchResult],
    limit: int = 5,
) -> list[RagSearchResult]:
    settings = get_settings()
    if not settings.rag_enable_reranker or not results:
        return results[:limit]

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        reranker = TextCrossEncoder(model_name=settings.rag_reranker_model)
        raw_scores = list(
            reranker.rerank(query, [result.content for result in results])
        )
        rerank_scores = {
            result.id: float(getattr(score, "score", score))
            for result, score in zip(results, raw_scores)
        }
        normalized_scores = normalize_scores(rerank_scores)
        reranked_results = [
            result.model_copy(
                update={
                    "rerank_score": rerank_scores.get(result.id, 0.0),
                    "final_score": normalized_scores.get(result.id, result.final_score),
                }
            )
            for result in results
        ]
        reranked_results.sort(key=lambda result: result.final_score, reverse=True)
        logger.info(
            "Reranked RAG candidates with cross-encoder",
            model=settings.rag_reranker_model,
            candidate_count=len(results),
        )
        return reranked_results[:limit]
    except Exception:
        logger.exception(
            "Failed to rerank RAG candidates; falling back to hybrid scores",
            model=settings.rag_reranker_model,
        )
        return results[:limit]


def get_qdrant_client() -> Any:
    settings = get_settings()
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.qdrant_url)


def ensure_rag_collection(client: Any | None = None) -> None:
    settings = get_settings()
    qdrant_client = client or get_qdrant_client()

    try:
        qdrant_client.get_collection(settings.qdrant_collection)
        return
    except Exception:
        logger.info(
            "Creating Qdrant RAG collection",
            collection=settings.qdrant_collection,
            vector_size=settings.rag_embedding_size,
        )

    from qdrant_client import models

    qdrant_client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            "dense": models.VectorParams(
                size=settings.rag_embedding_size,
                distance=models.Distance.COSINE,
            )
        },
    )


def _qdrant_point_id(document_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, document_id))


def index_knowledge_documents(documents: list[KnowledgeDocument]) -> int:
    if not documents:
        return 0

    settings = get_settings()
    qdrant_client = get_qdrant_client()
    ensure_rag_collection(qdrant_client)

    from qdrant_client import models

    points = [
        models.PointStruct(
            id=_qdrant_point_id(document.id),
            vector={
                "dense": create_hash_embedding(
                    document.content,
                    settings.rag_embedding_size,
                )
            },
            payload={
                "source_id": document.id,
                "content": document.content,
                "metadata": document.metadata,
            },
        )
        for document in documents
    ]
    qdrant_client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,
    )
    logger.info(
        "Indexed knowledge documents into Qdrant",
        collection=settings.qdrant_collection,
        document_count=len(documents),
    )
    return len(documents)


def _document_from_payload(payload: dict[str, Any] | None) -> KnowledgeDocument | None:
    if not payload:
        return None

    source_id = payload.get("source_id")
    content = payload.get("content")
    metadata = payload.get("metadata", {})

    if not isinstance(source_id, str) or not isinstance(content, str):
        return None
    if not isinstance(metadata, dict):
        metadata = {}

    return KnowledgeDocument(id=source_id, content=content, metadata=metadata)


def _scroll_knowledge_documents(client: Any, limit: int) -> list[KnowledgeDocument]:
    settings = get_settings()
    documents: list[KnowledgeDocument] = []
    offset: Any | None = None

    while len(documents) < limit:
        records, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=min(100, limit - len(documents)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            document = _document_from_payload(record.payload)
            if document is not None:
                documents.append(document)

        if offset is None:
            break

    return documents


def hybrid_search_knowledge(query: str, limit: int = 5) -> list[RagSearchResult]:
    settings = get_settings()
    qdrant_client = get_qdrant_client()
    ensure_rag_collection(qdrant_client)

    candidate_limit = max(settings.rag_candidate_limit, limit)
    query_embedding = create_hash_embedding(query, settings.rag_embedding_size)
    dense_response = qdrant_client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        using="dense",
        limit=candidate_limit,
        with_payload=True,
    )

    documents_by_id: dict[str, KnowledgeDocument] = {}
    dense_scores: dict[str, float] = {}
    for point in dense_response.points:
        document = _document_from_payload(point.payload)
        if document is None:
            continue
        documents_by_id[document.id] = document
        dense_scores[document.id] = float(point.score)

    for document in _scroll_knowledge_documents(
        qdrant_client,
        settings.rag_bm25_corpus_limit,
    ):
        documents_by_id.setdefault(document.id, document)

    hybrid_results = hybrid_rank_documents(
        query=query,
        documents=list(documents_by_id.values()),
        dense_scores=dense_scores,
        limit=candidate_limit,
    )
    return rerank_results(query, hybrid_results, limit=limit)
