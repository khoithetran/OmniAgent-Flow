"""Tests for the RAG service (hybrid retrieval + reranking)."""

from __future__ import annotations

from src.services.rag_service import (
    KnowledgeDocument,
    create_hash_embedding,
    hybrid_rank_documents,
    normalize_scores,
    rerank_results,
    tokenize_text,
)


def test_tokenize_text_is_case_insensitive_and_alphanumeric() -> None:
    tokens = tokenize_text("Facebook CRM 2024!")
    assert tokens == ["facebook", "crm", "2024"]


def test_create_hash_embedding_has_stable_dimension() -> None:
    embedding = create_hash_embedding("facebook crm automation", size=16)
    assert len(embedding) == 16
    norm = sum(value * value for value in embedding) ** 0.5
    assert 0.99 <= norm <= 1.01  # vector is L2 normalized


def test_normalize_scores_uses_min_max() -> None:
    result = normalize_scores({"a": 0.0, "b": 0.5, "c": 1.0})
    assert result == {"a": 0.0, "b": 0.5, "c": 1.0}


def test_normalize_scores_handles_constant_input() -> None:
    result = normalize_scores({"a": 0.7, "b": 0.7})
    assert result == {"a": 1.0, "b": 1.0}


def test_hybrid_rank_documents_combines_dense_and_bm25() -> None:
    documents = [
        KnowledgeDocument(id="doc_facebook", content="Facebook Messenger CRM automation"),
        KnowledgeDocument(id="doc_billing", content="Billing policy and invoice export guide"),
    ]
    dense_scores = {"doc_facebook": 0.82, "doc_billing": 0.1}

    results = hybrid_rank_documents(
        query="facebook crm automation",
        documents=documents,
        dense_scores=dense_scores,
        limit=2,
    )

    assert results[0].id == "doc_facebook"
    assert results[0].bm25_score > results[1].bm25_score
    assert results[0].final_score >= results[1].final_score


def test_rerank_results_passthrough_when_disabled() -> None:
    documents = [KnowledgeDocument(id="doc_1", content="RAG with Qdrant")]
    ranked = hybrid_rank_documents(
        query="qdrant",
        documents=documents,
        dense_scores={"doc_1": 1.0},
        limit=1,
    )

    reranked = rerank_results("qdrant", ranked, limit=1)

    assert reranked[0].id == "doc_1"
    assert reranked[0].rerank_score is None
