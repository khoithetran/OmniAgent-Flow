from collections.abc import Callable

from src.services.rag_service import (
    KnowledgeDocument,
    create_hash_embedding,
    hybrid_rank_documents,
    rerank_results,
)


def test_hash_embedding_has_stable_dimension() -> None:
    embedding = create_hash_embedding("facebook crm automation", size=16)

    assert len(embedding) == 16
    assert sum(value * value for value in embedding) > 0


def test_hybrid_rank_documents_combines_dense_and_bm25_scores() -> None:
    documents = [
        KnowledgeDocument(
            id="doc_facebook",
            content="Facebook Messenger CRM automation for customer support",
        ),
        KnowledgeDocument(
            id="doc_billing",
            content="Billing policy and invoice export guide",
        ),
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


def test_rerank_results_falls_back_when_disabled() -> None:
    results = [
        KnowledgeDocument(id="doc_1", content="RAG with Qdrant")
    ]
    ranked_results = hybrid_rank_documents(
        query="qdrant",
        documents=results,
        dense_scores={"doc_1": 1.0},
        limit=1,
    )

    reranked = rerank_results("qdrant", ranked_results, limit=1)

    assert reranked[0].id == "doc_1"


def main() -> None:
    tests: list[Callable[[], None]] = [
        test_hash_embedding_has_stable_dimension,
        test_hybrid_rank_documents_combines_dense_and_bm25_scores,
        test_rerank_results_falls_back_when_disabled,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    main()
