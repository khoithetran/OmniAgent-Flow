"""Tests for src/rag.py — RAG pipeline (chunking, embedding, search)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag import (
    RagSearchResult,
    format_context,
    _split_oversize_text,
    _chunk_payload,
    _ensure_collection,
    _upsert_points,
)
from src.config import Settings


# ---------------------------------------------------------------------------
# _split_oversize_text
# ---------------------------------------------------------------------------

class TestSplitOversizeText:
    def test_short_text_returns_unchanged(self):
        text = "a" * 100
        assert _split_oversize_text(text) == [text]

    def test_exact_max_returns_single(self):
        text = "a" * 7000
        assert len(_split_oversize_text(text, max_chars=7000)) == 1

    def test_longer_text_splits(self):
        text = "a" * 15000
        chunks = _split_oversize_text(text, max_chars=7000)
        assert len(chunks) == 3

    def test_max_chars_respected(self):
        text = "x" * 10000
        for chunk in _split_oversize_text(text, max_chars=7000):
            assert len(chunk) <= 7000


# ---------------------------------------------------------------------------
# _chunk_payload
# ---------------------------------------------------------------------------

class TestChunkPayload:
    def test_returns_correct_dict(self):
        payload = _chunk_payload(
            chunk="Hello world",
            url="https://example.com/",
            title="Example",
            page_chunk_index=2,
            page_chunk_total=10,
        )
        assert payload["text"] == "Hello world"
        assert payload["url"] == "https://example.com/"
        assert payload["title"] == "Example"
        assert payload["page_chunk_index"] == 2
        assert payload["page_chunk_total"] == 10


# ---------------------------------------------------------------------------
# _upsert_points
# ---------------------------------------------------------------------------

class TestUpsertPoints:
    def test_calls_qdrant_upsert(self):
        mock_settings = MagicMock()
        mock_settings.qdrant_collection = "test_collection"
        mock_client = MagicMock()
        with patch("src.rag.get_settings", return_value=mock_settings):
            _upsert_points(
                mock_client,
                ids=["id1", "id2"],
                vectors=[[0.1] * 1536, [0.2] * 1536],
                payloads=[{"text": "a"}, {"text": "b"}],
            )
        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_collection"
        assert len(call_kwargs["points"]) == 2


# ---------------------------------------------------------------------------
# _ensure_collection
# ---------------------------------------------------------------------------

class TestEnsureCollection:
    def test_creates_collection_if_missing(self):
        mock_settings = MagicMock()
        mock_settings.qdrant_collection = "new_collection"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []

        with patch("src.rag.get_settings", return_value=mock_settings):
            with patch("src.rag.qmodels"):
                _ensure_collection(mock_client, vector_size=1536)

        mock_client.create_collection.assert_called_once()

    def test_skips_if_collection_exists(self):
        from src.config import Settings

        mock_settings = Settings.model_construct(qdrant_collection="existing_collection")
        mock_client = MagicMock()

        # Use setattr instead of MagicMock(name=...) because unittest.mock
        # shadows the .name attribute with the special _mock_name.
        collection_mock = MagicMock()
        setattr(collection_mock, "name", "existing_collection")
        mock_client.get_collections.return_value.collections = [collection_mock]

        from src.rag import get_settings as rag_gs

        rag_gs.cache_clear()
        with patch("src.rag.get_settings", return_value=mock_settings):
            _ensure_collection(mock_client, vector_size=1536)

        mock_client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# RagSearchResult
# ---------------------------------------------------------------------------

class TestRagSearchResult:
    def test_attributes(self):
        r = RagSearchResult(
            text="Some text",
            url="https://example.com",
            title="Example",
            score=0.85,
            page_chunk_index=1,
            page_chunk_total=5,
        )
        assert r.text == "Some text"
        assert r.score == 0.85
        assert r.page_chunk_index == 1
        assert r.page_chunk_total == 5

    def test_to_dict(self):
        r = RagSearchResult(
            text="Text",
            url="https://example.com",
            title="Title",
            score=0.5,
        )
        d = r.to_dict()
        assert d["score"] == 0.5
        assert d["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_empty_results(self):
        assert format_context([]) == ""

    def test_single_hit(self):
        hits = [
            RagSearchResult(
                text="AI models benchmark",
                url="https://example.com/models",
                title="Models",
                score=0.8,
            )
        ]
        ctx = format_context(hits)
        assert "Models" in ctx
        assert "example.com/models" in ctx
        assert "AI models benchmark" in ctx
        assert "[1]" in ctx

    def test_max_chars_truncates(self):
        hits = [
            RagSearchResult(
                text="a" * 5000,
                url="https://example.com/",
                title="Title",
                score=0.8,
            )
            for _ in range(5)
        ]
        ctx = format_context(hits, max_chars=2000)
        assert len(ctx) <= 2000 + 10  # small fudge for ellipsis

    def test_block_ends_with_ellipsis_if_truncated(self):
        hits = [
            RagSearchResult(
                text="a" * 3000,
                url="https://example.com/",
                title="Title",
                score=0.8,
            )
        ]
        ctx = format_context(hits, max_chars=100)
        assert ctx.endswith("...")

    def test_multiple_hits_separated_by_delimiter(self):
        hits = [
            RagSearchResult(text=f"Text {i}", url=f"https://example.com/{i}", title=f"Title {i}", score=0.7)
            for i in range(3)
        ]
        ctx = format_context(hits)
        # Should contain delimiters between blocks
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "[3]" in ctx
