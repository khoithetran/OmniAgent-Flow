"""Tests for src/session.py — Redis-backed session state."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src import session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestNormalizeForCache:
    def test_lowercase(self):
        assert session._normalize_for_cache("Công Ty A") == "công ty a"

    def test_strip(self):
        assert session._normalize_for_cache("  hello world  ") == "hello world"

    def test_collapse_whitespace(self):
        assert session._normalize_for_cache("hello   world\n\nfoo") == "hello world foo"

    def test_combined(self):
        assert session._normalize_for_cache("  Công Ty   ABC  ") == "công ty abc"


class TestCacheKey:
    def test_format(self):
        assert session._cache_key("hello world") == "cache:hello world"


# ---------------------------------------------------------------------------
# Cache get/set — use unittest.mock.patch directly
# ---------------------------------------------------------------------------

class TestCacheGetSet:
    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, mock_redis, mock_settings):
        client, store = mock_redis
        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                result = await session.cache_get("any question")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_reply(self, mock_redis, mock_settings):
        client, store = mock_redis
        payload = json.dumps({"reply": "cached answer", "created_at": "2024-01-01T00:00:00Z"})
        store["cache:test question"] = payload

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                result = await session.cache_get("test question")
        assert result == "cached answer"

    @pytest.mark.asyncio
    async def test_cache_hit_case_insensitive(self, mock_redis, mock_settings):
        client, store = mock_redis
        payload = json.dumps({"reply": "cached answer", "created_at": "2024-01-01T00:00:00Z"})
        store["cache:công ty abc"] = payload

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                result = await session.cache_get("Công Ty ABC")
        assert result == "cached answer"

    @pytest.mark.asyncio
    async def test_cache_miss_on_invalid_json(self, mock_redis, mock_settings):
        client, store = mock_redis
        store["cache:bad json"] = b"not valid json"

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                result = await session.cache_get("bad json")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_stores_normalized_key(self, mock_redis, mock_settings):
        client, store = mock_redis

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                await session.cache_set("  Công Ty ABC  ", "reply text")

        assert "cache:công ty abc" in store
        stored = json.loads(store["cache:công ty abc"])
        assert stored["reply"] == "reply text"
        assert "created_at" in stored

    @pytest.mark.asyncio
    async def test_cache_set_non_fatal_on_redis_error(self, mock_settings):
        """cache_set must swallow Redis errors and not raise."""
        async def bad_set(key, value, ex=None):
            raise OSError("Redis unavailable")

        mock_client = AsyncMock()
        mock_client.set = bad_set
        mock_client.get = AsyncMock(return_value=None)
        mock_client.delete = AsyncMock()

        with patch("src.session._get_client", return_value=mock_client):
            with patch("src.session.get_settings", return_value=mock_settings):
                # Must not raise — error is logged but swallowed
                await session.cache_set("question", "answer")


# ---------------------------------------------------------------------------
# Pending crawl
# ---------------------------------------------------------------------------

class TestPendingCrawl:
    @pytest.mark.asyncio
    async def test_set_pending_crawl(self, mock_redis, mock_settings):
        client, store = mock_redis

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                await session.set_pending_crawl("sender123", "Acme Corp")

        assert "pending_crawl:sender123" in store
        stored = json.loads(store["pending_crawl:sender123"])
        assert stored["company"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_pop_pending_crawl_removes_marker(self, mock_redis, mock_settings):
        client, store = mock_redis
        payload = json.dumps({"company": "Acme Corp", "created_at": "2024-01-01T00:00:00Z"})
        store["pending_crawl:sender123"] = payload

        async def _getdel(key):
            return store.pop(key, None)

        client.getdel = _getdel

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                result = await session.pop_pending_crawl("sender123")

        assert result is not None
        assert result["company"] == "Acme Corp"
        assert "pending_crawl:sender123" not in store

    @pytest.mark.asyncio
    async def test_peek_pending_crawl_keeps_marker(self, mock_redis, mock_settings):
        client, store = mock_redis
        payload = json.dumps({"company": "Acme Corp", "created_at": "2024-01-01T00:00:00Z"})
        store["pending_crawl:sender456"] = payload

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                result = await session.peek_pending_crawl("sender456")

        assert result is not None
        assert result["company"] == "Acme Corp"
        # Marker still exists
        assert "pending_crawl:sender456" in store

    @pytest.mark.asyncio
    async def test_pop_pending_crawl_returns_none_when_absent(self, mock_redis, mock_settings):
        client, store = mock_redis
        # Ensure getdel returns None (simulating absent key) in this fresh mock.
        client.getdel = AsyncMock(return_value=None)
        client.get = AsyncMock(return_value=None)

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                result = await session.pop_pending_crawl("unknown")

        assert result is None


# ---------------------------------------------------------------------------
# Email capture
# ---------------------------------------------------------------------------

class TestEmailCapture:
    @pytest.mark.asyncio
    async def test_save_and_get_email(self, mock_redis, mock_settings):
        client, store = mock_redis

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                await session.save_email("sender789", "test@example.com")

        assert "email:sender789" in store
        stored = json.loads(store["email:sender789"])
        assert stored["email"] == "test@example.com"

        async def _get(key):
            return store.get(key)

        client.get = _get

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                result = await session.get_email("sender789")

        assert result == "test@example.com"

    @pytest.mark.asyncio
    async def test_set_and_pop_pending_email(self, mock_redis, mock_settings):
        client, store = mock_redis

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                await session.set_pending_email("sender999")

        assert "pending_email:sender999" in store

        async def _getdel(key):
            return store.pop(key, None)

        client.getdel = _getdel

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                result = await session.pop_pending_email("sender999")

        assert result is not None
        assert "pending_email:sender999" not in store


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

class TestChatHistory:
    @pytest.mark.asyncio
    async def test_save_and_get_history(self, mock_redis, mock_settings):
        client, store = mock_redis

        # Use a plain function instead of AsyncMock so all methods resolve
        # to the real async functions defined in the fixture.
        async def get_client():
            return client

        with patch("src.session._get_client", get_client):
            with patch("src.session.get_settings", return_value=mock_settings):
                await session.save_message("user1", "user", "Hello")
                await session.save_message("user1", "assistant", "Hi there")

        # Both messages should be stored as a list of JSON strings (real Redis behaviour).
        raw = store.get("session:user1", [])
        assert isinstance(raw, list)
        assert len(raw) == 2

        # Now read back via get_history — same client, same store.
        with patch("src.session._get_client", get_client):
            with patch("src.session.get_settings", return_value=mock_settings):
                history = await session.get_history("user1")

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there"

    @pytest.mark.asyncio
    async def test_get_history_skips_invalid_json(self, mock_redis, mock_settings):
        client, store = mock_redis
        # Store an invalid JSON string directly.
        store["session:user2"] = "not json"

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                history = await session.get_history("user2")

        assert history == []

    @pytest.mark.asyncio
    async def test_clear_history(self, mock_redis, mock_settings):
        client, store = mock_redis
        store["session:user3"] = '{"role":"user","content":"hi"}'

        with patch("src.session._get_client", return_value=client):
            with patch("src.session.get_settings", return_value=mock_settings):
                await session.clear_history("user3")

        assert "session:user3" not in store
