"""Shared pytest fixtures for OmniAgent Flow tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear get_settings lru_cache before every test.

    Several test modules (test_rag.py, test_session.py) patch
    get_settings and rely on a fresh cache. Without this fixture,
    test ordering can leave a stale cached value.
    """
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_redis():
    """A mock redis.asyncio.Redis client that stores data in memory.

    The store dict is shared between all async operations so that
    e.g. save_message (which calls rpush/expire) writes to the same
    backing store that get_history/lrange reads from.

    Key conventions:
    - Scalar keys (e.g. "cache:..."): store[str] = JSON string or bytes
    - List keys  (e.g. "session:..."): store[str] = list[str]
    """
    store: dict[str, list[str] | str] = {}

    client = AsyncMock()

    async def _ping():
        return True

    async def _get(key: str):
        val = store.get(key)
        if val is None:
            return None
        # Scalar keys stored as str; list keys stored as list (handled by lrange)
        return val if isinstance(val, str) else None

    async def _set(key: str, value, ex=None):
        store[key] = value.decode("utf-8") if isinstance(value, bytes) else value

    async def _delete(key: str):
        store.pop(key, None)

    async def _rpush(key: str, *values):
        # Each message is stored as a separate element in a list (real Redis behaviour).
        if key not in store:
            store[key] = []
        elif not isinstance(store[key], list):
            store[key] = []
        for v in values:
            store[key].append(str(v))

    async def _ltrim(key: str, start: int, end: int):
        pass  # no-op for tests; we read the full list directly

    async def _expire(key: str, seconds: int):
        pass  # no-op for tests; TTL is enforced by the fixture

    async def _getdel(key: str):
        return store.pop(key, None)

    async def _lrange(key: str, start: int, end: int):
        val = store.get(key)
        if not isinstance(val, list):
            # A scalar was stored under this key (should not happen for session lists,
            # but defensively return empty so callers don't try to JSON-parse a scalar).
            return []
        if end == -1:
            return val[start:]
        return val[start : end + 1]

    client.ping = _ping
    client.get = _get
    client.set = _set
    client.delete = _delete
    client.rpush = _rpush
    client.lrange = _lrange
    client.ltrim = _ltrim
    client.expire = _expire
    client.getdel = _getdel

    return client, store


@pytest.fixture
def mock_settings():
    """Minimal settings object that satisfies get_settings()."""
    s = MagicMock()
    s.session_ttl_seconds = 1800
    s.pending_crawl_ttl_seconds = 300
    s.cache_ttl_seconds = 3600
    s.openai_api_key_value = "sk-test-key"
    s.openai_model = "gpt-4o-mini"
    s.qdrant_collection = "test_collection"
    s.rag_embedding_size = 1536
    s.rag_top_k = 3
    s.rag_chunk_size = 1000
    return s
