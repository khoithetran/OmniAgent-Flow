"""In-memory fallback for session when Redis is unavailable (e.g. HF Spaces).

This module provides a thread-safe, dict-based store that mimics the Redis
session API. It is only used as a fallback when Redis connection fails.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone
from typing import Any

from loguru import logger

# Thread-safe store
_lock = threading.RLock()
_store: dict[str, Any] = {}

# TTL tracking: key -> expiry_timestamp
_ttl: dict[str, float] = {}


def _now() -> float:
    return asyncio.get_event_loop().time()


def _is_expired(key: str) -> bool:
    if key not in _ttl:
        return False
    return _now() > _ttl[key]


def _set_with_ttl(key: str, value: Any, ttl_seconds: int | None = None) -> None:
    with _lock:
        _store[key] = value
        if ttl_seconds is not None:
            _ttl[key] = _now() + ttl_seconds
        elif key in _ttl:
            del _ttl[key]


def _get(key: str) -> Any | None:
    with _lock:
        if key not in _store:
            return None
        if _is_expired(key):
            del _store[key]
            _ttl.pop(key, None)
            return None
        return _store[key]


def _delete(key: str) -> None:
    with _lock:
        _store.pop(key, None)
        _ttl.pop(key, None)


def _rpush(key: str, *values: str) -> None:
    with _lock:
        if key not in _store:
            _store[key] = []
        elif not isinstance(_store[key], list):
            _store[key] = []
        for v in values:
            _store[key].append(v)


def _lrange(key: str, start: int, end: int) -> list[str]:
    with _lock:
        val = _store.get(key)
        if not isinstance(val, list):
            return []
        if end == -1:
            return val[start:]
        return val[start : end + 1]


def _ltrim(key: str, start: int, end: int) -> None:
    # Not needed for in-memory fallback
    pass


def _expire(key: str, seconds: int) -> None:
    with _lock:
        if key in _store:
            _ttl[key] = _now() + seconds


def _getdel(key: str) -> str | None:
    with _lock:
        val = _store.pop(key, None)
        _ttl.pop(key, None)
        if isinstance(val, str):
            return val
        return None


# -------------------------------------------------------------------
# Public wrappers (async, mimicking redis.asyncio.Redis)
# -------------------------------------------------------------------


class InMemoryClient:
    """Async wrapper around the thread-safe in-memory store."""

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> str | None:
        return _get(key)

    async def set(
        self, key: str, value: str | bytes, ex: int | None = None
    ) -> bool:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        _set_with_ttl(key, value, ex)
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in _store:
                count += 1
            _delete(k)
        return count

    async def rpush(self, key: str, *values: str) -> int:
        _rpush(key, *values)
        with _lock:
            return len(_store.get(key, []))

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        return _lrange(key, start, end)

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        _ltrim(key, start, end)
        return True

    async def expire(self, key: str, seconds: int) -> bool:
        _expire(key, seconds)
        return True

    async def getdel(self, key: str) -> str | None:
        return _getdel(key)

    async def aclose(self) -> None:
        pass
