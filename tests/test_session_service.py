"""Tests for the session service (Redis-backed chat memory)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services import session_service


@pytest.fixture
def fake_redis() -> MagicMock:
    redis_mock = MagicMock()
    redis_mock.rpush = AsyncMock(return_value=1)
    redis_mock.ltrim = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock(return_value=1)
    redis_mock.lrange = AsyncMock(return_value=[])
    redis_mock.ttl = AsyncMock(return_value=1800)
    return redis_mock


@pytest.mark.asyncio
async def test_save_session_message_appends_and_trims(
    fake_redis: MagicMock, env_override: Any
) -> None:
    env_override(SESSION_TTL_SECONDS="900")

    with patch_redis(fake_redis):
        await session_service.save_session_message(
            "user_1", role="user", content="Xin chao"
        )

    fake_redis.rpush.assert_awaited_once()
    fake_redis.ltrim.assert_awaited_once_with("session:user_1", -10, -1)
    fake_redis.expire.assert_awaited_once_with("session:user_1", 900)


@pytest.mark.asyncio
async def test_get_session_history_decodes_payloads(
    fake_redis: MagicMock,
) -> None:
    fake_redis.lrange = AsyncMock(
        return_value=['{"role":"user","content":"Hi"}', "not-json", "garbage"]
    )

    with patch_redis(fake_redis):
        history = await session_service.get_session_history("user_1")

    # JSONDecodeError logs and skips; we keep the dict entry only.
    assert history == [{"role": "user", "content": "Hi"}]


@pytest.mark.asyncio
async def test_save_session_message_rejects_invalid_max_messages(
    fake_redis: MagicMock,
) -> None:
    with patch_redis(fake_redis):
        with pytest.raises(ValueError):
            await session_service.save_session_message(
                "user_1", role="user", content="Hi", max_messages=0
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def patch_redis(fake_redis: MagicMock) -> Any:
    """Return a context manager that swaps the redis client getter."""

    return _PatchRedis(fake_redis)


class _PatchRedis:
    def __init__(self, fake_redis: MagicMock) -> None:
        self._fake_redis = fake_redis

    def __enter__(self) -> _PatchRedis:
        self._original = session_service.get_redis_client
        session_service.get_redis_client = AsyncMock(return_value=self._fake_redis)
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        session_service.get_redis_client = self._original
