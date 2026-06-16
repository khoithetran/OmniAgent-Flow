"""Shared pytest fixtures.

The fixtures here centralize the env-var manipulation we used to do
inline in every test module. The pattern is:

1. Snapshot the current env var.
2. Override the values we want for the test.
3. Call ``get_settings.cache_clear()`` so the next call rebuilds the
   ``Settings`` instance with the new env.
4. Restore the snapshot and clear the cache again on teardown.

Doing this once in a fixture keeps every test deterministic and avoids
the leaking state we saw in the old ``test_hubspot.py``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

from src.config import get_settings


@pytest.fixture
def env_override(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Fixture yielding a small helper for ad-hoc env overrides.

    Usage::

        def test_something(env_override):
            env_override(OPENAI_API_KEY="sk-test", HUBSPOT_SYNC_ENABLED="true")
            ...
    """

    saved: dict[str, str | None] = {}

    def _set(**values: str) -> None:
        for key, value in values.items():
            saved.setdefault(key, os.environ.get(key))
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield _set

    get_settings.cache_clear()


@pytest.fixture
def reset_settings() -> Iterator[None]:
    """Force settings to be recomputed before and after a test."""

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
