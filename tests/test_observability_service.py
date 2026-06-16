"""Tests for the LangFuse observability wrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.services import observability_service


@pytest.fixture(autouse=True)
def reset_observability_state() -> None:
    observability_service._client = None
    observability_service._initialized = False
    yield  # type: ignore[misc]
    observability_service._client = None
    observability_service._initialized = False


def test_observability_disabled_returns_null_client() -> None:
    with patch.object(observability_service, "get_settings") as get_settings:
        get_settings.return_value.langfuse_enabled = False
        client = observability_service._get_client()

    assert isinstance(client, observability_service._NullLangFuseClient)


def test_trace_agent_run_no_op_when_disabled() -> None:
    with patch.object(observability_service, "is_observability_enabled", return_value=False):
        with observability_service.trace_agent_run(
            sender_id="user_1", user_message="hi"
        ) as trace:
            assert isinstance(trace, observability_service._NullTrace)


def test_record_intent_generation_swallows_errors() -> None:
    class _BrokenTrace:
        def generation(self, **_kwargs: Any) -> "_BrokenTrace":
            raise RuntimeError("kaboom")

    with patch.object(observability_service, "is_observability_enabled", return_value=True):
        # Should NOT raise even though the span is broken.
        observability_service.record_intent_generation(
            trace=_BrokenTrace(),
            model="gpt-4o-mini",
            user_message="hi",
            output={},
        )
