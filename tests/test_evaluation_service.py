"""Tests for the AI evaluation service."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.evaluation_service import evaluate_response


@pytest.mark.asyncio
async def test_evaluate_response_uses_fallback_when_no_openai_key(
    env_override: Any,
) -> None:
    env_override(OPENAI_API_KEY="")
    scores = await evaluate_response(
        question="Toi muon xin bao gia",
        answer="Chi phi se phu thuoc vao so kenh tich hop.",
        context="Chi phi phu thuoc vao so kenh tich hop va luong hoi thoai.",
    )

    assert scores.method == "fallback"
    # The answer shares the word "phi" and "kenh" with the context.
    assert 0.0 < scores.faithfulness <= 1.0
    # The answer contains the keyword "chi phi" -> relevance boost.
    assert scores.answer_relevance > 0.0


@pytest.mark.asyncio
async def test_evaluate_response_zero_relevance_when_no_overlap(
    env_override: Any,
) -> None:
    env_override(OPENAI_API_KEY="")
    scores = await evaluate_response(
        question="abc xyz",
        answer="lorem ipsum dolor",
        context="",
    )
    # No question overlap and no intent keyword -> 0.
    assert scores.answer_relevance == 0.0
    # Empty context -> neutral 0.5 fallback.
    assert scores.faithfulness == 0.5
