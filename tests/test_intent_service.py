"""Tests for the customer intent extractor (structured outputs + fallback)."""

from __future__ import annotations

import pytest

from src.services.intent_service import (
    CustomerIntent,
    extract_customer_intent,
)


@pytest.mark.asyncio
async def test_fallback_extractor_returns_consultation_intent() -> None:
    result = await extract_customer_intent(
        sender_id="user_test",
        user_message="Can tu van automation tren Facebook va HubSpot",
        session_history=[],
        use_structured_output=False,
    )

    assert result.intent == CustomerIntent.CONSULTATION
    assert "facebook" in result.channels
    assert "hubspot" in result.channels
    assert 0.0 < result.confidence <= 1.0


@pytest.mark.asyncio
async def test_fallback_extractor_returns_pricing_intent() -> None:
    result = await extract_customer_intent(
        sender_id="user_test",
        user_message="Cho minh xin bang gia di",
        session_history=[],
        use_structured_output=False,
    )

    assert result.intent == CustomerIntent.PRICING
    assert result.budget == "Cho minh xin bang gia di"


@pytest.mark.asyncio
async def test_fallback_extractor_returns_handoff_intent() -> None:
    result = await extract_customer_intent(
        sender_id="user_test",
        user_message="Cho minh gap nhan vien tu van vien",
        session_history=[],
        use_structured_output=False,
    )

    assert result.intent == CustomerIntent.HANDOFF
    assert result.urgency == "medium"


@pytest.mark.asyncio
async def test_fallback_extractor_returns_fallback_intent() -> None:
    result = await extract_customer_intent(
        sender_id="user_test",
        user_message="abcdef",
        session_history=[],
        use_structured_output=False,
    )

    assert result.intent == CustomerIntent.FALLBACK
    assert result.confidence == 0.45
    assert result.product_interest is None
