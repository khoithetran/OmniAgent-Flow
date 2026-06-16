"""Tests for the LangGraph customer support agent."""

from __future__ import annotations

import pytest

from src.agents.customer_support_agent import run_customer_support_agent


@pytest.mark.asyncio
async def test_agent_routes_pricing_intent() -> None:
    result = await run_customer_support_agent(
        sender_id="user_test",
        user_message="Cho toi xin bao gia giai phap",
        session_history=[{"role": "user", "content": "Xin chao"}],
        use_structured_output=False,
    )

    assert result["intent"] == "pricing"
    assert result["action"] == "pricing_response"
    assert "Chi phi" in result["response"]
    assert result["metadata"]["budget"] == "Cho toi xin bao gia giai phap"


@pytest.mark.asyncio
async def test_agent_routes_handoff_intent() -> None:
    result = await run_customer_support_agent(
        sender_id="user_test",
        user_message="Toi muon gap nhan vien sales",
        session_history=[],
        use_structured_output=False,
    )

    assert result["intent"] == "handoff"
    assert result["action"] == "handoff_response"


@pytest.mark.asyncio
async def test_agent_routes_consultation_intent() -> None:
    result = await run_customer_support_agent(
        sender_id="user_test",
        user_message="Toi can tu van automation cho fanpage",
        session_history=[],
        use_structured_output=False,
    )

    assert result["intent"] == "consultation"
    assert result["action"] == "consultation_response"


@pytest.mark.asyncio
async def test_agent_falls_back_when_no_keywords_match() -> None:
    result = await run_customer_support_agent(
        sender_id="user_test",
        user_message="Hom nay troi dep qua",
        session_history=[],
        use_structured_output=False,
    )

    assert result["intent"] == "fallback"
    assert result["action"] == "fallback_response"
    assert result["metadata"]["channels"] == []
