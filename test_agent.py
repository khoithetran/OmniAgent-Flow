import asyncio
from collections.abc import Awaitable, Callable

from src.agents.customer_support_agent import run_customer_support_agent
from src.services.intent_service import extract_customer_intent


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


async def test_agent_routes_handoff_intent() -> None:
    result = await run_customer_support_agent(
        sender_id="user_test",
        user_message="Toi muon gap nhan vien sales",
        session_history=[],
        use_structured_output=False,
    )

    assert result["intent"] == "handoff"
    assert result["action"] == "handoff_response"


async def test_fallback_extractor_returns_structured_metadata() -> None:
    result = await extract_customer_intent(
        sender_id="user_test",
        user_message="Can tu van automation tren Facebook va HubSpot",
        session_history=[],
        use_structured_output=False,
    )

    assert result.intent.value == "consultation"
    assert result.channels == ["facebook", "hubspot"]
    assert result.confidence > 0


async def main() -> None:
    tests: list[Callable[[], Awaitable[None]]] = [
        test_agent_routes_pricing_intent,
        test_agent_routes_handoff_intent,
        test_fallback_extractor_returns_structured_metadata,
    ]
    for test in tests:
        await test()


if __name__ == "__main__":
    asyncio.run(main())
