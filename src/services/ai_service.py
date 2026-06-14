from loguru import logger

from src.agents.customer_support_agent import AgentResult, run_customer_support_agent
from src.services.session_service import get_session_history


async def generate_agent_result(sender_id: str, user_message: str) -> AgentResult:
    session_history = await get_session_history(sender_id)
    agent_result = await run_customer_support_agent(
        sender_id=sender_id,
        user_message=user_message,
        session_history=session_history,
    )
    logger.info(
        "Generated LangGraph assistant response",
        sender_id=sender_id,
        intent=agent_result["intent"],
        action=agent_result["action"],
        history_size=len(session_history),
    )
    return agent_result


async def generate_agent_response(sender_id: str, user_message: str) -> str:
    agent_result = await generate_agent_result(
        sender_id=sender_id,
        user_message=user_message,
    )
    return agent_result["response"]
