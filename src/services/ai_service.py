from typing import Any

from loguru import logger

from src.agents.customer_support_agent import AgentResult, run_customer_support_agent
from src.services.evaluation_service import build_evaluation_metadata, evaluate_response
from src.services.observability_service import (
    atrace_agent_run,
    is_observability_enabled,
    record_evaluation_score,
)
from src.services.rag_service import hybrid_search_knowledge
from src.services.session_service import get_session_history


async def generate_agent_result(sender_id: str, user_message: str) -> AgentResult:
    session_history = await get_session_history(sender_id)

    # RAG retrieval happens outside LangGraph so the graph stays a pure
    # orchestrator. We pull the context first, then let the agent pick
    # the response branch.
    rag_results = hybrid_search_knowledge(user_message, limit=3)
    rag_context = "\n\n".join(result.content for result in rag_results)
    if rag_context:
        session_history = session_history + [
            {
                "role": "system",
                "content": f"Retrieved knowledge context:\n{rag_context}",
            }
        ]

    observability_on = is_observability_enabled()

    if observability_on:
        async with atrace_agent_run(
            sender_id=sender_id, user_message=user_message
        ) as trace:
            agent_result = await _run_agent_with_eval(
                trace=trace,
                sender_id=sender_id,
                user_message=user_message,
                session_history=session_history,
                rag_context=rag_context,
            )
    else:
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


async def _run_agent_with_eval(
    *,
    trace: Any,
    sender_id: str,
    user_message: str,
    session_history: list[dict[str, Any]],
    rag_context: str,
) -> AgentResult:
    agent_result = await run_customer_support_agent(
        sender_id=sender_id,
        user_message=user_message,
        session_history=session_history,
    )

    scores = await evaluate_response(
        question=user_message,
        answer=agent_result["response"],
        context=rag_context,
    )
    evaluation_metadata = build_evaluation_metadata(scores)
    agent_result["metadata"] = {**(agent_result.get("metadata") or {}), **evaluation_metadata}

    record_evaluation_score(
        trace=trace, name="faithfulness", value=scores.faithfulness
    )
    record_evaluation_score(
        trace=trace, name="answer_relevance", value=scores.answer_relevance
    )

    logger.info(
        "Generated LangGraph assistant response with evaluation",
        sender_id=sender_id,
        intent=agent_result["intent"],
        action=agent_result["action"],
        history_size=len(session_history),
        faithfulness=scores.faithfulness,
        answer_relevance=scores.answer_relevance,
        eval_method=scores.method,
    )
    return agent_result


async def generate_agent_response(sender_id: str, user_message: str) -> str:
    agent_result = await generate_agent_result(
        sender_id=sender_id,
        user_message=user_message,
    )
    return agent_result["response"]
