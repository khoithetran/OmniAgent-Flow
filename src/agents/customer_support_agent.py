from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.services.intent_service import CustomerIntent, extract_customer_intent


AgentIntent = Literal["consultation", "pricing", "handoff", "fallback"]
AgentAction = Literal[
    "consultation_response",
    "pricing_response",
    "handoff_response",
    "fallback_response",
]


class AgentState(TypedDict, total=False):
    sender_id: str
    user_message: str
    session_history: list[dict[str, Any]]
    intent: AgentIntent
    action: AgentAction
    response: str
    metadata: dict[str, Any]
    use_structured_output: bool


class AgentResult(TypedDict):
    sender_id: str
    user_message: str
    session_history: list[dict[str, Any]]
    intent: AgentIntent
    action: AgentAction
    response: str
    metadata: dict[str, Any]


def _history_size(state: AgentState) -> int:
    return len(state.get("session_history", []))


def _action_for_intent(intent: CustomerIntent) -> AgentAction:
    action_map: dict[CustomerIntent, AgentAction] = {
        CustomerIntent.CONSULTATION: "consultation_response",
        CustomerIntent.PRICING: "pricing_response",
        CustomerIntent.HANDOFF: "handoff_response",
        CustomerIntent.FALLBACK: "fallback_response",
    }
    return action_map.get(intent, "fallback_response")


async def classify_intent(state: AgentState) -> AgentState:
    extraction = await extract_customer_intent(
        sender_id=state["sender_id"],
        user_message=state["user_message"],
        session_history=state.get("session_history", []),
        use_structured_output=state.get("use_structured_output", True),
    )
    intent: AgentIntent = extraction.intent.value
    action = _action_for_intent(extraction.intent)
    metadata = extraction.model_dump(mode="json")
    metadata["history_size"] = _history_size(state)

    logger.info(
        "Classified customer support intent",
        sender_id=state["sender_id"],
        intent=intent,
        action=action,
        confidence=extraction.confidence,
        history_size=_history_size(state),
    )
    return {
        "intent": intent,
        "action": action,
        "metadata": metadata,
    }


def route_agent_action(state: AgentState) -> AgentAction:
    return state.get("action", "fallback_response")


def consultation_response(state: AgentState) -> AgentState:
    return {
        "response": (
            "Cam on ban da chia se nhu cau. Toi co the tu van luong cham soc "
            "khach hang da kenh, gom webhook, hang doi xu ly nen, RAG va dong bo CRM "
            "de doi ngu van hanh nam ro boi canh truoc khi phan hoi."
        )
    }


def pricing_response(state: AgentState) -> AgentState:
    return {
        "response": (
            "Chi phi se phu thuoc vao so kenh tich hop, luong hoi thoai, muc do tu "
            "dong hoa va yeu cau RAG/CRM. Toi da ghi nhan day la nhu cau bao gia "
            "de doi ngu sales co the tiep tuc khai thac thong tin."
        )
    }


def handoff_response(state: AgentState) -> AgentState:
    return {
        "response": (
            "Toi da ghi nhan yeu cau can nguoi ho tro truc tiep. Doi ngu phu trach "
            "co the tiep nhan hoi thoai nay cung lich su gan nhat trong session."
        )
    }


def fallback_response(state: AgentState) -> AgentState:
    return {
        "response": (
            "Toi da nhan duoc thong tin cua ban. Doi ngu ho tro se tiep tuc trao doi "
            "de nam ro nhu cau va huong xu ly phu hop."
        )
    }


def build_customer_support_graph() -> Any:
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("classify_intent", classify_intent)
    graph_builder.add_node("consultation_response", consultation_response)
    graph_builder.add_node("pricing_response", pricing_response)
    graph_builder.add_node("handoff_response", handoff_response)
    graph_builder.add_node("fallback_response", fallback_response)

    graph_builder.add_edge(START, "classify_intent")
    graph_builder.add_conditional_edges(
        "classify_intent",
        route_agent_action,
        [
            "consultation_response",
            "pricing_response",
            "handoff_response",
            "fallback_response",
        ],
    )
    graph_builder.add_edge("consultation_response", END)
    graph_builder.add_edge("pricing_response", END)
    graph_builder.add_edge("handoff_response", END)
    graph_builder.add_edge("fallback_response", END)

    return graph_builder.compile()


customer_support_graph = build_customer_support_graph()


async def run_customer_support_agent(
    sender_id: str,
    user_message: str,
    session_history: list[dict[str, Any]],
    use_structured_output: bool = True,
) -> AgentResult:
    initial_state: AgentState = {
        "sender_id": sender_id,
        "user_message": user_message,
        "session_history": session_history,
        "use_structured_output": use_structured_output,
    }
    final_state = await customer_support_graph.ainvoke(initial_state)
    return {
        "sender_id": sender_id,
        "user_message": user_message,
        "session_history": session_history,
        "intent": final_state.get("intent", "fallback"),
        "action": final_state.get("action", "fallback_response"),
        "response": final_state.get("response", ""),
        "metadata": final_state.get("metadata", {}),
    }
