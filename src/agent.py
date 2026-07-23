"""ReAct AI Agent & Tool Calling Module.

Implements the ReAct (Reason + Act) pattern without heavy frameworks:
1. Thought: Agent reasons about what tool to call or whether it has enough info.
2. Action: Agent invokes tools:
   - ``search_knowledge_base``: Two-Stage RAG retrieval (Dense + BM25 + Reranker)
   - ``get_document_metadata``: Summary of loaded Knowledge Base
   - ``calculate``: Safe arithmetic evaluation
   - ``get_current_time``: Current system date & time
3. Observation: Tool output returned to Agent loop.
4. Final Answer: Grounded response provided to user.
"""

from __future__ import annotations

import datetime
import math
import re

from typing import Any, AsyncIterator
from loguru import logger

from src.config import get_settings
import src.rag as rag


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------


async def tool_search_knowledge_base(
    query: str,
    enable_hybrid: bool = True,
    enable_rerank: bool = True,
) -> str:
    """Tool: Retrieve relevant context chunks using Two-Stage Search."""
    try:
        hits = await rag.search(
            query,
            top_k=3,
            enable_hybrid=enable_hybrid,
            enable_rerank=enable_rerank,
        )
        if not hits:
            return "Không tìm thấy thông tin phù hợp trong Knowledge Base."
        return rag.format_context(hits)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent KB search failed")
        return f"Lỗi tra cứu KB: {exc}"


def tool_get_document_metadata(kb_state: dict[str, Any] | None = None) -> str:
    """Tool: Retrieve metadata and statistics about loaded documents."""
    kb_state = kb_state or {}
    domain = kb_state.get("kb_domain", "Chưa xác định")
    pages = kb_state.get("kb_pages", 0)
    chunks = kb_state.get("kb_chunks", 0)
    ready = kb_state.get("kb_ready", False)

    if not ready:
        return "⚠️ Knowledge Base hiện đang trống (chưa nạp tài liệu nào)."

    return (
        f"📊 **Knowledge Base Statistics**:\n"
        f"- Nguồn tài liệu: {domain}\n"
        f"- Tổng số trang/sheets: {pages}\n"
        f"- Tổng số chunks: {chunks}\n"
        f"- Trạng thái Vector DB: Đã sẵn sàng tra cứu"
    )


def tool_calculate(expression: str) -> str:
    """Tool: Safely evaluate mathematical expressions."""
    # Sanitize string: allow digits, operators, math functions, whitespace
    cleaned = re.sub(r"[^0-9\+\-\*\/\%\(\)\.\,\s]", "", expression)
    if not cleaned.strip():
        return "Lỗi: Biểu thức toán học không hợp lệ."

    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "pi": math.pi,
    }

    try:
        # Use eval with safe builtins scope
        result = eval(cleaned, {"__builtins__": None}, safe_dict)  # noqa: S307
        return f"{result}"
    except Exception as exc:  # noqa: BLE001
        return f"Lỗi tính toán ({expression}): {exc}"


def tool_get_current_time() -> str:
    """Tool: Get the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")


# ---------------------------------------------------------------------------
# ReAct Agent Engine
# ---------------------------------------------------------------------------


TOOL_DECLARATIONS = """
Các công cụ khả dụng (Tools):
1. search_knowledge_base(query="..."): Tra cứu thông tin trong tài liệu doanh nghiệp bằng Two-Stage RAG.
2. get_document_metadata(): Lấy thống kê về tài liệu đã nạp (số trang, số chunk, nguồn).
3. calculate(expression="..."): Thực hiện phép tính toán học (ví dụ: "50 * 1.10").
4. get_current_time(): Lấy ngày giờ hiện tại.

Cấu trúc phản hồi ReAct:
Nếu cần gọi tool, xuất ra đúng định dạng:
Thought: <suy luận của bạn>
Action: <tên_tool>(<các_tham_số>)

Nếu đã có đủ thông tin trả lời người dùng:
Thought: <suy luận cuối cùng>
Final Answer: <câu trả lời đầy đủ bằng tiếng Việt>
"""


async def run_agent_stream(
    user_message: str,
    history: list[dict[str, str]],
    kb_state: dict[str, Any] | None = None,
    enable_hybrid: bool = True,
    enable_rerank: bool = True,
    max_steps: int = 4,
) -> AsyncIterator[str]:
    """Execute the ReAct Agent loop and yield streaming progress updates.

    Yields intermediate Reasoning/Thought, Tool Executions, and Final Answer.
    """
    settings = get_settings()
    anthropic_key = settings.anthropic_api_key_value

    if not anthropic_key:
        yield "⚠️ ANTHROPIC_API_KEY chưa được cấu hình trong file .env."
        return

    # Build conversation system prompt & messages
    system_prompt = (
        "Bạn là AI Agent thông minh của hệ thống OmniAgent Flow.\n"
        "Bạn có khả năng suy luận (Thought) và sử dụng Công cụ (Action) để trả lời người dùng.\n"
        + TOOL_DECLARATIONS
    )

    agent_messages = []
    # Include history turns
    for turn in history[-4:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            agent_messages.append({"role": role, "content": content})

    # Append current turn
    agent_messages.append({"role": "user", "content": user_message})

    accumulated_output = f"🤖 **AI Agent đang suy luận...**\n\n"
    yield accumulated_output

    step = 0
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=anthropic_key)

    while step < max_steps:
        step += 1
        logger.info("Agent step starting", step=step)

        # Call Anthropic Claude API for next Thought/Action
        try:
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=600,
                system=system_prompt,
                messages=agent_messages,
                temperature=0.2,
            )
            model_text = response.content[0].text if response.content else ""
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent LLM call failed")
            yield accumulated_output + f"\n❌ Lỗi gọi LLM: {exc}"
            return

        # Append LLM turn to agent messages history
        agent_messages.append({"role": "assistant", "content": model_text})

        # Check if model arrived at Final Answer
        if "Final Answer:" in model_text:
            final_part = model_text.split("Final Answer:", 1)[1].strip()
            accumulated_output += f"💡 **Final Answer**:\n{final_part}"
            yield accumulated_output
            return

        # Parse Action from model_text
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", model_text, re.DOTALL)
        if not action_match:
            # If model didn't use strict Action syntax, yield response as final
            accumulated_output += f"\n{model_text}"
            yield accumulated_output
            return

        tool_name = action_match.group(1).strip()
        raw_args = action_match.group(2).strip()

        # Format Thought & Action display
        thought_part = model_text.split("Action:")[0].replace("Thought:", "").strip()
        accumulated_output += f"🧠 **Thought (Bước {step})**: {thought_part}\n"
        accumulated_output += f"🛠️ **Action**: `{tool_name}({raw_args})`\n"
        yield accumulated_output

        # Execute Tool
        obs_text = ""
        if tool_name == "search_knowledge_base":
            query_val = raw_args.strip("\"'")
            if "query=" in raw_args:
                query_val = re.sub(r'query\s*=\s*["\']?', '', raw_args).strip("\"'")
            obs_text = await tool_search_knowledge_base(
                query_val, enable_hybrid=enable_hybrid, enable_rerank=enable_rerank
            )
        elif tool_name == "get_document_metadata":
            obs_text = tool_get_document_metadata(kb_state)
        elif tool_name == "calculate":
            expr_val = raw_args.strip("\"'")
            if "expression=" in raw_args:
                expr_val = re.sub(r'expression\s*=\s*["\']?', '', raw_args).strip("\"'")
            obs_text = tool_calculate(expr_val)
        elif tool_name == "get_current_time":
            obs_text = tool_get_current_time()
        else:
            obs_text = f"Lỗi: Không tìm thấy công cụ '{tool_name}'"

        accumulated_output += f"👁️ **Observation**: {obs_text[:200]}...\n\n"
        yield accumulated_output

        # Append Observation back to Agent conversation
        agent_messages.append({
            "role": "user",
            "content": f"Observation from {tool_name}: {obs_text}",
        })

    # If max steps reached
    accumulated_output += "\n⚠️ Agent đạt giới hạn số bước suy luận tối đa."
    yield accumulated_output
