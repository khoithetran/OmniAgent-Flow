"""LangChain Expression Language (LCEL) chain orchestration for OmniAgent Flow.

This module builds clean, declarative LCEL chains using pipe ``|`` syntax:

    chain = (
        RunnableParallel({"context": ..., "question": ..., "chat_history": ...})
        | prompt_template
        | primary_model.with_fallbacks([fallback_model])
        | StrOutputParser()
    )

Features:
1. **Streaming & Async**: Automatic streaming token by token via ``chain.astream()``.
2. **Parallel Execution**: ``RunnableParallel`` executes context retrieval and prompt formatting in parallel.
3. **Model Fallbacks**: Automatically fall back from OpenAI to Anthropic (or vice versa) when one provider fails.
"""

from __future__ import annotations

from operator import itemgetter
from typing import Any, AsyncIterator

from loguru import logger
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.config import get_settings

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_GENERAL = (
    "Bạn là trợ lý ảo thân thiện, trả lời ngắn gọn bằng tiếng Việt có dấu. "
    "Nếu câu hỏi liên quan đến công ty/tổ chức cụ thể mà bạn chưa có thông tin, "
    "hãy gợi ý người dùng nhập URL website để được tra cứu chính xác hơn."
)

SYSTEM_PROMPT_RAG = (
    "Bạn là trợ lý ảo chỉ trả lời dựa trên 'Knowledge Base' được cung cấp bên dưới. "
    "MỖI phát biểu phải gắn citation theo số thứ tự trong ngoặc vuông, ví dụ: [1], [2]. "
    "Nếu thông tin không có trong Knowledge Base, hãy trả lời đúng: "
    "'Không tìm thấy thông tin này trong tài liệu được cung cấp.' "
    "TUYỆT ĐỐI KHÔNG bịa đặt, suy đoán, hay diễn giải thêm thông tin không có trong tài liệu.\n\n"
    "--- KNOWLEDGE BASE ---\n{context}"
)

# ---------------------------------------------------------------------------
# LLM Factory with LCEL Fallbacks
# ---------------------------------------------------------------------------

def get_lcel_model(selected_model_name: str | None = None) -> Any:
    """Build primary model and attach secondary model as fallback using LCEL ``with_fallbacks``.

    If OpenAI API key is missing or fails (e.g. out of credit), LCEL automatically
    switches execution to Anthropic Claude (and vice-versa).
    """
    settings = get_settings()
    openai_key = settings.openai_api_key_value
    anthropic_key = settings.anthropic_api_key_value

    models = []

    # Map Gradio selected model string or settings to LangChain model instance
    is_openai_requested = selected_model_name and "OpenAI" in selected_model_name

    # Primary OpenAI model
    openai_model_name = settings.openai_model or "gpt-4o-mini"
    if is_openai_requested and "GPT 5.4" in selected_model_name:
        openai_model_name = "gpt-4o-mini" # Map to actual OpenAI deployment

    primary_openai = (
        ChatOpenAI(
            model=openai_model_name,
            api_key=openai_key,
            temperature=0.2,
            streaming=True,
        )
        if openai_key
        else None
    )

    # Primary Anthropic model
    anthropic_model_name = "claude-3-5-sonnet-20241022"
    primary_anthropic = (
        ChatAnthropic(
            model=anthropic_model_name,
            api_key=anthropic_key,
            temperature=0.2,
            streaming=True,
        )
        if anthropic_key
        else None
    )

    if is_openai_requested:
        if primary_openai:
            models.append(primary_openai)
        if primary_anthropic:
            models.append(primary_anthropic)
    else:
        if primary_anthropic:
            models.append(primary_anthropic)
        if primary_openai:
            models.append(primary_openai)

    if not models:
        # Fallback to dummy / default OpenAI model if no key present
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)

    primary = models[0]
    fallbacks = models[1:]

    if fallbacks:
        logger.info(
            "Configured LCEL Model with Fallbacks",
            primary=primary.__class__.__name__,
            fallbacks=[f.__class__.__name__ for f in fallbacks],
        )
        return primary.with_fallbacks(fallbacks)

    return primary


# ---------------------------------------------------------------------------
# LCEL Chain Construction
# ---------------------------------------------------------------------------

def create_rag_lcel_chain(model: Any):
    """Build an LCEL pipeline for RAG mode using pipe ``|`` syntax.

    Chain architecture:
    Inputs -> PromptTemplate -> Model (with Fallbacks) -> StrOutputParser
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_RAG),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()
    return chain


def create_general_lcel_chain(model: Any):
    """Build an LCEL pipeline for General mode using pipe ``|`` syntax."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_GENERAL),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = prompt | model | StrOutputParser()
    return chain


# ---------------------------------------------------------------------------
# Streaming Execution Helper
# ---------------------------------------------------------------------------

async def stream_lcel_chain(
    chain: Any,
    inputs: dict[str, Any],
) -> AsyncIterator[str]:
    """Execute LCEL chain asynchronously using ``astream()`` and yield accumulated text."""
    accumulated = ""
    async for chunk in chain.astream(inputs):
        accumulated += chunk
        yield accumulated
