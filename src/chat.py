"""Single-function chat layer for the chatbot (Telegram + Gradio).

This module is the only place that orchestrates a reply. It supports two
modes, both with and without RAG context:

**General mode (no KB)**
    1. Read sliding-window chat history from Redis.
    2. Check LLM response cache.
    3. Build messages with ``SYSTEM_PROMPT_GENERAL``.
    4. Call OpenAI chat completions (stream or non-stream).

**RAG mode (KB ready)**
    1. Read sliding-window chat history from Redis.
    2. Check LLM response cache.
    3. Search Qdrant for relevant chunks.
    4. Build messages with ``SYSTEM_PROMPT_RAG`` + context block.
    5. Call OpenAI chat completions (stream or non-stream).

The system prompts are the only place where the bot's persona lives.
``SYSTEM_PROMPT_RAG`` is strict: if the answer is not in the KB, the LLM
must say so and never make up information.

Public entry points:

- ``chat`` - non-streaming, returns string. (backward compatible)
- ``chat_stream`` - streaming with ``kb_ready`` flag (Telegram legacy).
- ``chat_general_stream`` - streaming general LLM (Gradio).
- ``chat_rag_stream`` - streaming RAG (Gradio).
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from loguru import logger
from openai import AsyncOpenAI

from src import entity, rag
from src import session
from src.config import get_settings


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

#: Prompt for general mode — used when no knowledge base is loaded.
#: Bot can answer general questions and should recommend the user
#: provide a URL for company-specific queries.
SYSTEM_PROMPT_GENERAL = (
    "Bạn là trợ lý ảo thân thiện, trả lời ngắn gọn bằng tiếng Việt có dấu. "
    "Nếu câu hỏi liên quan đến công ty/tổ chức cụ thể mà bạn chưa có thông tin, "
    "hãy gợi ý người dùng nhập URL website để được tra cứu chính xác hơn."
)

#: Prompt for RAG mode — used when a knowledge base is loaded.
#: Bot MUST answer only from the provided Knowledge Base.
SYSTEM_PROMPT_RAG = (
    "Bạn là trợ lý ảo chỉ trả lời dựa trên 'Knowledge Base' được cung cấp bên dưới. "
    "MỖI phát biểu phải gắn citation theo số thứ tự trong ngoặc vuông, ví dụ: [1], [2]. "
    "Nếu thông tin không có trong Knowledge Base, hãy trả lời đúng: "
    "'Không tìm thấy thông tin này trong tài liệu được cung cấp.' "
    "TUYỆT ĐỐI KHÔNG bịa đặt, suy đoán, hay diễn giải thêm thông tin không có trong tài liệu."
)

#: Backward-compatible alias for tests and old call sites.
SYSTEM_PROMPT = SYSTEM_PROMPT_RAG


# ---------------------------------------------------------------------------
# Chat entry point (non-streaming, backward compatible)
# ---------------------------------------------------------------------------


async def chat(sender_id: str, user_message: str) -> str:
    """Generate a reply for ``sender_id`` given the new ``user_message``.

    Returns the assistant text. Errors are caught and converted into a
    friendly fallback so the webhook can always return *something* to
    the user. This is the legacy non-streaming entry point kept for
    backward compatibility; new code should use ``chat_rag_stream`` or
    ``chat_general_stream``.
    """
    if not get_settings().openai_api_key_value:
        return _missing_key_reply()

    # 1. Load chat history (oldest first, capped at 10 messages).
    try:
        history = await session.get_history(sender_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load chat history", sender_id=sender_id)
        history = []

    # 1b. Check LLM response cache — return immediately on hit.
    cached = await session.cache_get(user_message)
    if cached is not None:
        await _persist_turn(sender_id, user_message, cached)
        if _should_capture_email("", cached):
            try:
                await session.set_pending_email(sender_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to set pending_email marker", sender_id=sender_id
                )
        return cached

    # 2. RAG retrieval.
    context_block = await _retrieve_context(user_message)

    # 3. Build the OpenAI messages array.
    messages = _build_messages(history, context_block, user_message)

    # 4. Call OpenAI.
    reply_text = await _call_openai(messages)

    # 4b. Cache the reply so identical questions are served instantly later.
    await session.cache_set(user_message, reply_text)

    # 5. Save to session for the next turn.
    await _persist_turn(sender_id, user_message, reply_text)

    # 6. If the LLM could not find anything in the knowledge base,
    # set a pending_email marker so the webhook knows to ask for
    # an email on the next turn. We rely on the LLM's reply text
    # containing the canonical fallback phrase, which the system
    # prompt asks it to use.
    if _should_capture_email(context_block, reply_text):
        try:
            await session.set_pending_email(sender_id)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to set pending_email marker", sender_id=sender_id)

    return reply_text


async def chat_stream(
    sender_id: str,
    user_message: str,
    *,
    kb_ready: bool = True,
    model: str | None = None,
) -> AsyncIterator[str]:
    """Stream the LLM reply as it is produced.

    Yields the full accumulated text every time a new token arrives.
    The caller can decide when to ship an update - for Telegram we
    wait until at least one new character has arrived and at least
    ``min_interval`` seconds have elapsed since the previous yield.

    Parameters
    ----------
    sender_id:
        Unique identifier for the conversation (Telegram user id, Gradio
        session hash, etc).
    user_message:
        The new user turn to reply to.
    kb_ready:
        When True, use the RAG prompt and search Qdrant. When False, use
        the general prompt and skip RAG (general LLM mode).
    model:
        OpenAI model name. Defaults to ``settings.openai_model``.

    Falls back to the non-streaming :func:`chat` if streaming is
    unavailable (no API key, no streaming client, OpenAI error).
    """
    if not get_settings().openai_api_key_value:
        yield _missing_key_reply()
        return

    try:
        history = await session.get_history(sender_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load chat history", sender_id=sender_id)
        history = []

    # Cache check before doing any RAG or LLM work.
    cached = await session.cache_get(user_message)
    if cached is not None:
        await _persist_turn(sender_id, user_message, cached)
        if _should_capture_email("", cached):
            try:
                await session.set_pending_email(sender_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to set pending_email marker", sender_id=sender_id
                )
        yield cached
        return

    # RAG context — only when KB is ready.
    if kb_ready:
        context_block = await _retrieve_context(user_message)
        messages = _build_messages(history, context_block, user_message)
    else:
        context_block = ""
        messages = _build_messages_general(history, user_message)

    settings = get_settings()
    client: AsyncOpenAI = rag._get_openai()  # type: ignore[assignment]
    chosen_model = model or settings.openai_model

    accumulated = ""
    try:
        response = await client.chat.completions.create(
            **_build_completion_kwargs(
                model=chosen_model,
                messages=messages,
                temperature=0.4,
                max_tokens=600,
                stream=True,
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("OpenAI streaming failed; falling back to non-streaming")
        full = await _call_openai(messages, model=chosen_model)
        yield full
        await session.cache_set(user_message, full)
        await _persist_turn(sender_id, user_message, full)
        if _should_capture_email(context_block, full):
            try:
                await session.set_pending_email(sender_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to set pending_email marker", sender_id=sender_id
                )
        return

    async for chunk in response:
        delta = _extract_delta(chunk)
        if not delta:
            continue
        accumulated += delta
        yield accumulated

    if not accumulated:
        # Stream completed but produced no text - rare but possible
        # when the model is filtered. Yield the static fallback so
        # the caller can still ship *something*.
        accumulated = (
            "Xin lỗi, tôi không nhận được phản hồi từ hệ thống. "
            "Vui lòng thử lại."
        )
        yield accumulated

    await _persist_turn(sender_id, user_message, accumulated)

    # Cache the final accumulated reply for identical future questions.
    await session.cache_set(user_message, accumulated)

    if _should_capture_email(context_block, accumulated):
        try:
            await session.set_pending_email(sender_id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to set pending_email marker", sender_id=sender_id
            )


# ---------------------------------------------------------------------------
# Gradio-specific entry points
# ---------------------------------------------------------------------------


async def chat_general_stream(
    sender_id: str,
    user_message: str,
    model: str | None = None,
) -> AsyncIterator[str]:
    """Stream a general (non-RAG) LLM reply.

    Used by the Gradio interface when no knowledge base is loaded.
    Skips Qdrant entirely and uses ``SYSTEM_PROMPT_GENERAL``.
    """
    async for accumulated in chat_stream(
        sender_id,
        user_message,
        kb_ready=False,
        model=model,
    ):
        yield accumulated


async def chat_rag_stream(
    sender_id: str,
    user_message: str,
    model: str | None = None,
) -> AsyncIterator[str]:
    """Stream a RAG-grounded LLM reply.

    Used by the Gradio interface when a knowledge base is loaded.
    Uses ``SYSTEM_PROMPT_RAG`` and Qdrant retrieval.
    """
    async for accumulated in chat_stream(
        sender_id,
        user_message,
        kb_ready=True,
        model=model,
    ):
        yield accumulated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _missing_key_reply() -> str:
    return (
        "Xin lỗi, hệ thống hiện chưa được cấu hình OpenAI key. "
        "Vui lòng liên hệ quản trị viên."
    )


async def _retrieve_context(user_message: str) -> str:
    """Run the RAG search and return a formatted context block.

    Returns an empty string when the OpenAI client is not ready or
    the search fails. Failures are logged but never bubble up.
    """
    if rag.openai_client is None:
        return ""
    try:
        hits = await rag.search(user_message)
        return rag.format_context(hits)
    except Exception:  # noqa: BLE001
        logger.exception("RAG search failed")
        return ""


def _should_capture_email(context_block: str, reply_text: str) -> bool:
    """Decide whether the LLM reply signals an out-of-scope answer.

    The system prompt instructs the LLM to use a canonical phrase
    when the knowledge base is empty. We rely on that phrase as a
    signal rather than parsing the LLM output for intent.
    """
    if context_block.strip():
        # The RAG KB had at least one chunk. The LLM was able to
        # ground its answer, so we do not ask for an email.
        return False
    if "email" in reply_text.lower() and "để lại" in reply_text.lower():
        return True
    return False


def _build_messages(
    history: list[dict[str, Any]],
    context_block: str,
    user_message: str,
) -> list[dict[str, Any]]:
    """Assemble the OpenAI messages array for one chat turn."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(context_block)},
    ]
    messages.extend(_sanitise_history(history))
    messages.append({"role": "user", "content": user_message})
    return messages


def _build_system_prompt(context_block: str) -> str:
    """Combine the static system prompt with the optional RAG context.

    When ``context_block`` is empty, returns the RAG prompt (since
    ``_build_messages`` is only called from the RAG path, this branch
    is rare in practice; for general mode we bypass the helper).
    When context is present, appends the Knowledge Base section.
    """
    if not context_block:
        return SYSTEM_PROMPT_RAG

    return (
        f"{SYSTEM_PROMPT_RAG}\n\n"
        "Knowledge Base (cited as [n]):\n"
        f"{context_block}"
    )


def _build_messages_general(
    history: list[dict[str, Any]],
    user_message: str,
) -> list[dict[str, Any]]:
    """Build the OpenAI messages array for general (non-RAG) mode."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT_GENERAL},
    ]
    messages.extend(_sanitise_history(history))
    messages.append({"role": "user", "content": user_message})
    return messages


def _sanitise_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop any history entries that are not valid OpenAI messages.

    Defensive code: protects the API from a malformed Redis payload.
    """
    out: list[dict[str, Any]] = []
    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant", "system"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        out.append({"role": role, "content": content})
    return out


async def _call_openai(messages: list[dict[str, Any]], model: str | None = None) -> str:
    """Send ``messages`` to OpenAI and return the assistant text.

    Any exception is converted into a friendly fallback string so the
    chat layer never bubbles errors up to the webhook.
    """
    settings = get_settings()
    client: AsyncOpenAI = rag._get_openai()  # type: ignore[assignment]
    chosen_model = model or settings.openai_model

    try:
        response = await client.chat.completions.create(
            **_build_completion_kwargs(
                model=chosen_model,
                messages=messages,
                temperature=0.4,
                max_tokens=600,
                stream=False,
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("OpenAI chat completion failed")
        return (
            "Xin lỗi, hệ thống đang gặp sự cố khi xử lý câu hỏi của bạn. "
            "Vui lòng thử lại sau ít phút."
        )

    try:
        choice = response.choices[0]
        return (choice.message.content or "").strip()
    except (AttributeError, IndexError, TypeError):
        logger.exception("Unexpected OpenAI response shape")
        return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."


def _extract_delta(chunk: Any) -> str:
    """Pull the new text out of one streamed chunk from OpenAI.

    The chunk structure is a Pydantic model; we guard every access
    so a malformed chunk never crashes the stream loop.
    """
    try:
        choices = chunk.choices
    except AttributeError:
        return ""
    if not choices:
        return ""
    delta = choices[0].delta
    content = getattr(delta, "content", None)
    return content or ""


def _is_reasoning_model(model: str) -> bool:
    """Return True when ``model`` is an OpenAI o-series reasoning model.

    Reasoning models (o1, o3, o4, …) reject ``max_tokens`` and require
    ``max_completion_tokens``. We detect by name prefix so any future
    o-series variant is handled automatically.
    """
    if not model:
        return False
    name = model.lower().strip()
    return name.startswith(("o1", "o3", "o4", "o5"))


def _build_completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    """Build kwargs for ``chat.completions.create``.

    For o-series reasoning models, swap ``max_tokens`` for
    ``max_completion_tokens`` and drop ``temperature`` (reasoning
    models only support the default temperature=1).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if _is_reasoning_model(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
    return kwargs


async def _persist_turn(sender_id: str, user_message: str, reply_text: str) -> None:
    """Save one turn to the Redis session, ignoring storage failures."""
    try:
        await session.save_message(sender_id, "user", user_message)
        await session.save_message(sender_id, "assistant", reply_text)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist session", sender_id=sender_id)
