"""Single-function chat layer for the Telegram chatbot.

This module is the only place that orchestrates a reply. The flow is:

1. Read the sliding-window chat history from Redis so the LLM has
   conversational context.
2. Search the Qdrant knowledge base with the user's message. If hits
   arrive, format them as a numbered context block.
3. Call OpenAI chat completions with the system prompt, history,
   context (if any), and the new user message.
4. Save both the user message and the assistant reply to Redis.
5. Return the reply text so the webhook can ship it to Telegram.

The system prompt is the only place where the bot's persona lives.
It instructs the LLM to answer in Vietnamese, fall back gracefully
when the knowledge base does not contain the answer, and ask the
user to leave an email when the question is outside scope so a human
can follow up.

Two public entry points:

- ``chat`` - returns the full reply as a string. Use this when
  simple sync-style replies are enough.
- ``chat_stream`` - yields partial text as the LLM produces it. The
  Telegram webhook uses this to show a live "typing" experience by
  editing the same message every few hundred milliseconds.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from loguru import logger
from openai import AsyncOpenAI

from src import entity, rag
from src import session
from src.config import get_settings


SYSTEM_PROMPT = (
    "Bạn là trợ lý ảo của công ty, trả lời các câu hỏi của khách hàng một cách "
    "thân thiện, ngắn gọn và chính xác bằng tiếng Việt có dấu. "
    "Mỗi câu trả lời phải dựa trên thông tin được cung cấp trong phần "
    "'Knowledge Base' bên dưới. "
    "Nếu thông tin không có trong Knowledge Base, hãy trả lời: "
    "'Xin lỗi, tôi không tìm thấy thông tin này trong cơ sở dữ liệu. "
    "Bạn có muốn để lại email để đội ngũ hỗ trợ liên hệ lại không?' "
    "và KHÔNG bịa đặt thông tin. "
    "Mỗi phát biểu nên được gắn citation theo số thứ tự trong ngoặc vuông, "
    "ví dụ: [1], [2], để người dùng có thể truy ngược nguồn."
)


# ---------------------------------------------------------------------------
# Chat entry point
# ---------------------------------------------------------------------------


async def chat(sender_id: str, user_message: str) -> str:
    """Generate a reply for ``sender_id`` given the new ``user_message``.

    Returns the assistant text. Errors are caught and converted into a
    friendly fallback so the webhook can always return *something* to
    the user.
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
) -> AsyncIterator[str]:
    """Stream the LLM reply as it is produced.

    Yields the full accumulated text every time a new token arrives.
    The caller can decide when to ship an update - for Telegram we
    wait until at least one new character has arrived and at least
    ``min_interval`` seconds have elapsed since the previous yield.

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

    context_block = await _retrieve_context(user_message)
    messages = _build_messages(history, context_block, user_message)

    settings = get_settings()
    client: AsyncOpenAI = rag._get_openai()  # type: ignore[assignment]

    accumulated = ""
    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.4,
            max_tokens=600,
            stream=True,
        )
    except Exception:  # noqa: BLE001
        logger.exception("OpenAI streaming failed; falling back to non-streaming")
        full = await _call_openai(messages)
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
    """Combine the static system prompt with the optional RAG context."""
    if not context_block:
        return SYSTEM_PROMPT

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Knowledge Base (cited as [n]):\n"
        f"{context_block}"
    )


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


async def _call_openai(messages: list[dict[str, Any]]) -> str:
    """Send ``messages`` to OpenAI and return the assistant text.

    Any exception is converted into a friendly fallback string so the
    chat layer never bubbles errors up to the webhook.
    """
    settings = get_settings()
    client: AsyncOpenAI = rag._get_openai()  # type: ignore[assignment]

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.4,
            max_tokens=600,
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


async def _persist_turn(sender_id: str, user_message: str, reply_text: str) -> None:
    """Save one turn to the Redis session, ignoring storage failures."""
    try:
        await session.save_message(sender_id, "user", user_message)
        await session.save_message(sender_id, "assistant", reply_text)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist session", sender_id=sender_id)
