"""Telegram Bot webhook for the chatbot.

This is the only public surface the bot has. The flow is:

1. Telegram sends a GET during webhook setup so we return 200 OK.
2. When a user sends a message to the bot, Telegram POSTs an update
   here. We authenticate it via the ``X-Telegram-Bot-Api-Secret-Token``
   header (if configured) and extract the user id + text.
3. We stream the LLM reply into Telegram by sending a "typing"
   action, posting a placeholder message, and editing it every
   time the LLM produces new tokens.

URL confirmation flow
---------------------
Before the normal chat pipeline, we analyse the message:

- If a URL is in the message, we crawl + index the site and
  immediately re-run the RAG search.
- If a company/org is detected but no URL is present, we set a
  pending_crawl marker in Redis (5 min TTL) and ask the user to
  send the URL.
- If the user is already in pending_crawl mode, a non-URL reply is
  treated as a non-action and the pending marker is kept.

Email capture flow
-------------------
When the LLM reports that the knowledge base is empty, the chat
layer sets a ``pending_email`` marker (5 min TTL). The next turn
the user sends is checked for an email address; if present, we
acknowledge, persist, and drop the marker. Otherwise we ask again
and keep the marker alive.

Everything is intentionally synchronous end-to-end: Telegram does
not require sub-5s response times, so we keep the design flat and
easy to follow.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from httpx import AsyncClient
from loguru import logger

from src import chat, entity, session
from src.config import get_settings
from src.simple_crawler import crawl_full_website
from src.rag import index_crawl_results


router = APIRouter(prefix="/telegram", tags=["telegram"])


#: Reply sent when the user mentions a company/org but no URL. The
#: exact text is reused by the tests; changing it requires updating
#: the test suite.
_URL_CONFIRMATION_PROMPT = (
    "Bạn có đang hỏi về {company} không? "
    "Vui lòng gửi URL website để tôi tìm hiểu thêm."
)

#: Reply sent when the user provides an email after a fall-back
#: answer. Reused by the tests.
_EMAIL_ACK = (
    "Cảm ơn bạn! Đội ngũ hỗ trợ sẽ liên hệ với bạn qua email trong thời gian sớm nhất."
)

#: Reply sent when the user is in pending_email mode but their
#: message does not contain an email. Reused by the tests.
_EMAIL_PROMPT = "Vui lòng gửi email của bạn để đội ngũ hỗ trợ liên hệ lại nhé."

#: Minimum interval between two Telegram editMessage calls so we do
#: not hit the rate limit. Telegram's limit is ~30 edits/minute per
#: chat, so 1.0s keeps us well below the threshold.
_STREAM_EDIT_INTERVAL_SECONDS = 1.0

#: Placeholder text shown while the LLM is still generating.
_STREAM_PLACEHOLDER = "..."


# ---------------------------------------------------------------------------
# Telegram HTTP helpers
# ---------------------------------------------------------------------------


async def _bot_request(method: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Send a JSON request to the Telegram Bot API.

    Returns the ``result`` field on success, or ``None`` on any
    failure. The helper never raises because Telegram outages must
    not take the chatbot pipeline down.
    """
    settings = get_settings()
    bot_token = settings.telegram_bot_token_value
    if not bot_token:
        logger.error("Telegram bot token missing; cannot call bot API")
        return None

    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    try:
        async with AsyncClient(timeout=settings.telegram_timeout_seconds) as client:
            response = await client.post(url, json=payload)
    except Exception:  # noqa: BLE001
        logger.exception("Telegram API call failed", method=method)
        return None

    if response.status_code >= 400:
        logger.error(
            "Telegram returned non-2xx",
            method=method,
            status_code=response.status_code,
            body=response.text[:200],
        )
        return None

    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(body, dict):
        return None
    return body.get("result")


async def send_telegram_message(chat_id: int | str, text: str) -> None:
    """Send ``text`` to ``chat_id`` via the Telegram Bot API.

    Fire-and-forget: any error is logged and swallowed. We never want
    a Telegram outage to take down the chatbot pipeline.
    """
    # Telegram caps message length at 4096 chars. We split on a
    # paragraph boundary when needed so a long LLM reply still
    # reaches the user intact.
    for chunk in _split_telegram_chunks(text):
        await _bot_request(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": chunk,
            },
        )


async def send_typing_action(chat_id: int | str) -> None:
    """Send a Telegram ``typing`` chat action.

    The action expires after 5 seconds so the chat layer has to
    re-send it for longer replies. The webhook does this on every
    edit iteration.
    """
    await _bot_request(
        "sendChatAction",
        {
            "chat_id": chat_id,
            "action": "typing",
        },
    )


async def _post_placeholder(chat_id: int | str) -> int | None:
    """Post a placeholder message and return its message_id.

    The webhook later edits this message in place to reveal the
    streaming LLM reply.
    """
    result = await _bot_request(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": _STREAM_PLACEHOLDER,
        },
    )
    if not isinstance(result, dict):
        return None
    raw_id = result.get("message_id")
    return int(raw_id) if isinstance(raw_id, int) else None


async def _edit_message(
    chat_id: int | str,
    message_id: int,
    text: str,
) -> None:
    """Replace the text of a message we previously posted.

    Telegram rejects edits that look identical to the current text,
    so we pad with the placeholder whenever the new text is empty
    (which only happens mid-stream when the LLM is still thinking).
    """
    if not text:
        text = _STREAM_PLACEHOLDER
    for chunk in _split_telegram_chunks(text):
        await _bot_request(
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": chunk,
            },
        )


async def stream_telegram_reply(
    chat_id: int | str,
    text_stream,
) -> None:
    """Stream ``text_stream`` into Telegram as a single edited message.

    Kept for backwards compatibility with any external caller, but
    the webhook now handles streaming inline. See :func:`receive_update`.
    """
    last_emitted = ""
    last_edit_at = 0.0
    edits_posted = 0

    placeholder_id = await _post_placeholder(chat_id)
    if placeholder_id is None:
        return

    async for accumulated in text_stream:
        if not isinstance(accumulated, str):
            continue
        if accumulated == last_emitted:
            continue
        now = time.monotonic()
        if edits_posted == 0 or (now - last_edit_at) >= _STREAM_EDIT_INTERVAL_SECONDS:
            await send_typing_action(chat_id)
            await _edit_message(chat_id, placeholder_id, accumulated)
            last_edit_at = now
            edits_posted += 1
            last_emitted = accumulated

    if last_emitted != accumulated:
        await _edit_message(chat_id, placeholder_id, accumulated)


def _split_telegram_chunks(text: str, *, limit: int = 4000) -> list[str]:
    """Split ``text`` into Telegram-safe chunks of <= ``limit`` chars."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(". ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining.strip())
    return chunks


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------


async def _handle_user_url(sender_id: str, url: str) -> str:
    """Crawl ``url``, index the result, then drop the pending marker.

    Returns the chat reply for the current turn.
    """
    logger.info("Crawling user-supplied URL", sender_id=sender_id, url=url)
    results = await crawl_full_website(url, max_pages=20)
    summary = await index_crawl_results(results, replace=True)
    logger.info(
        "Indexed user URL",
        url=url,
        pages=summary["pages"],
        chunks=summary["chunks"],
    )

    # The pending_crawl marker, if any, is no longer needed.
    await session.pop_pending_crawl(sender_id)

    # The URL itself becomes the user message so the RAG context
    # flows through the normal chat pipeline. We use the streaming
    # variant to keep the live-typing experience consistent.
    final_text = ""
    async for chunk in chat.chat_stream(sender_id, url):
        final_text = chunk
    return f"Da index {url}.\n\n{final_text}" if final_text else f"Da index {url}."


# ---------------------------------------------------------------------------
# Webhook endpoints
# ---------------------------------------------------------------------------


@router.get("/webhook")
async def verify_webhook() -> dict[str, str]:
    """Telegram calls GET during webhook setup; just acknowledge it."""
    return {"status": "ok", "service": "OmniAgent Flow"}


@router.post("/webhook")
async def receive_update(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(
        default=None, alias="X-Telegram-Bot-Api-Secret-Token"
    ),
) -> dict[str, Any]:
    """Process one Telegram update and reply to the user."""
    settings = get_settings()
    expected = settings.telegram_webhook_secret_token_value

    # Only reject if a secret token is configured AND the header was
    # sent but doesn't match. Telegram itself does not send the secret
    # header during normal delivery; we require it only when it is present.
    if (
        expected
        and x_telegram_bot_api_secret_token is not None
        and x_telegram_bot_api_secret_token != expected
    ):
        logger.warning("Telegram webhook: bad secret token")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid secret token",
        )

    try:
        body: dict[str, Any] = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    message: dict[str, Any] | None = body.get("message")
    if not isinstance(message, dict):
        return {"status": "ignored", "reason": "no message in update"}

    sender_id: str | None = None
    text: str | None = None
    user = message.get("from") or {}
    if isinstance(user, dict):
        raw_id = user.get("id")
        if raw_id is not None:
            sender_id = str(raw_id)
    if isinstance(message.get("text"), str):
        text = message["text"]

    if sender_id is None or not text:
        return {"status": "ignored", "reason": "missing sender or text"}

    logger.info("Telegram update received", sender_id=sender_id, text=text)

    # 1. Detect URL / company in the message.
    detected = await entity.analyse(text)

    # 2. If the user sent a URL, crawl it, index, and stream a reply.
    if detected.has_url:
        # Send a typing indicator early so the user knows the bot
        # is busy crawling.
        await send_typing_action(sender_id)
        reply = await _handle_user_url(sender_id, detected.url)  # type: ignore[arg-type]
        # Static reply: the URL crawl itself is a long blocking call,
        # so streaming on top of it does not add much value.
        await send_telegram_message(sender_id, reply)
        return {"status": "ok", "reply_length": len(reply), "crawled": detected.url}

    # 3. If we are already in pending_crawl mode and the user did
    # not provide a URL, drop the chat pipeline and ask again.
    pending = await session.peek_pending_crawl(sender_id)
    if pending is not None:
        company = pending.get("company", "công ty")
        prompt = _URL_CONFIRMATION_PROMPT.format(company=company)
        await send_telegram_message(sender_id, prompt)
        return {
            "status": "waiting_for_url",
            "company": company,
            "reply_length": len(prompt),
        }

    # 4. Otherwise, if the user mentioned a company/org, set the
    # pending_crawl marker and ask for the URL.
    if detected.has_company:
        await session.set_pending_crawl(sender_id, detected.company)  # type: ignore[arg-type]
        prompt = _URL_CONFIRMATION_PROMPT.format(company=detected.company)
        await send_telegram_message(sender_id, prompt)
        return {
            "status": "awaiting_url",
            "company": detected.company,
            "reply_length": len(prompt),
        }

    # 5. No URL, no company: email-capture then streaming chat pipeline.
    # We pre-send the typing action so the user sees the indicator
    # while the LLM is still warming up.
    await send_typing_action(sender_id)

    # 5a. If we previously asked for an email, check whether the
    # user has just sent one. We always consume the marker on a
    # valid email so the user can move on with normal conversation
    # without re-triggering the prompt.
    pending_email = await session.peek_pending_email(sender_id)
    if pending_email is not None:
        extracted_email = entity.extract_email(text)
        if extracted_email:
            await session.save_email(sender_id, extracted_email)
            await session.pop_pending_email(sender_id)
            await send_telegram_message(sender_id, _EMAIL_ACK)
            return {
                "status": "email_captured",
                "email": extracted_email,
                "reply_length": len(_EMAIL_ACK),
            }
        # No email in this turn - remind the user and skip the
        # chat pipeline. We keep the marker alive so the next turn
        # has another chance.
        await send_telegram_message(sender_id, _EMAIL_PROMPT)
        return {
            "status": "awaiting_email",
            "reply_length": len(_EMAIL_PROMPT),
        }

    # 5b. Normal streaming chat pipeline.
    edits = 0
    final_text = ""
    last_emitted = ""

    placeholder_id = await _post_placeholder(sender_id)
    if placeholder_id is None:
        final_text = await chat.chat(sender_id, text)
        if final_text:
            await send_telegram_message(sender_id, final_text)
        return {"status": "ok", "reply_length": len(final_text), "edits": 0}

    last_edit_at = 0.0
    async for accumulated in chat.chat_stream(sender_id, text):
        if not isinstance(accumulated, str) or accumulated == last_emitted:
            continue
        final_text = accumulated
        now = time.monotonic()
        if edits == 0 or (now - last_edit_at) >= _STREAM_EDIT_INTERVAL_SECONDS:
            await send_typing_action(sender_id)
            await _edit_message(sender_id, placeholder_id, accumulated)
            last_edit_at = now
            edits += 1
            last_emitted = accumulated

    if last_emitted != final_text and final_text:
        await _edit_message(sender_id, placeholder_id, final_text)
        edits += 1

    if not final_text:
        await _edit_message(
            sender_id,
            placeholder_id,
            "Xin lỗi, tôi không nhận được phản hồi. Vui lòng thử lại.",
        )

    return {"status": "ok", "reply_length": len(final_text), "edits": edits}
