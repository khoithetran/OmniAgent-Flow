"""Telegram notification service for real-time event alerts.

This service sends formatted event notifications to a Telegram chat via
the official Bot HTTP API (sendMessage endpoint). It is intentionally
one-way: the system only pushes events, it does not consume incoming
updates from Telegram (no webhook for Telegram in this project).

Design notes
------------
- Pure async via httpx.AsyncClient so it fits inside Celery workers that
  already use asyncio.run for orchestration.
- The service is fail-soft: any error is logged via loguru and converted
  into a `TelegramNotificationResult` with status="failed". The webhook
  pipeline must never break because Telegram is down.
- HTML parse mode is used so we can bold intents, channels, etc. without
  escaping issues. We escape user-controlled fields before injection.
- A protocol is exposed for testing so unit tests can fake the HTTP
  transport without monkey-patching httpx.
"""

from html import escape
from typing import Any, Protocol

from httpx import AsyncClient, HTTPStatusError
from loguru import logger
from pydantic import BaseModel

from src.config import get_settings


class TelegramNotificationResult(BaseModel):
    """Outcome of a Telegram notification attempt.

    Attributes
    ----------
    status:
        One of "sent", "skipped", "failed". Mirrors the HubSpot service
        vocabulary so we can reuse the same audit event shape.
    chat_id:
        The destination chat id when the message was actually sent.
    message_id:
        The Telegram-assigned message id (useful for debugging).
    reason:
        Machine-friendly reason when status is "skipped" or "failed".
        Examples: "disabled", "missing_token", "telegram_http_error".
    """

    status: str
    chat_id: str | None = None
    message_id: int | None = None
    reason: str | None = None


class TelegramHTTPClient(Protocol):
    """Minimal HTTP contract used by this service.

    A real httpx.AsyncClient satisfies this protocol. Tests can pass a
    fake that records calls and returns canned responses.
    """

    async def post(self, url: str, json: dict[str, Any]) -> Any: ...


def _format_event_message(
    *,
    event_type: str,
    sender_id: str,
    intent: str,
    action: str,
    metadata: dict[str, Any],
    hubspot_status: str | None = None,
    hubspot_contact_id: str | None = None,
) -> str:
    """Build the human-readable HTML body for the Telegram alert.

    Parameters
    ----------
    event_type:
        Short tag describing the event, e.g. "new_message",
        "hubspot_sync_failed", "handoff_requested".
    sender_id:
        External user id from the originating channel.
    intent, action:
        Output of the LangGraph intent classifier.
    metadata:
        Full structured metadata. We only render a handful of fields
        to keep the alert scannable; the rest stays in PostgreSQL.
    hubspot_status, hubspot_contact_id:
        When the event is the result of a CRM sync, surface its outcome
        in the same alert to give the on-call team full context.
    """

    safe_sender = escape(sender_id)
    safe_intent = escape(intent)
    safe_action = escape(action)
    safe_event = escape(event_type)

    lines: list[str] = [
        f"<b>[{safe_event}]</b> OmniAgent Flow",
        f"Sender: <code>{safe_sender}</code>",
        f"Intent: <b>{safe_intent}</b> | Action: <code>{safe_action}</code>",
    ]

    company = metadata.get("company")
    customer_name = metadata.get("customer_name")
    if company or customer_name:
        bits: list[str] = []
        if customer_name:
            bits.append(escape(str(customer_name)))
        if company:
            bits.append(f"@ {escape(str(company))}")
        lines.append("Lead: " + " ".join(bits))

    channels = metadata.get("channels")
    if isinstance(channels, list) and channels:
        joined = ", ".join(escape(str(channel)) for channel in channels)
        lines.append(f"Channels: {joined}")

    budget = metadata.get("budget")
    if budget:
        lines.append(f"Budget: <i>{escape(str(budget))}</i>")

    urgency = metadata.get("urgency")
    if urgency:
        lines.append(f"Urgency: <b>{escape(str(urgency))}</b>")

    if hubspot_status:
        suffix = ""
        if hubspot_contact_id:
            suffix = f" (<code>{escape(hubspot_contact_id)}</code>)"
        lines.append(f"HubSpot: {escape(hubspot_status)}{suffix}")

    return "\n".join(lines)


async def send_telegram_notification(
    *,
    event_type: str,
    sender_id: str,
    intent: str,
    action: str,
    metadata: dict[str, Any],
    hubspot_status: str | None = None,
    hubspot_contact_id: str | None = None,
    client: TelegramHTTPClient | None = None,
) -> TelegramNotificationResult:
    """Send a single Telegram notification. Fail-soft by design.

    Returns
    -------
    TelegramNotificationResult
        Status string plus diagnostic context. Never raises to the
        caller so the Celery task flow stays unblocked.
    """

    settings = get_settings()
    bot_token = settings.telegram_bot_token_value
    chat_id = settings.telegram_chat_id_value

    if not settings.telegram_notifications_enabled:
        logger.info("Telegram notification skipped because it is disabled")
        return TelegramNotificationResult(status="skipped", reason="disabled")

    if not bot_token or not chat_id:
        logger.warning(
            "Telegram notification skipped because bot token or chat id is missing"
        )
        return TelegramNotificationResult(status="skipped", reason="missing_config")

    text = _format_event_message(
        event_type=event_type,
        sender_id=sender_id,
        intent=intent,
        action=action,
        metadata=metadata,
        hubspot_status=hubspot_status,
        hubspot_contact_id=hubspot_contact_id,
    )
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    owns_client = client is None
    resolved_client = client or AsyncClient(timeout=settings.telegram_timeout_seconds)

    try:
        response = await resolved_client.post(url, json=payload)
        response.raise_for_status()
        data = response.json() if hasattr(response, "json") else {}
        result_payload = data.get("result") if isinstance(data, dict) else None
        message_id: int | None = None
        if isinstance(result_payload, dict):
            raw_id = result_payload.get("message_id")
            if isinstance(raw_id, int):
                message_id = raw_id

        logger.info(
            "Sent Telegram notification",
            event_type=event_type,
            sender_id=sender_id,
            intent=intent,
            chat_id=chat_id,
        )
        return TelegramNotificationResult(
            status="sent",
            chat_id=chat_id,
            message_id=message_id,
        )
    except HTTPStatusError as exc:
        logger.exception(
            "Telegram API returned an error",
            event_type=event_type,
            status_code=exc.response.status_code,
        )
        return TelegramNotificationResult(
            status="failed", reason="telegram_http_error"
        )
    except Exception:
        logger.exception(
            "Failed to send Telegram notification", event_type=event_type
        )
        return TelegramNotificationResult(status="failed", reason="unexpected_error")
    finally:
        if owns_client and isinstance(resolved_client, AsyncClient):
            await resolved_client.aclose()
