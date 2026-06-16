import asyncio
from typing import Any

from loguru import logger

from src.database import close_postgres, close_redis
from src.services.ai_service import generate_agent_result
from src.services.conversation_service import (
    init_conversation_schema,
    save_conversation_message,
    save_hubspot_sync_event,
)
from src.services.hubspot_service import sync_hubspot_lead
from src.services.session_service import save_session_message
from src.services.telegram_service import send_telegram_notification
from src.workers.celery_app import celery_app

app = celery_app
celery = celery_app


@celery_app.task(name="src.workers.tasks.health_check")
def health_check() -> dict[str, str]:
    logger.info("Celery health check task executed")
    return {"status": "ok", "component": "celery"}


@celery_app.task(name="src.workers.tasks.process_incoming_message")
def process_incoming_message(payload: dict[str, Any]) -> dict[str, Any]:
    message_events = _extract_facebook_text_messages(payload)

    if not message_events:
        logger.info("No valid Facebook text messages found in payload")
        return {"status": "skipped", "processed": 0}

    try:
        asyncio.run(_save_message_events(message_events))
    except Exception:
        logger.exception(
            "Failed to process incoming Facebook messages",
            message_count=len(message_events),
        )
        raise

    logger.info(
        "Processed incoming Facebook messages",
        processed=len(message_events),
    )
    return {"status": "processed", "processed": len(message_events)}


def _extract_facebook_text_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    entries = payload.get("entry", [])
    message_events: list[dict[str, str]] = []

    if not isinstance(entries, list):
        logger.warning("Facebook payload entry field is not a list")
        return message_events

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        messaging_items = entry.get("messaging", [])
        if not isinstance(messaging_items, list):
            continue

        for item in messaging_items:
            if not isinstance(item, dict):
                continue

            sender = item.get("sender", {})
            message = item.get("message", {})
            sender_id = sender.get("id") if isinstance(sender, dict) else None
            text = message.get("text") if isinstance(message, dict) else None

            if isinstance(sender_id, str) and isinstance(text, str) and text.strip():
                message_events.append({"sender_id": sender_id, "text": text})

    return message_events


async def _save_message_events(message_events: list[dict[str, str]]) -> None:
    try:
        await init_conversation_schema()

        for event in message_events:
            sender_id = event["sender_id"]
            user_message = event["text"]

            await save_session_message(
                sender_id,
                role="user",
                content=user_message,
            )
            await save_conversation_message(
                sender_id=sender_id,
                role="user",
                content=user_message,
            )

            agent_result = await generate_agent_result(
                sender_id=sender_id,
                user_message=user_message,
            )
            assistant_response = agent_result["response"]
            await save_session_message(
                sender_id,
                role="assistant",
                content=assistant_response,
            )
            await save_conversation_message(
                sender_id=sender_id,
                role="assistant",
                content=assistant_response,
                intent=agent_result["intent"],
                action=agent_result["action"],
                metadata=agent_result["metadata"],
            )
            sync_result = await sync_hubspot_lead(
                sender_id=sender_id,
                intent=agent_result["intent"],
                action=agent_result["action"],
                metadata=agent_result["metadata"],
            )
            await save_hubspot_sync_event(
                sender_id=sender_id,
                status=sync_result.status,
                hubspot_contact_id=sync_result.contact_id,
                action=sync_result.action,
                reason=sync_result.reason,
                intent=agent_result["intent"],
                payload=agent_result["metadata"],
            )

            # Realtime push to Telegram so on-call humans see hot leads,
            # CRM failures, and handoff requests the moment they happen.
            await _maybe_notify_telegram(
                sender_id=sender_id,
                agent_result=agent_result,
                hubspot_status=sync_result.status,
                hubspot_contact_id=sync_result.contact_id,
            )
    finally:
        await close_postgres()
        await close_redis()


async def _maybe_notify_telegram(
    *,
    sender_id: str,
    agent_result: dict[str, Any],
    hubspot_status: str,
    hubspot_contact_id: str | None,
) -> None:
    """Decide which event label to send and dispatch the Telegram alert.

    We classify the event so the on-call channel can filter easily:

    - "hubspot_sync_failed"  -> the CRM write failed; needs investigation.
    - "handoff_requested"    -> a customer asked for a human; sales should jump in.
    - "hot_lead_captured"    -> a high-urgency pricing lead was captured cleanly.
    - "new_message"          -> default catch-all for everything else.
    """

    intent = agent_result["intent"]
    action = agent_result["action"]
    metadata = agent_result.get("metadata", {}) or {}

    if hubspot_status == "failed":
        event_type = "hubspot_sync_failed"
    elif intent == "handoff":
        event_type = "handoff_requested"
    elif intent == "pricing" and hubspot_status == "synced":
        event_type = "hot_lead_captured"
    else:
        event_type = "new_message"

    await send_telegram_notification(
        event_type=event_type,
        sender_id=sender_id,
        intent=intent,
        action=action,
        metadata=metadata,
        hubspot_status=hubspot_status,
        hubspot_contact_id=hubspot_contact_id,
    )
