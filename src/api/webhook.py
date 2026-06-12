from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from loguru import logger

from src.config import get_settings
from src.workers.tasks import process_incoming_message

router = APIRouter(prefix="/webhook", tags=["webhook"])


def _extract_message_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("entry", [])
    events: list[dict[str, Any]] = []

    if not isinstance(entries, list):
        return events

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
            events.append(
                {
                    "sender_id": sender.get("id") if isinstance(sender, dict) else None,
                    "message": message if isinstance(message, dict) else {},
                }
            )

    return events


@router.get("", response_class=PlainTextResponse)
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge"),
) -> PlainTextResponse:
    settings = get_settings()

    if hub_mode == "subscribe" and hub_verify_token == settings.webhook_verify_token:
        logger.info("Facebook webhook verified successfully")
        return PlainTextResponse(content=hub_challenge, status_code=status.HTTP_200_OK)

    logger.warning(
        "Facebook webhook verification failed",
        hub_mode=hub_mode,
        hub_verify_token=hub_verify_token,
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid webhook verification token",
    )


@router.post("")
async def receive_webhook(payload: dict[str, Any] = Body(...)) -> dict[str, str]:
    message_events = _extract_message_events(payload)
    logger.info(
        "Received Facebook webhook payload",
        payload=payload,
        message_events=message_events,
    )
    task = process_incoming_message.delay(payload)
    logger.info("Queued Facebook webhook payload", task_id=task.id)
    return {"status": "success", "task_id": task.id}
