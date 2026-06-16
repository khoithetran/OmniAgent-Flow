"""Tests for Celery task orchestration.

The worker runs ``process_incoming_message`` in response to a webhook
POST. We mock the heavy services (DB, agent, HubSpot, Telegram) and
verify the task:

1. Extracts the text messages from the Facebook payload.
2. Calls save/save/agent/save/save in the right order.
3. Surfaces unexpected errors via the Celery failure path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.workers import tasks


def _payload() -> dict[str, Any]:
    return {
        "object": "page",
        "entry": [
            {
                "messaging": [
                    {
                        "sender": {"id": "user_celery_1"},
                        "message": {"text": "Xin chao"},
                    }
                ]
            }
        ],
    }


def test_process_incoming_message_skips_when_no_text_messages() -> None:
    payload = {"object": "page", "entry": [{"messaging": [{"sender": {"id": "u"}}]}]}

    result = tasks.process_incoming_message(payload)

    assert result == {"status": "skipped", "processed": 0}


def test_process_incoming_message_orchestrates_pipeline() -> None:
    payload = _payload()

    with patch.object(tasks, "init_conversation_schema", new=AsyncMock()), patch.object(
        tasks, "save_session_message", new=AsyncMock()
    ) as save_session, patch.object(
        tasks, "save_conversation_message", new=AsyncMock()
    ) as save_conv, patch.object(
        tasks, "generate_agent_result", new=AsyncMock()
    ) as gen, patch.object(
        tasks, "sync_hubspot_lead", new=AsyncMock()
    ) as hubspot, patch.object(
        tasks, "save_hubspot_sync_event", new=AsyncMock()
    ) as save_hubspot, patch.object(
        tasks, "send_telegram_notification", new=AsyncMock()
    ) as telegram, patch.object(
        tasks, "close_postgres", new=AsyncMock()
    ), patch.object(
        tasks, "close_redis", new=AsyncMock()
    ):
        gen.return_value = {
            "intent": "pricing",
            "action": "pricing_response",
            "response": "Cam on ban",
            "metadata": {"customer_name": "Tran"},
        }
        hubspot.return_value.status = "synced"
        hubspot.return_value.contact_id = "c-1"
        hubspot.return_value.action = "created"
        hubspot.return_value.reason = None

        result = tasks.process_incoming_message(payload)

    assert result == {"status": "processed", "processed": 1}
    # Two session writes (user + assistant), two conv writes.
    assert save_session.await_count == 2
    assert save_conv.await_count == 2
    save_hubspot.assert_awaited_once()
    telegram.assert_awaited_once()
