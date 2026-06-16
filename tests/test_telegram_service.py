"""Tests for the Telegram notification service."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import Request, Response

from src.services.telegram_service import (
    _format_event_message,
    send_telegram_notification,
)


class FakeTelegramClient:
    def __init__(self, *, status_code: int = 200, message_id: int | None = 99) -> None:
        self.status_code = status_code
        self.message_id = message_id
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def post(self, url: str, json: dict[str, Any]) -> Response:
        self.calls.append((url, json))
        request = Request("POST", url)
        if self.status_code >= 400:
            return Response(self.status_code, request=request, text="boom")
        return Response(
            self.status_code,
            request=request,
            json={"ok": True, "result": {"message_id": self.message_id}},
        )


@pytest.fixture
def telegram_enabled(env_override: Any) -> None:
    env_override(
        TELEGRAM_NOTIFICATIONS_ENABLED="true",
        TELEGRAM_BOT_TOKEN="bot-xyz",
        TELEGRAM_CHAT_ID="-100200300",
    )


def test_format_event_message_escapes_user_input() -> None:
    text = _format_event_message(
        event_type="new_message",
        sender_id="<user>",
        intent="pricing",
        action="pricing_response",
        metadata={"customer_name": "<script>"},
    )
    # Raw HTML tags must be escaped before reaching Telegram.
    assert "<user>" not in text
    assert "&lt;user&gt;" in text
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_format_event_message_includes_hubspot_block() -> None:
    text = _format_event_message(
        event_type="hubspot_sync_failed",
        sender_id="sender_1",
        intent="pricing",
        action="pricing_response",
        metadata={"company": "Acme"},
        hubspot_status="failed",
        hubspot_contact_id=None,
    )
    assert "HubSpot: failed" in text
    assert "Acme" in text


@pytest.mark.asyncio
async def test_send_telegram_notification_calls_send_message(
    telegram_enabled: None,
) -> None:
    fake = FakeTelegramClient()
    result = await send_telegram_notification(
        event_type="new_message",
        sender_id="sender_1",
        intent="pricing",
        action="pricing_response",
        metadata={"customer_name": "Tran"},
        client=fake,
    )

    assert result.status == "sent"
    assert result.chat_id == "-100200300"
    assert result.message_id == 99
    assert len(fake.calls) == 1
    url, payload = fake.calls[0]
    assert url.endswith("/botbot-xyz/sendMessage")
    assert payload["chat_id"] == "-100200300"
    assert payload["parse_mode"] == "HTML"
    assert "Tran" in payload["text"]


@pytest.mark.asyncio
async def test_send_telegram_notification_returns_skipped_when_disabled(
    env_override: Any,
) -> None:
    env_override(TELEGRAM_NOTIFICATIONS_ENABLED="false")
    result = await send_telegram_notification(
        event_type="new_message",
        sender_id="sender_1",
        intent="pricing",
        action="pricing_response",
        metadata={},
    )
    assert result.status == "skipped"
    assert result.reason == "disabled"


@pytest.mark.asyncio
async def test_send_telegram_notification_returns_skipped_when_missing_config(
    env_override: Any,
) -> None:
    env_override(
        TELEGRAM_NOTIFICATIONS_ENABLED="true",
        TELEGRAM_BOT_TOKEN="",
        TELEGRAM_CHAT_ID="",
    )
    result = await send_telegram_notification(
        event_type="new_message",
        sender_id="sender_1",
        intent="pricing",
        action="pricing_response",
        metadata={},
    )
    assert result.status == "skipped"
    assert result.reason == "missing_config"


@pytest.mark.asyncio
async def test_send_telegram_notification_handles_http_error(
    telegram_enabled: None,
) -> None:
    fake = FakeTelegramClient(status_code=500)
    result = await send_telegram_notification(
        event_type="new_message",
        sender_id="sender_1",
        intent="handoff",
        action="handoff_response",
        metadata={},
        client=fake,
    )
    assert result.status == "failed"
    assert result.reason == "telegram_http_error"
