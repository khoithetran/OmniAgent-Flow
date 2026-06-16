"""Tests for the HubSpot lead sync service."""

from __future__ import annotations

import os
from typing import Any

import pytest
from httpx import Request, Response

from src.config import get_settings
from src.services.hubspot_service import sync_hubspot_lead


class FakeHubSpotClient:
    def __init__(self, existing_contact_id: str | None = None) -> None:
        self.existing_contact_id = existing_contact_id
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    async def post(self, url: str, json: dict[str, Any]) -> Response:
        self.requests.append(("POST", url, json))
        request = Request("POST", f"https://api.hubapi.test{url}")
        if url.endswith("/search"):
            if self.existing_contact_id:
                return Response(
                    200,
                    json={"results": [{"id": self.existing_contact_id}]},
                    request=request,
                )
            return Response(200, json={"results": []}, request=request)
        return Response(201, json={"id": "contact_123"}, request=request)

    async def patch(self, url: str, json: dict[str, Any]) -> Response:
        self.requests.append(("PATCH", url, json))
        request = Request("PATCH", f"https://api.hubapi.test{url}")
        return Response(200, json={"id": "contact_123"}, request=request)


@pytest.fixture
def hubspot_enabled(env_override: Any) -> None:
    env_override(HUBSPOT_SYNC_ENABLED="true", HUBSPOT_ACCESS_TOKEN="fake-token")


@pytest.mark.asyncio
async def test_sync_hubspot_lead_creates_contact_with_fake_client(
    hubspot_enabled: None,
) -> None:
    client = FakeHubSpotClient()
    result = await sync_hubspot_lead(
        sender_id="sender_1",
        intent="pricing",
        action="pricing_response",
        metadata={
            "email": "lead@example.com",
            "customer_name": "Nguyen Van A",
            "company": "Acme",
        },
        client=client,
    )

    assert result.status == "synced"
    assert result.action == "created"
    assert result.contact_id == "contact_123"
    assert client.requests[0][1] == "/crm/v3/objects/contacts/search"
    assert client.requests[1][1] == "/crm/v3/objects/contacts"


@pytest.mark.asyncio
async def test_sync_hubspot_lead_updates_existing_contact(
    hubspot_enabled: None,
) -> None:
    client = FakeHubSpotClient(existing_contact_id="contact_456")
    result = await sync_hubspot_lead(
        sender_id="sender_1",
        intent="consultation",
        action="consultation_response",
        metadata={
            "phone": "0909000000",
            "customer_name": "Nguyen Van B",
        },
        client=client,
    )

    assert result.status == "synced"
    assert result.action == "updated"
    assert result.contact_id == "contact_456"
    assert client.requests[1][0] == "PATCH"
    assert client.requests[1][1] == "/crm/v3/objects/contacts/contact_456"


@pytest.mark.asyncio
async def test_sync_hubspot_lead_skipped_when_disabled() -> None:
    get_settings.cache_clear()
    os.environ["HUBSPOT_SYNC_ENABLED"] = "false"
    os.environ["HUBSPOT_ACCESS_TOKEN"] = "fake-token"
    get_settings.cache_clear()

    try:
        result = await sync_hubspot_lead(
            sender_id="sender_1",
            intent="pricing",
            action="pricing_response",
            metadata={"email": "lead@example.com"},
        )
    finally:
        os.environ["HUBSPOT_SYNC_ENABLED"] = "true"
        get_settings.cache_clear()

    assert result.status == "skipped"
    assert result.reason == "disabled"


@pytest.mark.asyncio
async def test_sync_hubspot_lead_skipped_when_no_identifier(
    hubspot_enabled: None,
) -> None:
    result = await sync_hubspot_lead(
        sender_id="sender_1",
        intent="handoff",
        action="handoff_response",
        metadata={"customer_name": "Anonymous"},
    )

    assert result.status == "skipped"
    assert result.reason == "missing_identifier"
