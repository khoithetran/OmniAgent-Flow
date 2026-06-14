import asyncio
import os
from typing import Any

from httpx import Request, Response

from src.config import get_settings
from src.services.hubspot_service import (
    build_hubspot_lead_payload,
    sync_hubspot_lead,
)


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


def test_build_hubspot_payload_maps_standard_contact_fields() -> None:
    payload = build_hubspot_lead_payload(
        sender_id="sender_1",
        intent="pricing",
        action="pricing_response",
        metadata={
            "email": "lead@example.com",
            "phone": "0909000000",
            "customer_name": "Nguyen Van A",
            "company": "Acme",
            "summary": "Can bao gia",
        },
    )

    assert payload.email == "lead@example.com"
    assert payload.firstname == "Nguyen Van"
    assert payload.lastname == "A"
    assert payload.company == "Acme"


async def test_sync_hubspot_lead_creates_contact_with_fake_client() -> None:
    previous_enabled = os.environ.get("HUBSPOT_SYNC_ENABLED")
    previous_token = os.environ.get("HUBSPOT_ACCESS_TOKEN")
    os.environ["HUBSPOT_SYNC_ENABLED"] = "true"
    os.environ["HUBSPOT_ACCESS_TOKEN"] = "fake-token"
    get_settings.cache_clear()

    try:
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
    finally:
        if previous_enabled is None:
            os.environ.pop("HUBSPOT_SYNC_ENABLED", None)
        else:
            os.environ["HUBSPOT_SYNC_ENABLED"] = previous_enabled

        if previous_token is None:
            os.environ.pop("HUBSPOT_ACCESS_TOKEN", None)
        else:
            os.environ["HUBSPOT_ACCESS_TOKEN"] = previous_token

        get_settings.cache_clear()

    assert result.status == "synced"
    assert result.action == "created"
    assert result.contact_id == "contact_123"
    assert client.requests[0][1] == "/crm/v3/objects/contacts/search"
    assert client.requests[1][1] == "/crm/v3/objects/contacts"


async def test_sync_hubspot_lead_updates_existing_contact_with_fake_client() -> None:
    previous_enabled = os.environ.get("HUBSPOT_SYNC_ENABLED")
    previous_token = os.environ.get("HUBSPOT_ACCESS_TOKEN")
    os.environ["HUBSPOT_SYNC_ENABLED"] = "true"
    os.environ["HUBSPOT_ACCESS_TOKEN"] = "fake-token"
    get_settings.cache_clear()

    try:
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
    finally:
        if previous_enabled is None:
            os.environ.pop("HUBSPOT_SYNC_ENABLED", None)
        else:
            os.environ["HUBSPOT_SYNC_ENABLED"] = previous_enabled

        if previous_token is None:
            os.environ.pop("HUBSPOT_ACCESS_TOKEN", None)
        else:
            os.environ["HUBSPOT_ACCESS_TOKEN"] = previous_token

        get_settings.cache_clear()

    assert result.status == "synced"
    assert result.action == "updated"
    assert result.contact_id == "contact_456"
    assert client.requests[1][0] == "PATCH"
    assert client.requests[1][1] == "/crm/v3/objects/contacts/contact_456"


async def main() -> None:
    test_build_hubspot_payload_maps_standard_contact_fields()
    await test_sync_hubspot_lead_creates_contact_with_fake_client()
    await test_sync_hubspot_lead_updates_existing_contact_with_fake_client()


if __name__ == "__main__":
    asyncio.run(main())
