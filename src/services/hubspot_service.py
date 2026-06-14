from typing import Any, Protocol

from httpx import AsyncClient, HTTPStatusError, Response
from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_settings


class HubSpotLeadSyncResult(BaseModel):
    status: str
    contact_id: str | None = None
    action: str | None = None
    reason: str | None = None


class HubSpotHTTPClient(Protocol):
    async def post(self, url: str, json: dict[str, Any]) -> Response:
        """Send a POST request to HubSpot."""

    async def patch(self, url: str, json: dict[str, Any]) -> Response:
        """Send a PATCH request to HubSpot."""


class HubSpotLeadPayload(BaseModel):
    email: str | None = None
    phone: str | None = None
    firstname: str | None = None
    lastname: str | None = None
    company: str | None = None
    lifecyclestage: str = "lead"
    source_sender_id: str
    intent: str
    action: str
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _split_customer_name(customer_name: str | None) -> tuple[str | None, str | None]:
    if not customer_name:
        return None, None

    name_parts = customer_name.strip().split()
    if not name_parts:
        return None, None
    if len(name_parts) == 1:
        return name_parts[0], None

    return " ".join(name_parts[:-1]), name_parts[-1]


def build_hubspot_lead_payload(
    sender_id: str,
    intent: str,
    action: str,
    metadata: dict[str, Any],
) -> HubSpotLeadPayload:
    firstname, lastname = _split_customer_name(metadata.get("customer_name"))
    return HubSpotLeadPayload(
        email=metadata.get("email"),
        phone=metadata.get("phone"),
        firstname=firstname,
        lastname=lastname,
        company=metadata.get("company"),
        source_sender_id=sender_id,
        intent=intent,
        action=action,
        summary=metadata.get("summary"),
        metadata=metadata,
    )


def _contact_properties(payload: HubSpotLeadPayload) -> dict[str, str]:
    properties: dict[str, str] = {
        "lifecyclestage": payload.lifecyclestage,
    }

    optional_properties: dict[str, str | None] = {
        "email": payload.email,
        "phone": payload.phone,
        "firstname": payload.firstname,
        "lastname": payload.lastname,
        "company": payload.company,
    }
    for property_name, value in optional_properties.items():
        if value:
            properties[property_name] = value

    return properties


def _search_filter_payload(property_name: str, value: str) -> dict[str, Any]:
    return {
        "filterGroups": [
            {
                "filters": [
                    {
                        "propertyName": property_name,
                        "operator": "EQ",
                        "value": value,
                    }
                ]
            }
        ],
        "properties": ["email", "phone", "firstname", "lastname", "company"],
        "limit": 1,
    }


async def _find_contact_id(
    client: HubSpotHTTPClient,
    payload: HubSpotLeadPayload,
) -> str | None:
    search_fields = [
        ("email", payload.email),
        ("phone", payload.phone),
    ]

    for property_name, value in search_fields:
        if not value:
            continue

        response = await client.post(
            "/crm/v3/objects/contacts/search",
            json=_search_filter_payload(property_name, value),
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if isinstance(results, list) and results:
            contact_id = results[0].get("id")
            if isinstance(contact_id, str):
                return contact_id

    return None


async def _create_contact(
    client: HubSpotHTTPClient,
    payload: HubSpotLeadPayload,
) -> str:
    response = await client.post(
        "/crm/v3/objects/contacts",
        json={"properties": _contact_properties(payload)},
    )
    response.raise_for_status()
    data = response.json()
    contact_id = data.get("id")
    if not isinstance(contact_id, str):
        raise ValueError("HubSpot create contact response did not include an id")

    return contact_id


async def _update_contact(
    client: HubSpotHTTPClient,
    contact_id: str,
    payload: HubSpotLeadPayload,
) -> None:
    response = await client.patch(
        f"/crm/v3/objects/contacts/{contact_id}",
        json={"properties": _contact_properties(payload)},
    )
    response.raise_for_status()


async def sync_hubspot_lead(
    sender_id: str,
    intent: str,
    action: str,
    metadata: dict[str, Any],
    client: HubSpotHTTPClient | None = None,
) -> HubSpotLeadSyncResult:
    settings = get_settings()
    access_token = settings.hubspot_access_token_value

    if not settings.hubspot_sync_enabled:
        logger.info("HubSpot lead sync skipped because it is disabled")
        return HubSpotLeadSyncResult(status="skipped", reason="disabled")
    if access_token is None:
        logger.warning("HubSpot lead sync skipped because access token is missing")
        return HubSpotLeadSyncResult(status="skipped", reason="missing_token")

    payload = build_hubspot_lead_payload(
        sender_id=sender_id,
        intent=intent,
        action=action,
        metadata=metadata,
    )
    if not payload.email and not payload.phone:
        logger.info(
            "HubSpot lead sync skipped because no email or phone was extracted",
            sender_id=sender_id,
            intent=intent,
        )
        return HubSpotLeadSyncResult(status="skipped", reason="missing_identifier")

    owns_client = client is None
    resolved_client = client or AsyncClient(
        base_url=settings.hubspot_base_url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        timeout=settings.hubspot_timeout_seconds,
    )

    try:
        contact_id = await _find_contact_id(resolved_client, payload)
        if contact_id is None:
            contact_id = await _create_contact(resolved_client, payload)
            action_taken = "created"
        else:
            await _update_contact(resolved_client, contact_id, payload)
            action_taken = "updated"

        logger.info(
            "Synced lead to HubSpot",
            sender_id=sender_id,
            contact_id=contact_id,
            action=action_taken,
            intent=intent,
        )
        return HubSpotLeadSyncResult(
            status="synced",
            contact_id=contact_id,
            action=action_taken,
        )
    except HTTPStatusError as exc:
        logger.exception(
            "HubSpot API returned an error",
            sender_id=sender_id,
            status_code=exc.response.status_code,
            response_text=exc.response.text,
        )
        return HubSpotLeadSyncResult(status="failed", reason="hubspot_http_error")
    except Exception:
        logger.exception("Failed to sync lead to HubSpot", sender_id=sender_id)
        return HubSpotLeadSyncResult(status="failed", reason="unexpected_error")
    finally:
        if owns_client and isinstance(resolved_client, AsyncClient):
            await resolved_client.aclose()
