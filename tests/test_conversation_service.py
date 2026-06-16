"""Tests for the conversation service (PostgreSQL schema) and
shared HubSpot payload helpers (kept here for historical reasons)."""

from __future__ import annotations

from src.services.conversation_service import SCHEMA_STATEMENTS
from src.services.hubspot_service import (
    HubSpotLeadPayload,
    _contact_properties,
    _split_customer_name,
    build_hubspot_lead_payload,
)


def test_conversation_schema_declares_required_tables() -> None:
    schema_sql = "\n".join(SCHEMA_STATEMENTS)

    assert "CREATE TABLE IF NOT EXISTS conversations" in schema_sql
    assert "CREATE TABLE IF NOT EXISTS conversation_messages" in schema_sql
    assert "CREATE TABLE IF NOT EXISTS hubspot_lead_syncs" in schema_sql
    assert "metadata JSONB" in schema_sql
    assert "last_intent TEXT" in schema_sql


def test_split_customer_name_handles_single_word() -> None:
    assert _split_customer_name("Tran") == ("Tran", None)


def test_split_customer_name_handles_multiple_words() -> None:
    assert _split_customer_name("Nguyen Van A") == ("Nguyen Van", "A")


def test_split_customer_name_returns_none_for_empty() -> None:
    assert _split_customer_name(None) == (None, None)


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

    assert isinstance(payload, HubSpotLeadPayload)
    assert payload.email == "lead@example.com"
    assert payload.phone == "0909000000"
    assert payload.firstname == "Nguyen Van"
    assert payload.lastname == "A"
    assert payload.company == "Acme"
    assert payload.lifecyclestage == "lead"


def test_contact_properties_omits_empty_optional_fields() -> None:
    payload = HubSpotLeadPayload(
        source_sender_id="sender_1",
        intent="handoff",
        action="handoff_response",
        email="lead@example.com",
    )

    properties = _contact_properties(payload)

    assert properties == {
        "lifecyclestage": "lead",
        "email": "lead@example.com",
    }
