"""Tests for the FastAPI webhook endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client(env_override: Any) -> TestClient:
    env_override(WEBHOOK_VERIFY_TOKEN="unit-test-token")
    return TestClient(app)


def test_verify_webhook_returns_challenge_when_token_matches(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "unit-test-token",
            "hub.challenge": "CHALLENGE_123",
        },
    )

    assert response.status_code == 200
    assert response.text == "CHALLENGE_123"


def test_verify_webhook_rejects_invalid_token(client: TestClient) -> None:
    response = client.get(
        "/api/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "CHALLENGE_123",
        },
    )

    assert response.status_code == 403


def test_receive_webhook_enqueues_celery_task(client: TestClient) -> None:
    payload: dict[str, Any] = {
        "object": "page",
        "entry": [
            {
                "messaging": [
                    {
                        "sender": {"id": "user_test_1"},
                        "message": {"text": "Tôi cần tư vấn"},
                    }
                ]
            }
        ]
    }

    with patch("src.api.webhook.process_incoming_message") as mocked_task:
        mocked_task.delay.return_value.id = "celery-task-42"
        response = client.post("/api/webhook", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["task_id"] == "celery-task-42"
    mocked_task.delay.assert_called_once_with(payload)


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "OmniAgent Flow"}
