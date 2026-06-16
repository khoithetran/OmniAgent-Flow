"""Smoke test for the application factory (lifespan + healthcheck)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def test_app_healthcheck_runs_with_lifespan(env_override) -> None:
    env_override(WEBHOOK_VERIFY_TOKEN="unit-test-token")
    with patch("src.main.init_redis", new=AsyncMock()), patch(
        "src.main.close_redis", new=AsyncMock()
    ), patch("src.main.init_conversation_schema", new=AsyncMock()), patch(
        "src.main.close_postgres", new=AsyncMock()
    ):
        # Importing after patching ensures the module-level state is reset.
        import importlib

        import src.main as main_module
        importlib.reload(main_module)
        client = TestClient(main_module.app)
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
