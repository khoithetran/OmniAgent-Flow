"""Quick smoke tests for app_gradio handlers.

Verifies that:
- handle_fetch validates URL and rejects malformed inputs
- handle_chat returns gracefully when OpenAI is not configured
- handle_chat returns NOT_FOUND_REPLY when RAG returns empty context
- handle_clear_kb resets the state and adds the warning message
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent))

from app_gradio import (
    GENERAL_MISSING_KEY,
    NOT_FOUND_REPLY,
    WARNING_AFTER_CLEAR,
    handle_chat,
    handle_clear_kb,
    handle_fetch,
)


def _empty_state() -> dict:
    return {
        "kb_ready": False,
        "kb_domain": "",
        "kb_pages": 0,
        "kb_chunks": 0,
        "kb_url": "",
        "selected_model": "gpt-4o-mini",
    }


async def test_handle_fetch_invalid_url():
    """Invalid URL should return error in status, no state change."""
    state = _empty_state()
    new_state, status, _, _ = await handle_fetch(state, "not a url")
    assert "URL phai bat dau bang" in status or "URL" in status, f"Got: {status}"
    assert new_state == state, "State should not change on validation error"
    print("PASS test_handle_fetch_invalid_url")


async def test_handle_fetch_empty_url():
    state = _empty_state()
    new_state, status, _, _ = await handle_fetch(state, "")
    assert "Vui long nhap URL" in status or "URL" in status, f"Got: {status}"
    print("PASS test_handle_fetch_empty_url")


async def test_handle_chat_no_api_key_no_kb():
    """When no API key is configured, bot returns the missing-key reply."""
    state = _empty_state()
    history: list = []
    with patch("app_gradio.get_settings") as mock_settings:
        mock_settings.return_value.openai_api_key_value = None
        out_history = None
        async for h, s, _usage in handle_chat("hello", history, state):
            out_history = h
            if len(h) >= 2:
                break
    assert out_history is not None
    assert out_history[-1]["role"] == "assistant"
    # Use a substring that's safe across encodings.
    assert "OpenAI key" in out_history[-1]["content"], (
        f"Got: {out_history[-1]['content']}"
    )
    print("PASS test_handle_chat_no_api_key_no_kb")


async def test_handle_chat_rag_empty_context():
    """RAG mode + empty Qdrant context -> NOT_FOUND_REPLY."""
    state = {
        **_empty_state(),
        "kb_ready": True,
        "kb_domain": "example.com",
    }
    history: list = []

    # Mock RAG search to return empty hits.
    with patch("src.rag.search", new=AsyncMock(return_value=[])):
        with patch("app_gradio.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key_value = "fake-key"
            out_history = None
            async for h, s, _usage in handle_chat("what is X?", history, state):
                out_history = h
                if len(h) >= 2:
                    break
    assert out_history is not None
    assert out_history[-1]["content"] == NOT_FOUND_REPLY, (
        f"Got: {out_history[-1]['content']}"
    )
    print("PASS test_handle_chat_rag_empty_context")


async def test_handle_clear_kb():
    """clear KB should reset state and add warning message."""
    state = {
        **_empty_state(),
        "kb_ready": True,
        "kb_domain": "stripe.com",
        "kb_pages": 12,
        "kb_chunks": 76,
        "kb_url": "https://stripe.com",
    }
    new_state, status, history = await handle_clear_kb(state)
    assert new_state["kb_ready"] is False
    assert new_state["kb_domain"] == ""
    assert new_state["kb_pages"] == 0
    assert new_state["kb_chunks"] == 0
    assert new_state["kb_url"] == ""
    # Match the canonical warning prefix using ASCII-safe substring.
    assert WARNING_AFTER_CLEAR in history[0]["content"], f"Got: {history[0]['content']}"
    print("PASS test_handle_clear_kb")


async def main():
    await test_handle_fetch_invalid_url()
    await test_handle_fetch_empty_url()
    await test_handle_chat_no_api_key_no_kb()
    await test_handle_chat_rag_empty_context()
    await test_handle_clear_kb()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
