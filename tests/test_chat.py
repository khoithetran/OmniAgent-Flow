"""Tests for src/chat.py — chat orchestration (chat + chat_stream)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src import chat


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_includes_system_prompt(self):
        messages = chat._build_messages([], "", "hello")
        assert messages[0]["role"] == "system"

    def test_system_prompt_with_context(self):
        messages = chat._build_messages([], "[1] Some content", "hello")
        assert "[1] Some content" in messages[0]["content"]

    def test_system_prompt_without_context(self):
        # The static SYSTEM_PROMPT mentions "Knowledge Base" in its
        # instructions even when no context is provided.
        messages = chat._build_messages([], "", "hello")
        assert "Knowledge Base" in messages[0]["content"]

    def test_user_message_appended(self):
        messages = chat._build_messages([], "", "What is FastAPI?")
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is FastAPI?"

    def test_history_prepended(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        messages = chat._build_messages(history, "", "reply")
        assert messages[1]["content"] == "Hello"
        assert messages[2]["content"] == "Hi"


# ---------------------------------------------------------------------------
# _sanitise_history
# ---------------------------------------------------------------------------

class TestSanitiseHistory:
    def test_passes_valid_entries(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert len(chat._sanitise_history(history)) == 2

    def test_drops_invalid_role(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "unknown", "content": "bad"},
            {"role": "assistant", "content": "hi"},
        ]
        assert len(chat._sanitise_history(history)) == 2

    def test_drops_empty_content(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ]
        assert len(chat._sanitise_history(history)) == 1

    def test_drops_missing_content(self):
        history = [{"role": "user"}]
        assert len(chat._sanitise_history(history)) == 0


# ---------------------------------------------------------------------------
# _should_capture_email
# ---------------------------------------------------------------------------

class TestShouldCaptureEmail:
    def test_context_not_empty_no_email_request(self):
        assert chat._should_capture_email("[1] content", "Answer text") is False

    def test_context_empty_reply_has_email_keywords(self):
        assert (
            chat._should_capture_email("", "Xin lỗi, bạn có muốn để lại email không?")
            is True
        )

    def test_context_empty_reply_no_email_keywords(self):
        assert chat._should_capture_email("", "The sky is blue.") is False


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    def test_without_context(self):
        # The static prompt always mentions "Knowledge Base" in its body.
        prompt = chat._build_system_prompt("")
        assert "Knowledge Base" in prompt

    def test_with_context(self):
        prompt = chat._build_system_prompt("[1] Some text")
        assert "Knowledge Base" in prompt
        assert "[1] Some text" in prompt


# ---------------------------------------------------------------------------
# _call_openai (mocked)
# ---------------------------------------------------------------------------

class TestCallOpenai:
    @pytest.mark.asyncio
    async def test_returns_reply(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Reply text"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.chat.rag._get_openai", return_value=mock_client):
            with patch("src.chat.get_settings") as mock_settings:
                mock_settings.return_value.openai_model = "gpt-4o-mini"
                result = await chat._call_openai([{"role": "user", "content": "hi"}])

        assert result == "Reply text"

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=OSError("Network error"))

        with patch("src.chat.rag._get_openai", return_value=mock_client):
            with patch("src.chat.get_settings") as mock_settings:
                mock_settings.return_value.openai_model = "gpt-4o-mini"
                result = await chat._call_openai([])

        assert "Xin lỗi" in result


# ---------------------------------------------------------------------------
# _extract_delta
# ---------------------------------------------------------------------------

class TestExtractDelta:
    def test_valid_chunk(self):
        mock_delta = MagicMock(content="hello")
        mock_chunk = MagicMock(choices=[MagicMock(delta=mock_delta)])
        assert chat._extract_delta(mock_chunk) == "hello"

    def test_empty_choices(self):
        mock_chunk = MagicMock(choices=[])
        assert chat._extract_delta(mock_chunk) == ""

    def test_missing_delta(self):
        mock_chunk = MagicMock(choices=[MagicMock(delta=None)])
        assert chat._extract_delta(mock_chunk) == ""

    def test_missing_content_attr(self):
        mock_chunk = MagicMock(choices=[MagicMock(delta=MagicMock(spec=[]))])
        assert chat._extract_delta(mock_chunk) == ""


# ---------------------------------------------------------------------------
# _persist_turn (mocked)
# ---------------------------------------------------------------------------

class TestPersistTurn:
    @pytest.mark.asyncio
    async def test_saves_both_messages(self):
        with patch("src.session.save_message", new_callable=AsyncMock) as mock_save:
            await chat._persist_turn("sender123", "user msg", "assistant reply")
        assert mock_save.call_count == 2
        # Positional args: (sender_id, role, content)
        assert mock_save.call_args_list[0][0][1] == "user"
        assert mock_save.call_args_list[1][0][1] == "assistant"

    @pytest.mark.asyncio
    async def test_non_fatal_on_session_error(self):
        with patch(
            "src.chat.session.save_message",
            new_callable=AsyncMock,
            side_effect=OSError("Redis error"),
        ):
            # Must not raise
            await chat._persist_turn("sender123", "user msg", "assistant reply")


# ---------------------------------------------------------------------------
# chat() integration tests
# ---------------------------------------------------------------------------

class TestChat:
    @pytest.mark.asyncio
    async def test_missing_api_key_returns_fallback(self):
        with patch("src.chat.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key_value = None
            result = await chat.chat("sender1", "Hello")
        assert "chưa được cấu hình" in result

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_without_llm(self):
        with patch("src.chat.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key_value = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o-mini"

            with patch("src.chat.session.cache_get", new_callable=AsyncMock, return_value="cached reply"):
                with patch("src.chat.session.save_message", new_callable=AsyncMock):
                    with patch("src.chat.session.set_pending_email", new_callable=AsyncMock):
                        result = await chat.chat("sender1", "Same question")

        assert result == "cached reply"

    @pytest.mark.asyncio
    async def test_cache_miss_calls_llm(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Fresh reply"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.chat.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key_value = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o-mini"

            with patch("src.chat.session.cache_get", new_callable=AsyncMock, return_value=None):
                with patch("src.chat.session.cache_set", new_callable=AsyncMock):
                    with patch("src.chat.session.save_message", new_callable=AsyncMock):
                        with patch("src.chat.rag._get_openai", return_value=mock_client):
                            with patch("src.chat.rag.search", new_callable=AsyncMock, return_value=[]):
                                result = await chat.chat("sender1", "New question")

        assert result == "Fresh reply"

    @pytest.mark.asyncio
    async def test_sets_pending_email_on_fallback(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Xin lỗi, bạn có muốn để lại email không?"))
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.chat.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key_value = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o-mini"

            with patch("src.chat.session.cache_get", new_callable=AsyncMock, return_value=None):
                with patch("src.chat.session.cache_set", new_callable=AsyncMock):
                    with patch("src.chat.session.save_message", new_callable=AsyncMock):
                        with patch(
                            "src.chat.session.set_pending_email",
                            new_callable=AsyncMock,
                        ) as mock_set_email:
                            with patch("src.chat.rag._get_openai", return_value=mock_client):
                                with patch("src.chat.rag.search", new_callable=AsyncMock, return_value=[]):
                                    await chat.chat("sender1", "Random question")

        mock_set_email.assert_called_once_with("sender1")


# ---------------------------------------------------------------------------
# chat_stream() integration tests
# ---------------------------------------------------------------------------

class TestChatStream:
    @pytest.mark.asyncio
    async def test_cache_hit_yields_cached(self):
        with patch("src.chat.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key_value = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o-mini"

            with patch("src.chat.session.cache_get", new_callable=AsyncMock, return_value="cached stream reply"):
                with patch("src.chat.session.cache_set", new_callable=AsyncMock):
                    with patch("src.chat.session.save_message", new_callable=AsyncMock):
                        with patch("src.chat.session.set_pending_email", new_callable=AsyncMock):
                            chunks = [chunk async for chunk in chat.chat_stream("sender1", "Same question")]

        assert chunks == ["cached stream reply"]

    @pytest.mark.asyncio
    async def test_missing_api_key_yields_fallback(self):
        with patch("src.chat.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key_value = None
            chunks = [chunk async for chunk in chat.chat_stream("sender1", "Hello")]
        assert len(chunks) == 1
        assert "chưa được cấu hình" in chunks[0]

    @pytest.mark.asyncio
    async def test_fallback_to_non_streaming_on_error(self):
        """When streaming raises, _call_openai is used as fallback."""
        # First call (stream=True) raises; second call (_call_openai) succeeds.
        error_response = OSError("Stream error")
        success_response = MagicMock()
        success_response.choices = [MagicMock(message=MagicMock(content="Non-stream fallback"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[error_response, success_response]
        )

        with patch("src.chat.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key_value = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o-mini"

            with patch("src.chat.session.cache_get", new_callable=AsyncMock, return_value=None):
                with patch("src.chat.session.cache_set", new_callable=AsyncMock):
                    with patch("src.chat.session.save_message", new_callable=AsyncMock):
                        with patch("src.chat.session.set_pending_email", new_callable=AsyncMock):
                            with patch("src.chat.rag._get_openai", return_value=mock_client):
                                with patch("src.chat.rag.search", new_callable=AsyncMock, return_value=[]):
                                    chunks = [
                                        chunk
                                        async for chunk in chat.chat_stream("sender1", "Question")
                                    ]

        assert chunks[-1] == "Non-stream fallback"

    @pytest.mark.asyncio
    async def test_stream_yields_accumulated_text(self):
        """Simulate a streaming response with two chunks."""
        mock_response = AsyncMock()
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="world"))]

        mock_response.__aiter__ = lambda self: self
        mock_response.__anext__ = AsyncMock(side_effect=[chunk1, chunk2])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.chat.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key_value = "sk-test"
            mock_settings.return_value.openai_model = "gpt-4o-mini"

            with patch("src.chat.session.cache_get", new_callable=AsyncMock, return_value=None):
                with patch("src.chat.session.cache_set", new_callable=AsyncMock):
                    with patch("src.chat.session.save_message", new_callable=AsyncMock):
                        with patch("src.chat.rag._get_openai", return_value=mock_client):
                            with patch("src.chat.rag.search", new_callable=AsyncMock, return_value=[]):
                                chunks = [
                                    chunk
                                    async for chunk in chat.chat_stream("sender1", "hi")
                                ]

        # First chunk yields "Hello ", second yields "Hello world"
        assert "Hello" in chunks[0]
        assert chunks[-1] == "Hello world"
