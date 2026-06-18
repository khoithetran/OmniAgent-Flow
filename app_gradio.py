"""Gradio web interface for the OmniAgent Flow chatbot.

Replaces (extends) the Telegram bot with a browser-based chat UI. The
flow is intentionally simple:

- Left column (~75%): chat history, model selector, message input.
- Right column (~25%): URL input, Fetch button, KB status, X button to
  clear the KB.

When no KB is loaded, the chat uses a general LLM (``SYSTEM_PROMPT_GENERAL``).
When a URL has been fetched and indexed, the chat switches to RAG mode
(``SYSTEM_PROMPT_RAG``) and refuses to invent answers outside the document.

Streaming is wired through Gradio's native ``type="messages"`` Chatbot +
``stream`` event so the user sees tokens appear in real time without the
throttling required by the Telegram webhook.

Run with::

    python app_gradio.py

The default port is 7860. Set ``GRADIO_SERVER_PORT`` to override.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from loguru import logger

# Allow ``python app_gradio.py`` to find the ``src`` package.
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr  # noqa: E402

from src import chat as chat_module  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.rag import index_crawl_results  # noqa: E402

# Use lightweight CPU crawler (no GPU/Chromium needed).
# Falls back to crawl4ai when available (local dev).
try:
    from src.simple_crawler import crawl_full_website  # noqa: E402
except ImportError:
    from src.crawler import crawl_full_website  # noqa: E402


# ---------------------------------------------------------------------------
# Logging (mirror the FastAPI main.py setup to avoid cp1252 issues on Windows)
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stdout,
    backtrace=True,
    diagnose=False,
    enqueue=True,
    serialize=True,
    format="{time:HH:mm:ss} | {level} | {message}",
    colorize=False,
)


# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------

#: Available models for the Gradio dropdown. Each entry is the
#: identifier passed to the OpenAI Chat Completions API.
#:
#: Tested against OpenAI API (June 2026):
#: - ``gpt-4o-mini``: supported, uses ``max_tokens``.
#: - ``gpt-4o``: supported, uses ``max_tokens``.
#: - ``o4-mini``: supported (reasoning model), uses ``max_completion_tokens``.
MODELS: list[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "o4-mini",
]

DEFAULT_MODEL = "gpt-4o-mini"

#: Maximum context window (in tokens) for each model. The values
#: are surfaced in the UI so the user knows the model capacity.
CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "o4-mini": 200_000,
}


def _format_context_window(tokens: int) -> str:
    """Return a human-friendly string for a context window size."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M tokens"
    if tokens >= 1_000:
        return f"{tokens // 1_000}k tokens"
    return f"{tokens} tokens"


def _context_window_label(model: str) -> str:
    """Return a Markdown label describing the selected model + its window."""
    window = CONTEXT_WINDOWS.get(model)
    if window is None:
        return f"**Model:** `{model}`"
    return f"**Model:** `{model}` • Context: {_format_context_window(window)}"


# ---------------------------------------------------------------------------
# System prompts (re-exported for clarity; actual prompts live in src.chat)
# ---------------------------------------------------------------------------

WARNING_AFTER_CRAWL = (
    "⚠️ **Lưu ý**: Tôi chỉ trả lời dựa trên nội dung tài liệu đã cung cấp. "
    "Tôi sẽ không trả lời các câu hỏi ngoài phạm vi này."
)

WARNING_AFTER_CLEAR = (
    "⚠️ **Đã xóa tài liệu**. Tôi đang ở chế độ kiến thức chung. "
    "Các câu hỏi về công ty có thể không chính xác."
)

NOT_FOUND_REPLY = (
    "Không tìm thấy thông tin này trong tài liệu được cung cấp."
)

GENERAL_MISSING_KEY = (
    "Xin lỗi, hệ thống hiện chưa được cấu hình OpenAI key. "
    "Vui lòng liên hệ quản trị viên."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_url(url: str) -> str | None:
    """Return an error message if the URL is malformed, else None."""
    if not url or not url.strip():
        return "Vui lòng nhập URL."
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "URL phải bắt đầu bằng http:// hoặc https://"
    parsed = urlparse(url)
    if not parsed.netloc or "." not in parsed.netloc:
        return "URL không hợp lệ (thiếu domain)."
    return None


def _domain_from_url(url: str) -> str:
    """Extract a display-friendly domain from a URL."""
    try:
        return urlparse(url.strip()).netloc or url
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Async handlers
# ---------------------------------------------------------------------------


async def handle_fetch(
    state: dict[str, Any],
    url: str,
) -> tuple[dict[str, Any], str, str, list[dict[str, str]]]:
    """Crawl ``url`` and index it into Qdrant.

    Returns the updated ``(state, status_text, url_input_value, history)``.
    History is updated with a system warning message so the user sees it
    in the chat.
    """
    err = _validate_url(url)
    if err is not None:
        return (
            state,
            f"⚠️ {err}",
            url,  # keep what the user typed so they can fix it
            [],
        )

    url = url.strip()
    domain = _domain_from_url(url)
    logger.info("Fetch started", sender_id="gradio", url=url)

    try:
        results = await crawl_full_website(url, max_pages=20)
    except Exception:  # noqa: BLE001
        logger.exception("Crawl failed", url=url)
        return (
            state,
            f"❌ Không truy cập được URL này. Kiểm tra kết nối hoặc thử lại.",
            url,
            [],
        )

    pages_ok = sum(1 for r in results if r.success)
    if pages_ok == 0:
        return (
            state,
            "❌ Không crawl được trang nào. Thử URL khác.",
            url,
            [],
        )

    try:
        summary = await index_crawl_results(results, replace=True)
    except Exception:  # noqa: BLE001
        logger.exception("Index failed", url=url)
        return (
            state,
            "❌ Index thất bại. Không thể lưu tài liệu vào vector store.",
            url,
            [],
        )

    new_state: dict[str, Any] = {
        **state,
        "kb_ready": True,
        "kb_domain": domain,
        "kb_pages": summary["pages"],
        "kb_chunks": summary["chunks"],
        "kb_url": url,
    }

    status = (
        f"✅ {summary['pages']} trang, {summary['chunks']} chunks indexed. "
        f"Sẵn sàng tra cứu."
    )
    new_history = [{"role": "assistant", "content": WARNING_AFTER_CRAWL}]

    logger.info(
        "Fetch complete",
        url=url,
        pages=summary["pages"],
        chunks=summary["chunks"],
    )

    return (new_state, status, "", new_history)


async def handle_clear_kb(
    state: dict[str, Any],
) -> tuple[dict[str, Any], str, list[dict[str, str]]]:
    """Drop the indexed KB and reset the bot to general mode."""
    new_state: dict[str, Any] = {
        **state,
        "kb_ready": False,
        "kb_domain": "",
        "kb_pages": 0,
        "kb_chunks": 0,
        "kb_url": "",
    }
    new_history = [{"role": "assistant", "content": WARNING_AFTER_CLEAR}]
    return (new_state, "Đã xóa tài liệu. Có thể fetch URL mới.", new_history)


async def handle_chat(
    user_message: str,
    history: list[dict[str, str]],
    state: dict[str, Any],
) -> AsyncIterator[tuple[list[dict[str, str]], dict[str, Any], str]]:
    """Yield streaming chat updates for the Gradio Chatbot.

    Each yield is ``(history, state, usage_text)`` where ``usage_text``
    is a Markdown-formatted "used / total tokens" string for the
    context window display. The first yield appends the user message
    so the UI shows it immediately. Subsequent yields replace the last
    assistant turn with the accumulating reply.
    """
    selected_model = state.get("selected_model") or DEFAULT_MODEL
    total_window = CONTEXT_WINDOWS.get(selected_model, 128_000)

    def _usage(used: int) -> str:
        return _format_usage(used, total_window)

    if not user_message or not user_message.strip():
        yield history, state, _usage(0)
        return

    user_message = user_message.strip()
    sender_id = "gradio-default"  # Could be per-session in the future
    kb_ready = bool(state.get("kb_ready"))

    # 1. Append the user message immediately so the UI reflects it.
    history = history + [{"role": "user", "content": user_message}]
    # Rough estimate: system + history + user message so far.
    used_est = sum(_estimate_tokens(m.get("content", "")) for m in history)
    yield history, state, _usage(used_est)

    # 2. RAG mode: if KB is empty after retrieval, return NOT_FOUND_REPLY.
    if kb_ready and not state.get("kb_domain"):
        # Defensive: state says ready but no domain — treat as no KB.
        kb_ready = False

    if not get_settings().openai_api_key_value:
        history = history + [{"role": "assistant", "content": GENERAL_MISSING_KEY}]
        used_est = sum(_estimate_tokens(m.get("content", "")) for m in history)
        yield history, state, _usage(used_est)
        return

    # 3. RAG pre-check: if KB is ready, try retrieval. If empty,
    #    short-circuit with NOT_FOUND_REPLY (don't even call LLM).
    if kb_ready:
        try:
            from src.rag import search, format_context
            hits = await search(user_message)
            context_block = format_context(hits)
        except Exception:  # noqa: BLE001
            logger.exception("RAG search failed")
            context_block = ""

        if not context_block.strip():
            history = history + [
                {"role": "assistant", "content": NOT_FOUND_REPLY}
            ]
            used_est = sum(_estimate_tokens(m.get("content", "")) for m in history)
            yield history, state, _usage(used_est)
            return

    # 4. Stream the LLM reply token by token.
    accumulated = ""
    try:
        stream_iter = (
            chat_module.chat_rag_stream(sender_id, user_message, model=selected_model)
            if kb_ready
            else chat_module.chat_general_stream(sender_id, user_message, model=selected_model)
        )
        async for partial in stream_iter:
            accumulated = partial
            # Replace the last assistant turn with the partial text.
            if history and history[-1].get("role") == "assistant":
                history = history[:-1] + [
                    {"role": "assistant", "content": accumulated}
                ]
            else:
                history = history + [
                    {"role": "assistant", "content": accumulated}
                ]
            used_est = sum(_estimate_tokens(m.get("content", "")) for m in history)
            yield history, state, _usage(used_est)
    except Exception:  # noqa: BLE001
        logger.exception("Streaming failed")
        history = history + [
            {
                "role": "assistant",
                "content": "Xin lỗi, đã xảy ra lỗi khi gọi LLM. Vui lòng thử lại.",
            }
        ]
        used_est = sum(_estimate_tokens(m.get("content", "")) for m in history)
        yield history, state, _usage(used_est)


def select_model(
    state: dict[str, Any],
    model: str,
) -> tuple[dict[str, Any]]:
    """Update the selected model in state."""
    new_state = {**state, "selected_model": model}
    return (new_state,)


# ---------------------------------------------------------------------------
# UI builders
# ---------------------------------------------------------------------------


def _build_model_buttons() -> list[gr.Button]:
    """Build the model selector row as a list of Gradio buttons."""
    buttons = []
    for model_name in MODELS:
        btn = gr.Button(
            model_name,
            variant="secondary",
            size="sm",
            min_width=120,
        )
        buttons.append(btn)
    return buttons


def _button_variants(selected: str) -> list[gr.update]:
    """Return a list of ``gr.update`` for each model button.

    The button matching ``selected`` gets ``variant="primary"``; all
    other buttons get ``variant="secondary"``. Use this in click
    handlers to keep the highlight in sync.
    """
    return [
        gr.update(variant="primary" if m == selected else "secondary")
        for m in MODELS
    ]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for a text string.

    Uses the common 1 token ≈ 4 chars heuristic for English/ Vietnamese
    mixed text. Not exact but close enough for the usage display.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _format_usage(used: int, total: int) -> str:
    """Format a usage display string like ``~3.2k / 128k tokens``."""
    def _fmt(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 10_000:
            return f"{n // 1_000}k"
        if n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return str(n)

    pct = (used / total * 100) if total > 0 else 0
    return f"~{_fmt(used)} / {_fmt(total)} tokens ({pct:.1f}%)"


def build_ui() -> gr.Blocks:
    """Construct the full Gradio app."""
    initial_state: dict[str, Any] = {
        "kb_ready": False,
        "kb_domain": "",
        "kb_pages": 0,
        "kb_chunks": 0,
        "kb_url": "",
        "selected_model": DEFAULT_MODEL,
    }

    # Gradio 6+: theme must be passed to launch(), not Blocks().
    with gr.Blocks(
        title="OmniAgent Flow",
    ) as demo:
        state = gr.State(value=initial_state)

        with gr.Row():
            # ---------------- Left column: chat ----------------
            with gr.Column(scale=3):
                # Gradio 6: Chatbot uses 'messages' format by default
                # (no need for type="messages").
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_label=False,
                    value=[
                        {
                            "role": "assistant",
                            "content": (
                                "Xin chào! Tôi có thể giúp gì cho bạn?\n"
                                "Nếu bạn cần thông tin chính xác từ nguồn có sẵn, "
                                "hãy dùng chức năng Fetch & Index trước khi đặt câu hỏi."
                            ),
                        }
                    ],
                )

                # Model selector row.
                with gr.Row():
                    model_buttons = _build_model_buttons()

                # Context window display — shows the active model's
                # context window size. Updated when a model button is
                # clicked.
                context_window_display = gr.Markdown(
                    value=_context_window_label(DEFAULT_MODEL),
                )

                # Token usage display — shows used / total tokens for
                # the current request. Updated after every chat turn.
                token_usage_display = gr.Markdown(
                    value=_format_usage(0, CONTEXT_WINDOWS[DEFAULT_MODEL]),
                )

                # Input row.
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Nhập câu hỏi...",
                        show_label=False,
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("Gửi", variant="primary", scale=1)

            # ---------------- Right column: control panel ----------------
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Nguồn tài liệu")

                # Status area (always visible).
                status = gr.Markdown("⚠️ Chưa có tài liệu.")

                # Domain display + X button (visible only when KB is ready).
                with gr.Row(visible=False) as kb_row:
                    domain_display = gr.Markdown("🔗 ...")
                    clear_x_btn = gr.Button("✕", variant="stop", size="sm", scale=0)

                clear_kb_btn = gr.Button(
                    "Clear KB",
                    variant="stop",
                    visible=False,
                )

                gr.Markdown("---")
                gr.Markdown("**URL**")
                url_input = gr.Textbox(
                    placeholder="https://stripe.com",
                    show_label=False,
                )
                fetch_btn = gr.Button("Fetch & Index", variant="primary")

        # ---------------- Event wiring ----------------

        # 1. Model buttons: click to set active model.
        # One handler per button — updates state, then propagates the new
        # selected_model to ALL model buttons (only the matching one becomes
        # "primary") and refreshes the context window display.
        def _make_select_handler(m: str):
            def _handler(s: dict[str, Any]):
                new_state = select_model(s, m)[0]
                variants = _button_variants(m)  # list of len(MODELS)
                label = _context_window_label(m)
                # Unpack the variants list so we return 1 value per output:
                # state, btn1, btn2, btn3, btn4, context_window_display.
                return (new_state, *variants, label)
            return _handler

        for btn, model_name in zip(model_buttons, MODELS):
            # Each click must update ALL model buttons (only the matching
            # one becomes "primary"). We splat the list of buttons into
            # the outputs sequence; the handler returns a list of
            # ``gr.update`` of the same length.
            outputs = [state, *model_buttons, context_window_display]
            btn.click(
                fn=_make_select_handler(model_name),
                inputs=[state],
                outputs=outputs,
            )

        # 2. Fetch URL.
        fetch_btn.click(
            fn=handle_fetch,
            inputs=[state, url_input],
            outputs=[state, status, url_input, chatbot],
        ).then(
            fn=lambda s: gr.update(visible=bool(s.get("kb_ready"))),
            inputs=[state],
            outputs=[kb_row],
        ).then(
            fn=lambda s: gr.update(visible=bool(s.get("kb_ready"))),
            inputs=[state],
            outputs=[clear_kb_btn],
        ).then(
            fn=lambda s: (
                f"🔗 **{s.get('kb_domain', '')}**"
                if s.get("kb_ready")
                else "⚠️ Chưa có tài liệu."
            ),
            inputs=[state],
            outputs=[domain_display],
        )

        # 3. Clear KB via X button.
        clear_x_btn.click(
            fn=handle_clear_kb,
            inputs=[state],
            outputs=[state, status, chatbot],
        ).then(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[kb_row],
        ).then(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[clear_kb_btn],
        ).then(
            fn=lambda: "⚠️ Chưa có tài liệu.",
            inputs=[],
            outputs=[domain_display],
        )

        # 4. Clear KB via "Clear KB" button (same handler).
        clear_kb_btn.click(
            fn=handle_clear_kb,
            inputs=[state],
            outputs=[state, status, chatbot],
        ).then(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[kb_row],
        ).then(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[clear_kb_btn],
        ).then(
            fn=lambda: "⚠️ Chưa có tài liệu.",
            inputs=[],
            outputs=[domain_display],
        )

        # 5. Chat: Enter or Send button.
        chat_outputs = [chatbot, state, token_usage_display]
        send_event = msg.submit(
            fn=handle_chat,
            inputs=[msg, chatbot, state],
            outputs=chat_outputs,
        )
        send_btn.click(
            fn=handle_chat,
            inputs=[msg, chatbot, state],
            outputs=chat_outputs,
        )

        # 6. Clear input after send.
        msg.submit(fn=lambda: "", inputs=[], outputs=[msg])
        send_btn.click(fn=lambda: "", inputs=[], outputs=[msg])

    return demo


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


async def _init_clients() -> None:
    """Initialise Redis / Qdrant / OpenAI clients once on startup.

    Redis and Qdrant are best-effort: if unavailable (e.g. HF Spaces
    without a managed service), the app falls back to in-memory session
    and general-LLM mode respectively.
    """
    import src.main as main_mod
    from src.rag import init_openai, init_qdrant

    if main_mod.redis_client is None:
        try:
            from redis.asyncio import Redis
            settings = get_settings()
            client = Redis.from_url(
                f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
                decode_responses=True,
            )
            await client.ping()
            main_mod.redis_client = client
            logger.info("Redis connected", host=settings.redis_host)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Redis unavailable; using in-memory session store"
            )

    qdrant = init_qdrant()
    init_openai()
    if qdrant is not None:
        logger.info("Qdrant connected")
    else:
        logger.warning("Qdrant unavailable; RAG disabled (general LLM mode)")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def _health() -> dict[str, str]:
    """Readiness probe — returns 200 once clients have been initialised."""
    return {"status": "ok", "service": get_settings().app_name}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()
    logger.info("Starting Gradio app", app_name=settings.app_name)

    # Init clients synchronously (no async context manager in plain Gradio).
    asyncio.run(_init_clients())

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo = build_ui()

    # Mount a minimal /health route so container orchestrators (Railway,
    # Render, Fly.io, Kubernetes, etc.) can confirm readiness.
    # Gradio exposes its underlying FastAPI app as demo.app.
    demo.app.add_api_route("/health", _health, methods=["GET"])

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
