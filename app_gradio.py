"""Gradio web interface for the OmniAgent Flow chatbot.

The flow is intentionally simple:

- Sidebar column (Scale 1): KB Status, Data Ingestion, Search & Agent Settings, RAGAS Eval Dashboard.
- Main column (Scale 3): Chat history, Model Selector, message input.

When no KB is loaded, the chat uses a general LLM (``SYSTEM_PROMPT_GENERAL``).
When a document/URL has been fetched and indexed, the chat switches to RAG mode
(``SYSTEM_PROMPT_RAG``) and refuses to invent answers outside the document.

Streaming is wired through Gradio's native ``type="messages"`` Chatbot +
``stream`` event so the user sees tokens appear in real time.

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
from src.doc_loader import load_document_from_bytes, supported_extensions  # noqa: E402
from src.chunker import ChunkStrategy, chunk_pages  # noqa: E402

# Use lightweight CPU crawler (no GPU/Chromium needed).
# Falls back to crawl4ai when available (local dev).
try:
    from src.simple_crawler import crawl_full_website  # noqa: E402
except ImportError:
    from src.crawler import crawl_full_website  # noqa: E402


# Chunking strategy choices for the Gradio dropdown.
# Each tuple: (display_label, ChunkStrategy value)
CHUNK_STRATEGIES: list[tuple[str, str]] = [
    ("Recursive (recommended)", ChunkStrategy.RECURSIVE),
    ("Fixed-size", ChunkStrategy.FIXED),
    ("Parent-Child", ChunkStrategy.PARENT_CHILD),
    ("Tokenizer-aware", ChunkStrategy.TOKENIZER),
]
CHUNK_STRATEGY_LABELS = [label for label, _ in CHUNK_STRATEGIES]
CHUNK_STRATEGY_VALUES = {label: val for label, val in CHUNK_STRATEGIES}


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
    "OpenAI - GPT 5.4",
    "Anthropic - Claude Sonnet 5",
]

DEFAULT_MODEL = "Anthropic - Claude Sonnet 5"

CONTEXT_WINDOWS: dict[str, int] = {
    "OpenAI - GPT 5.4": 128_000,
    "Anthropic - Claude Sonnet 5": 200_000,
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


async def handle_upload(
    state: dict[str, Any],
    files: list | None,
    strategy_label: str,
) -> tuple[dict[str, Any], str, list[dict[str, str]]]:
    """Index uploaded document files (PDF/Word/Excel/MD) into Qdrant.

    Parameters
    ----------
    state:
        Current Gradio state dict.
    files:
        List of file objects from gr.File (each has .name path on disk).
    strategy_label:
        Display label from the chunking strategy dropdown.

    Returns
    -------
    (new_state, status_text, chat_history)
    """
    if not files:
        return state, "⚠️ Vui lòng chọn ít nhất một file.", []

    from src.rag import index_doc_chunks  # noqa: E402

    strategy = CHUNK_STRATEGY_VALUES.get(strategy_label, ChunkStrategy.RECURSIVE)
    logger.info("Upload started", files=[f.name for f in files], strategy=strategy)

    all_chunks: list = []
    file_names: list[str] = []
    total_pages = 0

    for file_obj in files:
        file_path = Path(file_obj.name)
        filename = file_path.name
        file_names.append(filename)

        try:
            # Step 1: Load document → list[DocPage]
            pages = await asyncio.to_thread(
                load_document_from_bytes, file_path.read_bytes(), filename
            )
            total_pages += len(pages)

            # Step 2: Chunk using selected strategy → list[Chunk]
            chunks = await asyncio.to_thread(
                chunk_pages,
                pages,
                strategy,
                chunk_size=500,
                overlap=50,
                parent_size=1000,
                child_size=200,
                max_tokens=256,
                overlap_tokens=32,
            )
            all_chunks.extend(chunks)

            retrieval_count = sum(1 for c in chunks if not c.is_parent)
            logger.info(
                "File processed",
                filename=filename,
                pages=len(pages),
                total_chunks=len(chunks),
                retrieval_chunks=retrieval_count,
                strategy=strategy,
            )

        except ValueError as e:
            return state, f"❌ {filename}: {e}", []
        except ImportError as e:
            return state, f"❌ Thiếu thư viện: {e}. Chạy: pip install -r requirements.txt", []
        except Exception:  # noqa: BLE001
            logger.exception("File processing failed", filename=filename)
            return state, f"❌ Không xử lý được file: {filename}", []

    if not all_chunks:
        return state, "❌ Không trích xuất được nội dung từ các file đã chọn.", []

    # Step 3: Embed + index vào Qdrant (hoặc in-memory fallback)
    try:
        summary = await index_doc_chunks(all_chunks, replace=False)
    except Exception:  # noqa: BLE001
        logger.exception("Upload index failed")
        return state, "❌ Index thất bại. Kiểm tra kết nối Qdrant.", []

    indexed = summary["indexed"]
    source_label = ", ".join(file_names[:3])
    if len(file_names) > 3:
        source_label += f" (+{len(file_names) - 3} files)"

    new_state: dict[str, Any] = {
        **state,
        "kb_ready": True,
        "kb_domain": source_label,
        "kb_pages": total_pages,
        "kb_chunks": indexed,
        "kb_url": "",
    }

    status = (
        f"✅ {len(file_names)} file(s), {total_pages} trang, "
        f"{indexed} chunks indexed.\n"
        f"Strategy: **{strategy_label}**"
    )
    new_history = [{
        "role": "assistant",
        "content": (
            f"⚠️ **Đã index {len(file_names)} file(s)** ({strategy_label}). "
            "Tôi chỉ trả lời dựa trên nội dung tài liệu đã cung cấp."
        ),
    }]

    logger.info(
        "Upload index complete",
        files=len(file_names),
        chunks=indexed,
        strategy=strategy,
    )
    return new_state, status, new_history


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
    enable_hybrid: bool = False,
    enable_rerank: bool = False,
    enable_agent: bool = False,
) -> AsyncIterator[tuple[list[dict[str, str]], dict[str, Any], str]]:
    """Yield streaming chat updates for the Gradio Chatbot."""
    selected_model = state.get("selected_model") or DEFAULT_MODEL
    total_window = CONTEXT_WINDOWS.get(selected_model, 128_000)

    def _usage(used: int) -> str:
        return _format_usage(used, total_window)

    if not user_message or not user_message.strip():
        yield history, state, _usage(0)
        return

    user_message = user_message.strip()
    sender_id = "gradio-default"
    kb_ready = bool(state.get("kb_ready"))

    # 1. Append user message
    history = history + [{"role": "user", "content": user_message}]
    used_est = sum(_estimate_tokens(m.get("content", "")) for m in history)
    yield history, state, _usage(used_est)

    # 2. AI Agent Mode (ReAct Tool Loop)
    if enable_agent:
        from src.agent import run_agent_stream

        accumulated = ""
        try:
            async for partial in run_agent_stream(
                user_message,
                history[:-1],  # previous turns
                kb_state=state,
                enable_hybrid=enable_hybrid,
                enable_rerank=enable_rerank,
            ):
                accumulated = partial
                if history and history[-1].get("role") == "assistant":
                    history = history[:-1] + [{"role": "assistant", "content": accumulated}]
                else:
                    history = history + [{"role": "assistant", "content": accumulated}]
                used_est = sum(_estimate_tokens(m.get("content", "")) for m in history)
                yield history, state, _usage(used_est)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent execution failed")
            history = history + [{"role": "assistant", "content": f"Lỗi Agent: {exc}"}]
            yield history, state, _usage(used_est)
        return

    # 3. Standard RAG mode
    if kb_ready and not state.get("kb_domain"):
        kb_ready = False

    if kb_ready:
        try:
            from src.rag import search, format_context
            hits = await search(
                user_message,
                enable_hybrid=enable_hybrid,
                enable_rerank=enable_rerank,
            )
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

    # 4. Stream LLM reply
    accumulated = ""
    try:
        stream_iter = (
            chat_module.chat_rag_stream(
                sender_id,
                user_message,
                model=selected_model,
                enable_hybrid=enable_hybrid,
                enable_rerank=enable_rerank,
            )
            if kb_ready
            else chat_module.chat_general_stream(sender_id, user_message, model=selected_model)
        )
        async for partial in stream_iter:
            accumulated = partial
            if history and history[-1].get("role") == "assistant":
                history = history[:-1] + [{"role": "assistant", "content": accumulated}]
            else:
                history = history + [{"role": "assistant", "content": accumulated}]
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


async def handle_eval_run(
    query: str,
    enable_hybrid: bool,
    enable_rerank: bool,
) -> str:
    """Run RAGAS evaluation on query and return markdown summary."""
    if not query or not query.strip():
        return "⚠️ Vui lòng nhập câu hỏi thử nghiệm để đánh giá."
    from src.eval import evaluate_rag_pipeline, format_eval_summary

    result = await evaluate_rag_pipeline(
        query.strip(),
        enable_hybrid=enable_hybrid,
        enable_rerank=enable_rerank,
    )
    return format_eval_summary(result)


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
        is_openai = "OpenAI" in model_name
        label = f"🚫 {model_name} (Hết kinh phí)" if is_openai else f"⚡ {model_name}"
        btn = gr.Button(
            label,
            variant="secondary" if is_openai else "primary",
            size="sm",
            min_width=180,
            interactive=not is_openai,  # Disabled / grayed out for OpenAI
        )
        buttons.append(btn)
    return buttons


def _button_variants(selected: str) -> list[gr.update]:
    return [
        gr.update(
            variant="primary" if m == selected else "secondary",
            interactive="OpenAI" not in m,
        )
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
CUSTOM_CSS = """
.gradio-container {
    max-width: 70% !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
@media (max-width: 1024px) {
    .gradio-container {
        max-width: 95% !important;
    }
}
"""


def build_ui() -> gr.Blocks:
    """Construct the full Gradio app with 1:3 Sidebar Dashboard layout."""
    initial_state: dict[str, Any] = {
        "kb_ready": False,
        "kb_domain": "",
        "kb_pages": 0,
        "kb_chunks": 0,
        "kb_url": "",
        "selected_model": DEFAULT_MODEL,
    }

    with gr.Blocks(title="OmniAgent Flow", css=CUSTOM_CSS) as demo:
        state = gr.State(value=initial_state)

        with gr.Row(equal_height=False):
            # ---------------- CỘT TRÁI: SIDEBAR CONTROL PANEL (Scale = 1) ----------------
            with gr.Column(scale=1):
                gr.Markdown("### 📄 KB Status & Control")
                status = gr.Markdown("⚠️ **Trạng thái**: Chưa có tài liệu nào được nạp.")
                with gr.Row(visible=False) as kb_row:
                    domain_display = gr.Markdown("🔗 ...")
                    clear_x_btn = gr.Button("✕ Xóa KB", variant="stop", size="sm", scale=0)
                clear_kb_btn = gr.Button("Clear KB", variant="stop", visible=False)

                # Accordion 1: Data Ingestion
                with gr.Accordion("📥 Nạp Dữ Liệu (File / Web)", open=True):
                    with gr.Tabs():
                        with gr.Tab("📁 Upload File"):
                            file_upload = gr.File(
                                label="Kéo thả hoặc chọn tệp (PDF/Word/Excel/MD)",
                                file_count="multiple",
                                file_types=supported_extensions(),
                                show_label=False,
                                height=120,
                            )
                            chunk_strategy_dropdown = gr.Dropdown(
                                choices=CHUNK_STRATEGY_LABELS,
                                value=CHUNK_STRATEGY_LABELS[0],
                                label="Chunking Strategy",
                            )
                            upload_btn = gr.Button("🚀 Upload & Index File", variant="primary")

                        with gr.Tab("🔗 Fetch Web"):
                            url_input = gr.Textbox(
                                placeholder="https://...",
                                show_label=False,
                                lines=1,
                            )
                            fetch_btn = gr.Button("🚀 Fetch & Index Web", variant="primary")

                # Accordion 2: Search & Agent Settings
                with gr.Accordion("⚙️ Tùy Chỉnh Tìm Kiếm & Agent", open=False):
                    enable_hybrid_cb = gr.Checkbox(
                        label="Hybrid Search (BM25 + Dense RRF)",
                        value=True,
                        info="Kết hợp từ khóa chính xác và ngữ nghĩa",
                    )
                    enable_rerank_cb = gr.Checkbox(
                        label="Cross-Encoder Reranking",
                        value=True,
                        info="Tái xếp hạng bằng ms-marco model",
                    )
                    enable_agent_cb = gr.Checkbox(
                        label="🤖 AI Agent Mode (ReAct Loop)",
                        value=False,
                        info="LLM tự suy luận và gọi công cụ",
                    )

                # Accordion 3: RAGAS Evaluation Dashboard
                with gr.Accordion("📊 RAGAS Evaluation Dashboard", open=False):
                    eval_query_input = gr.Textbox(
                        placeholder="Nhập query thử nghiệm...",
                        label="Query Thử Nghiệm",
                        lines=1,
                    )
                    run_eval_btn = gr.Button("⚡ Chạy Evaluation", variant="secondary")
                    eval_report_output = gr.Markdown("Chưa chạy evaluation.")

            # ---------------- CỘT PHẢI: MAIN CHAT CANVAS (Scale = 3) ----------------
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=600,
                    show_label=False,
                    value=[
                        {
                            "role": "assistant",
                            "content": (
                                "Xin chào! Tôi là trợ lý ảo **OmniAgent Flow**.\n"
                                "Hãy nạp tài liệu ở **Sidebar Cột Trái** hoặc đặt câu hỏi trực tiếp tại đây!"
                            ),
                        }
                    ],
                )

                # Model selector row.
                with gr.Row():
                    model_buttons = _build_model_buttons()

                with gr.Row():
                    context_window_display = gr.Markdown(
                        value=_context_window_label(DEFAULT_MODEL),
                    )
                    token_usage_display = gr.Markdown(
                        value=_format_usage(0, CONTEXT_WINDOWS[DEFAULT_MODEL]),
                    )

                # Input row.
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Nhập câu hỏi của bạn...",
                        show_label=False,
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("🚀 Gửi", variant="primary", scale=1)

        # ---------------- Event wiring ----------------

        # 1. Model buttons: click to set active model.
        def _make_select_handler(m: str):
            def _handler(s: dict[str, Any]):
                new_state = select_model(s, m)[0]
                variants = _button_variants(m)
                label = _context_window_label(m)
                return (new_state, *variants, label)
            return _handler

        for btn, model_name in zip(model_buttons, MODELS):
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
                else "⚠️ **Trạng thái**: Chưa có tài liệu nào được nạp."
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
            fn=lambda: "⚠️ **Trạng thái**: Chưa có tài liệu nào được nạp.",
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
            fn=lambda: "⚠️ **Trạng thái**: Chưa có tài liệu nào được nạp.",
            inputs=[],
            outputs=[domain_display],
        )

        # 5b. Upload files.
        upload_btn.click(
            fn=handle_upload,
            inputs=[state, file_upload, chunk_strategy_dropdown],
            outputs=[state, status, chatbot],
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
                f"📁 **{s.get('kb_domain', '')}**"
                if s.get("kb_ready")
                else "⚠️ **Trạng thái**: Chưa có tài liệu nào được nạp."
            ),
            inputs=[state],
            outputs=[domain_display],
        )

        # 5c. RAGAS Evaluation button
        run_eval_btn.click(
            fn=handle_eval_run,
            inputs=[eval_query_input, enable_hybrid_cb, enable_rerank_cb],
            outputs=[eval_report_output],
        )

        # 5. Chat: Enter or Send button.
        chat_outputs = [chatbot, state, token_usage_display]
        chat_inputs = [
            msg,
            chatbot,
            state,
            enable_hybrid_cb,
            enable_rerank_cb,
            enable_agent_cb,
        ]
        send_event = msg.submit(
            fn=handle_chat,
            inputs=chat_inputs,
            outputs=chat_outputs,
        )
        send_btn.click(
            fn=handle_chat,
            inputs=chat_inputs,
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
