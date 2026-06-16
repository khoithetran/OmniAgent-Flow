"""LangFuse observability integration for the agentic AI pipeline.

LangFuse (https://langfuse.com) is an open-source LLM observability
platform. It gives us three things we need for this project:

1. **Tracing** of every LLM call inside LangGraph nodes, so we can see
   the full graph execution (intent classification -> response branch)
   with inputs, outputs, latency, and token usage.
2. **Cost & latency dashboards** broken down by node, model, and
   conversation.
3. **Evaluation hooks**: we attach a small set of scorers (faithfulness
   and answer relevance) that can run offline on captured traces.

This module is intentionally a thin, opt-in wrapper:

- When ``LANGFUSE_ENABLED`` is false (the default) every call becomes a
  no-op via the ``NullSpan`` shim, so unit tests and local dev with no
  LangFuse account keep working.
- We never import the SDK at module top level. The import happens
  lazily inside ``_get_client`` so that simply importing this module
  does not require the dependency to be installed when the feature is
  disabled.

Integration with the rest of the system
---------------------------------------
- ``ai_service.generate_agent_result`` wraps the entire LangGraph
  invocation in a single LangFuse trace.
- ``intent_service.extract_customer_intent`` wraps the OpenAI
  structured-output call in a generation span so token counts and
  latency are captured per call.
- ``evaluation_service`` (see ``evaluation.py``) consumes the same
  traces to compute faithfuness / answer-relevance scores.
"""

from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator
import os

from loguru import logger

from src.config import get_settings


class _NullSpan:
    """Stand-in span used when LangFuse is disabled.

    All methods are no-ops but accept the same kwargs as the real
    LangFuse SDK so call-sites do not have to branch.
    """

    def update(self, **_kwargs: Any) -> None:
        return None

    def end(self) -> None:
        return None

    def score(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def generation(self, **_kwargs: Any) -> "_NullSpan":
        return self


class _NullTrace(_NullSpan):
    """A null parent trace. Behaves exactly like a no-op span."""


class _NullLangFuseClient:
    """Drop-in replacement for ``langfuse.Langfuse`` when disabled.

    Returns ``_NullTrace`` for ``trace`` and an empty list for
    ``get_current_trace_model``/``flush`` so any caller keeps working.
    """

    def trace(self, *_args: Any, **_kwargs: Any) -> _NullTrace:
        return _NullTrace()

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


_client: Any = None
_initialized: bool = False


def _get_client() -> Any:
    """Build or return the LangFuse client, or a null stub when disabled.

    The client is cached for the lifetime of the process. We set the
    LangFuse env vars from our settings when missing so the SDK picks
    them up; this also lets the OpenAI wrapper auto-instrument calls
    (when ``langfuse.openai`` instrumentation is configured elsewhere).
    """

    global _client, _initialized
    if _initialized:
        return _client

    _initialized = True
    settings = get_settings()

    if not settings.langfuse_enabled:
        logger.info("LangFuse is disabled; using null observability client")
        _client = _NullLangFuseClient()
        return _client

    public_key = settings.langfuse_public_key_value
    secret_key = settings.langfuse_secret_key_value

    if not public_key or not secret_key:
        logger.warning(
            "LangFuse is enabled but credentials are missing; falling back to null client"
        )
        _client = _NullLangFuseClient()
        return _client

    # Surface credentials via env so any deeper SDK auto-instrumentation
    # (e.g. the OpenAI wrapper) also picks them up automatically.
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)

    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]

        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=settings.langfuse_host,
        )
        logger.info(
            "Initialized LangFuse client", host=settings.langfuse_host
        )
    except Exception:
        logger.exception("Failed to initialize LangFuse client; using null stub")
        _client = _NullLangFuseClient()

    return _client


def is_observability_enabled() -> bool:
    """Quick check used by tests and other services."""

    return get_settings().langfuse_enabled


@contextmanager
def trace_agent_run(
    *,
    sender_id: str,
    user_message: str,
) -> Iterator[Any]:
    """Open a LangFuse trace for one full agent run.

    Use as a ``with`` block in synchronous code. For async code, prefer
    ``atrace_agent_run`` which is awaitable-safe.

    Parameters
    ----------
    sender_id:
        External user id. Used as the LangFuse ``user_id`` so traces
        can be filtered per conversation.
    user_message:
        The current user input. Stored as trace input for replay.
    """

    client = _get_client()
    try:
        trace = client.trace(
            name="customer_support_agent.run",
            user_id=sender_id,
            input={"user_message": user_message},
            metadata={"sender_id": sender_id},
        )
    except Exception:
        logger.exception("Failed to open LangFuse trace; using null span")
        trace = _NullTrace()

    try:
        yield trace
    finally:
        try:
            trace.end()
        except Exception:
            logger.exception("Failed to close LangFuse trace")
        try:
            client.flush()
        except Exception:
            logger.exception("Failed to flush LangFuse client")


@asynccontextmanager
async def atrace_agent_run(
    *,
    sender_id: str,
    user_message: str,
) -> AsyncIterator[Any]:
    """Async variant of :func:`trace_agent_run`."""

    client = _get_client()
    try:
        trace = client.trace(
            name="customer_support_agent.run",
            user_id=sender_id,
            input={"user_message": user_message},
            metadata={"sender_id": sender_id},
        )
    except Exception:
        logger.exception("Failed to open LangFuse trace; using null span")
        trace = _NullTrace()

    try:
        yield trace
    finally:
        try:
            trace.end()
        except Exception:
            logger.exception("Failed to close LangFuse trace")
        try:
            client.flush()
        except Exception:
            logger.exception("Failed to flush LangFuse client")


def record_intent_generation(
    *,
    trace: Any,
    model: str,
    user_message: str,
    output: dict[str, Any],
    usage: dict[str, int] | None = None,
    latency_ms: float | None = None,
) -> None:
    """Attach a generation span for the structured-output call.

    Captures model name, prompt/output, and token usage so the LangFuse
    dashboard shows cost and latency per intent classification.
    """

    try:
        generation = trace.generation(
            name="intent_classification",
            model=model,
            input=user_message,
            output=output,
            usage=usage or {},
            metadata={"latency_ms": latency_ms} if latency_ms is not None else {},
        )
        generation.end()
    except Exception:
        logger.exception("Failed to record LangFuse generation span")


def record_evaluation_score(
    *,
    trace: Any,
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """Attach an evaluation score (faithfulness, answer_relevance, ...)."""

    try:
        trace.score(name=name, value=value, comment=comment)
    except Exception:
        logger.exception("Failed to record LangFuse evaluation score", name=name)
