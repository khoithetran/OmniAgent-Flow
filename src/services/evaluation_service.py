"""LLM evaluation scorers used by LangFuse traces.

We implement two lightweight scorers that are commonly cited in
LLMOps interviews:

- **Faithfulness** (a.k.a. groundedness): does the answer stick to the
  retrieved context, or is it hallucinating facts that are not in the
  knowledge base?
- **Answer relevance**: is the reply actually addressing the user's
  question, or is it a generic template that does not help?

We deliberately keep the implementation deterministic and dependency
free. Both scorers use the same fallback heuristic so the system can
run without an OpenAI key:

1. Faithfulness = 1.0 when the answer contains at least one phrase from
   the retrieved context, 0.5 when context is empty (nothing to ground
   against), and 0.0 when there is no overlap at all.
2. Answer relevance = 1.0 when the answer contains at least one token
   from the question, 0.0 otherwise. A soft middle ground is given
   when the answer mentions one of the known intent keywords
   ("tu van", "bao gia", ...) so the four canned responses still get
   non-zero scores.

When an OpenAI key is configured and the call succeeds, we ask the
model to score 0.0-1.0 with a short rubric. The fallback is used on
any failure so observability never blocks the worker.
"""

from typing import Any
import re

from loguru import logger
from pydantic import BaseModel

from src.config import get_settings


class EvaluationScores(BaseModel):
    """Bundle of all evaluation scores for one trace."""

    faithfulness: float
    answer_relevance: float
    method: str  # "openai" or "fallback"
    rationale: str | None = None


_INTENT_KEYWORDS: tuple[str, ...] = (
    "tu van",
    "bao gia",
    "chi phi",
    "gia",
    "lien he",
    "nhan vien",
    "sales",
    "ho tro",
    "khach hang",
    "automation",
    "crm",
    "messenger",
    "facebook",
    "zalo",
    "telegram",
    "huong dan",
    "giai phap",
)

_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)}


def _fallback_faithfulness(answer: str, context: str) -> float:
    if not context.strip():
        # No retrieved docs; we cannot ground the answer. Return a
        # neutral score instead of 0 so the dashboard still has signal.
        return 0.5

    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(context)
    overlap = answer_tokens & context_tokens
    if not answer_tokens:
        return 0.0
    return round(len(overlap) / len(answer_tokens), 4)


def _fallback_answer_relevance(question: str, answer: str) -> float:
    question_tokens = _tokenize(question)
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0

    direct_overlap = question_tokens & answer_tokens
    if direct_overlap:
        # Weighted blend: overlap with the question plus a small boost
        # for matching a known intent keyword.
        base = len(direct_overlap) / max(len(question_tokens), 1)
        keyword_bonus = 0.1 if any(k in answer.casefold() for k in _INTENT_KEYWORDS) else 0.0
        return round(min(1.0, base + keyword_bonus), 4)

    if any(k in answer.casefold() for k in _INTENT_KEYWORDS):
        return 0.4
    return 0.0


async def _openai_score(
    *,
    question: str,
    answer: str,
    context: str,
) -> EvaluationScores | None:
    settings = get_settings()
    api_key = settings.openai_api_key_value
    if not api_key:
        return None

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        prompt = (
            "You are an evaluation judge. Score the assistant answer on two "
            "axes from 0.0 to 1.0. Respond strictly as JSON with keys "
            "'faithfulness', 'answer_relevance', 'rationale'.\n\n"
            f"Question: {question}\n\n"
            f"Context (may be empty): {context}\n\n"
            f"Answer: {answer}\n"
        )
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content or "{}"
        import json

        parsed = json.loads(content)
        return EvaluationScores(
            faithfulness=float(parsed.get("faithfulness", 0.0)),
            answer_relevance=float(parsed.get("answer_relevance", 0.0)),
            method="openai",
            rationale=str(parsed.get("rationale", "")) or None,
        )
    except Exception:
        logger.exception("OpenAI evaluation scoring failed; using fallback")
        return None


async def evaluate_response(
    *,
    question: str,
    answer: str,
    context: str = "",
) -> EvaluationScores:
    """Compute faithfulness + answer_relevance for one Q/A turn."""

    openai_scores = await _openai_score(
        question=question, answer=answer, context=context
    )
    if openai_scores is not None:
        return openai_scores

    return EvaluationScores(
        faithfulness=_fallback_faithfulness(answer, context),
        answer_relevance=_fallback_answer_relevance(question, answer),
        method="fallback",
        rationale=None,
    )


def build_evaluation_metadata(scores: EvaluationScores) -> dict[str, Any]:
    """Flatten :class:`EvaluationScores` for storage in conversation metadata."""

    return scores.model_dump()
