from enum import Enum
from typing import Any
import unicodedata

from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_settings


class CustomerIntent(str, Enum):
    CONSULTATION = "consultation"
    PRICING = "pricing"
    HANDOFF = "handoff"
    FALLBACK = "fallback"


class CustomerIntentExtraction(BaseModel):
    intent: CustomerIntent = Field(
        description=(
            "Primary customer intent: consultation, pricing, handoff, or fallback."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the intent classification.",
    )
    customer_name: str | None = Field(
        description="Customer name if explicitly provided, otherwise null."
    )
    company: str | None = Field(
        description="Company or organization name if explicitly provided, otherwise null."
    )
    phone: str | None = Field(
        description="Phone number if explicitly provided, otherwise null."
    )
    email: str | None = Field(
        description="Email address if explicitly provided, otherwise null."
    )
    budget: str | None = Field(
        description="Budget or price expectation if explicitly provided, otherwise null."
    )
    timeline: str | None = Field(
        description="Expected buying or implementation timeline if provided."
    )
    channels: list[str] = Field(
        description="Mentioned support channels such as Facebook, Zalo, website, or CRM."
    )
    pain_points: list[str] = Field(
        description="Customer pain points or operational problems mentioned."
    )
    product_interest: str | None = Field(
        description="Product, service, or solution area the customer is asking about."
    )
    urgency: str = Field(
        description="One of low, medium, or high based on the customer's wording."
    )
    language: str = Field(
        description="Detected language: vi, en, mixed, or unknown."
    )
    summary: str = Field(
        description="Short business summary of the customer message."
    )


CONSULTATION_KEYWORDS: tuple[str, ...] = (
    "tu van",
    "giai phap",
    "demo",
    "trien khai",
    "nhu cau",
    "workflow",
    "automation",
)
PRICING_KEYWORDS: tuple[str, ...] = (
    "gia",
    "bao gia",
    "chi phi",
    "phi",
    "pricing",
    "price",
    "quote",
)
HANDOFF_KEYWORDS: tuple[str, ...] = (
    "gap nguoi",
    "nhan vien",
    "tu van vien",
    "sales",
    "hotline",
    "lien he",
    "call",
)
CHANNEL_KEYWORDS: tuple[str, ...] = (
    "facebook",
    "messenger",
    "zalo",
    "website",
    "web",
    "telegram",
    "crm",
    "hubspot",
)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_marks = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return without_marks.casefold()


def _contains_keyword(message: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in message for keyword in keywords)


def _detect_language(message: str) -> str:
    normalized_message = _normalize_text(message)
    vietnamese_terms = ("toi", "ban", "tu van", "bao gia", "nhu cau", "khach hang")
    english_terms = ("price", "quote", "support", "customer", "workflow")
    has_vietnamese = _contains_keyword(normalized_message, vietnamese_terms)
    has_english = _contains_keyword(normalized_message, english_terms)

    if has_vietnamese and has_english:
        return "mixed"
    if has_vietnamese:
        return "vi"
    if has_english:
        return "en"
    return "unknown"


def _extract_channels(normalized_message: str) -> list[str]:
    return [
        channel
        for channel in CHANNEL_KEYWORDS
        if channel in normalized_message
    ]


def _build_fallback_extraction(
    user_message: str,
    session_history: list[dict[str, Any]],
) -> CustomerIntentExtraction:
    normalized_message = _normalize_text(user_message)

    if _contains_keyword(normalized_message, HANDOFF_KEYWORDS):
        intent = CustomerIntent.HANDOFF
        confidence = 0.7
    elif _contains_keyword(normalized_message, PRICING_KEYWORDS):
        intent = CustomerIntent.PRICING
        confidence = 0.72
    elif _contains_keyword(normalized_message, CONSULTATION_KEYWORDS):
        intent = CustomerIntent.CONSULTATION
        confidence = 0.68
    else:
        intent = CustomerIntent.FALLBACK
        confidence = 0.45

    return CustomerIntentExtraction(
        intent=intent,
        confidence=confidence,
        customer_name=None,
        company=None,
        phone=None,
        email=None,
        budget=user_message if intent == CustomerIntent.PRICING else None,
        timeline=None,
        channels=_extract_channels(normalized_message),
        pain_points=[],
        product_interest="OmniAgent Flow" if intent != CustomerIntent.FALLBACK else None,
        urgency="medium" if intent == CustomerIntent.HANDOFF else "low",
        language=_detect_language(user_message),
        summary=user_message[:240],
    )


def _format_recent_history(session_history: list[dict[str, Any]]) -> str:
    recent_messages = session_history[-6:]
    formatted_messages: list[str] = []

    for message in recent_messages:
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", ""))
        if content:
            formatted_messages.append(f"{role}: {content}")

    return "\n".join(formatted_messages) or "No previous session history."


def _build_extraction_input(
    user_message: str,
    session_history: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You extract structured CRM-ready customer intent metadata for an "
                "enterprise customer support automation platform. Only use facts "
                "explicitly present in the current message or recent history. Use null "
                "for unknown scalar fields and empty arrays for unknown list fields."
            ),
        },
        {
            "role": "user",
            "content": (
                "Recent session history:\n"
                f"{_format_recent_history(session_history)}\n\n"
                f"Current customer message:\n{user_message}"
            ),
        },
    ]


async def extract_customer_intent(
    sender_id: str,
    user_message: str,
    session_history: list[dict[str, Any]],
    use_structured_output: bool = True,
) -> CustomerIntentExtraction:
    settings = get_settings()
    api_key = settings.openai_api_key_value

    if not use_structured_output or api_key is None:
        logger.info(
            "Using fallback customer intent extractor",
            sender_id=sender_id,
            structured_output_enabled=use_structured_output,
            has_openai_api_key=api_key is not None,
        )
        return _build_fallback_extraction(user_message, session_history)

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        response = await client.responses.parse(
            model=settings.openai_model,
            input=_build_extraction_input(user_message, session_history),
            text_format=CustomerIntentExtraction,
        )
        parsed_output = response.output_parsed

        if parsed_output is None:
            raise ValueError("OpenAI structured output parser returned no data")

        logger.info(
            "Extracted customer intent with OpenAI Structured Outputs",
            sender_id=sender_id,
            model=settings.openai_model,
            intent=parsed_output.intent.value,
            confidence=parsed_output.confidence,
        )
        return parsed_output
    except Exception:
        logger.exception(
            "Failed to extract customer intent with OpenAI Structured Outputs",
            sender_id=sender_id,
            model=settings.openai_model,
        )
        return _build_fallback_extraction(user_message, session_history)
