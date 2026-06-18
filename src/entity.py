"""Entity detection and URL handling for the chatbot.

The chatbot needs to know when a user is talking about a specific
company or organisation so it can ask for the website URL before
answering. This module owns:

1. ``extract_url`` - find a URL anywhere in the user message.
2. ``detect_company`` - ask OpenAI (cheap model, no streaming) whether
   the message mentions a company/org and what its name is.
3. ``EntityResult`` - tiny dataclass bundling the two flags.

The detection is intentionally simple:

- We do URL extraction with a regex first because it is cheap and
  deterministic - a URL is a URL regardless of intent.
- We only call the LLM for company detection when there is no URL
  in the message. That keeps the hot path fast and avoids spending
  tokens on obvious questions.
- A 1-2 second timeout is enforced on the LLM call. The chat layer
  treats a timeout as "no company detected" so the conversation
  still proceeds.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from openai import AsyncOpenAI

from src import rag
from src.config import get_settings


#: Match http(s) URLs as well as bare domains like ``example.com`` and
#: ``www.example.com/path``. We deliberately keep this loose so the
#: chat layer can decide what to do with the candidate.
_URL_PATTERN = re.compile(
    r"""
    (?P<url>
        (?:https?://)?                  # optional scheme
        [^\s,;()<>"']+                  # host + path, no whitespace
        [^\s,;()<>"'.,!?]                # last char not punctuation
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

#: Match an RFC-5322-ish email address. Intentionally simple: the chat
#: layer uses this as a first pass and the OpenAI LLM confirms.
_EMAIL_PATTERN = re.compile(
    r"""
    (?P<email>
        [A-Za-z0-9._%+-]+                # local part
        @
        [A-Za-z0-9.-]+                   # domain
        \.
        [A-Za-z]{2,}                     # TLD
    )
    """,
    re.VERBOSE,
)

#: A small list of Vietnamese cue phrases (both with and without
#: diacritics) that often mean "this sentence is about a company".
#: We use a case-folded substring match so both ``công ty`` and
#: ``cong ty`` are caught. The set is intentionally broad: false
#: positives are cheap (one extra LLM call) while false negatives
#: drop a real question.
_COMPANY_CUE_WORDS: tuple[str, ...] = (
    # Explicit company markers
    "công ty",
    "cong ty",
    "cty",
    "doanh nghiệp",
    "doanh nghiep",
    "tổ chức",
    "to chuc",
    "shop",
    "cửa hàng",
    "cua hang",
    "brand",
    "thương hiệu",
    "thuong hieu",
    "agency",
    "startup",
    "company",
    "organization",
    "organisation",
    "corporation",
    "enterprise",
    "inc",
    "llc",
    "ltd",
    # Common Vietnamese question patterns about a company
    "gioi thieu",
    "giới thiệu",
    "ve cong ty",
    "về công ty",
    "cua cong ty",
    "của công ty",
    "cua hang",
    "của hãng",
    "cua hang",
    "của thương hiệu",
    # Service / product inquiry
    "dich vu",
    "dịch vụ",
    "san pham",
    "sản phẩm",
    "gia ca",
    "giá cả",
    "bao nhieu",
    "bao nhiêu",
    "gia",
    "giá",
)


@dataclass(slots=True)
class EntityResult:
    """Outcome of analysing one user message.

    Attributes
    ----------
    url:
        The first URL found in the message, or ``None``.
    company:
        The detected company / organisation name, or ``None`` when
        no entity is present.
    email:
        The first email address found in the message, or ``None``.
    """

    url: str | None = None
    company: str | None = None
    email: str | None = None

    @property
    def has_url(self) -> bool:
        return self.url is not None

    @property
    def has_company(self) -> bool:
        return self.company is not None

    @property
    def has_email(self) -> bool:
        return self.email is not None


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------


def _looks_like_url(candidate: str) -> bool:
    """Heuristic check that ``candidate`` is a real URL, not a phone
    number, a price, or a stray string of text.

    We require a dot in the host part and a valid TLD-like suffix
    (>= 2 alpha chars). This is intentionally loose - the chat
    layer will hand the URL to ``crawl_full_website`` which does the
    real validation.
    """
    text = candidate.strip()
    if not text:
        return False
    if "://" not in text:
        text = f"http://{text}"
    try:
        parsed = urlparse(text)
    except ValueError:
        return False
    host = parsed.netloc
    if not host or "." not in host:
        return False
    tld = host.rsplit(".", 1)[-1]
    if len(tld) < 2 or not tld.isalpha():
        return False
    return True


def extract_url(message: str) -> str | None:
    """Return the first URL-looking substring in ``message``.

    Skips email addresses (``foo@bar.com``) so a support contact does
    not get treated as a website URL.
    """
    if not message:
        return None

    for match in _URL_PATTERN.finditer(message):
        candidate = match.group("url").rstrip(".,;:!?")
        # Skip emails: ``user@host.com`` is not a crawlable URL.
        if "@" in candidate and "://" not in candidate:
            continue
        if _looks_like_url(candidate):
            return candidate
    return None


def extract_email(message: str) -> str | None:
    """Return the first email-shaped substring in ``message``.

    Returns the address lowercased so the same person is deduped
    regardless of how they typed their address. Defers to
    OpenAI for the final validation when the LLM layer decides
    to capture the contact.
    """
    if not message:
        return None

    match = _EMAIL_PATTERN.search(message)
    if not match:
        return None
    return match.group("email").lower()


# ---------------------------------------------------------------------------
# Company / org detection via OpenAI
# ---------------------------------------------------------------------------


_COMPANY_DETECTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "report_company",
        "description": (
            "Report the company or organisation the user is asking about, "
            "if any. Return an empty string when the question is generic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": (
                        "The name of the company, organisation, or brand "
                        "the user is asking about. Empty string when none."
                    ),
                },
            },
            "required": ["company_name"],
        },
    },
}


def _looks_like_company_question(message: str) -> bool:
    """Cheap pre-filter: does the message mention a company cue word?

    Normalises Vietnamese diacritics so both ``công ty`` and
    ``cong ty`` match the cue list. Falls back to the raw lowercased
    message so English-only cues like ``company`` still match.
    """
    raw = message.casefold()
    decomposed = "".join(
        char
        for char in unicodedata.normalize("NFKD", raw)
        if not unicodedata.combining(char)
    )
    haystacks = (raw, decomposed)
    return any(
        any(cue in hay for cue in _COMPANY_CUE_WORDS)
        for hay in haystacks
    )


async def detect_company(message: str, *, timeout: float = 5.0) -> str | None:
    """Ask OpenAI whether the message is about a specific company.

    Returns the detected company name, or ``None`` when no entity is
    present, when the OpenAI key is missing, or when the call
    times out. A timeout never raises - the chat layer falls back to
    the generic RAG path.
    """
    if not _looks_like_company_question(message):
        # Skip the LLM entirely for generic questions like "What is
        # Python?" - the keyword pre-filter is the cheap, correct call
        # for those.
        return None

    if rag.openai_client is None:
        return None

    settings = get_settings()
    client: AsyncOpenAI = rag._get_openai()  # type: ignore[assignment]

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract the company or organisation the user "
                        "is asking about. Call the report_company function "
                        "with the company name. If no specific company is "
                        "mentioned, pass an empty string."
                    ),
                },
                {"role": "user", "content": message},
            ],
            tools=[_COMPANY_DETECTION_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "report_company"},
            },
            temperature=0.0,
            max_tokens=60,
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Company detection LLM call failed")
        return None

    try:
        tool_call = response.choices[0].message.tool_calls[0]
    except (AttributeError, IndexError, TypeError):
        return None

    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return None

    name = (args.get("company_name") or "").strip()
    return name or None


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


async def analyse(message: str) -> EntityResult:
    """Return the URL, company name, and email in one call.

    The URL and email checks are cheap and always run. The LLM call
    for company detection only runs when the message looks like a
    company question.
    """
    return EntityResult(
        url=extract_url(message),
        company=await detect_company(message),
        email=extract_email(message),
    )
