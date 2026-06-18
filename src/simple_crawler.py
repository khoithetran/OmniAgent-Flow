"""Lightweight CPU-based website crawler (no GPU/Chromium required).

Uses ``httpx`` + ``BeautifulSoup`` to fetch and extract clean text from
any public website. Falls back to ``crawl4ai`` when available (local dev
with Chromium).

Compared to crawl4ai:
- No JavaScript rendering (fine for content-heavy sites)
- No GPU required — runs on CPU
- Faster for simple pages

The output shape matches ``src.crawler.CrawlResult`` so the caller
(``app_gradio.handle_fetch``) works without any changes.
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger

# ---------------------------------------------------------------------------
# Public dataclass (compatible with src.crawler.CrawlResult)
# ---------------------------------------------------------------------------


@dataclass
class CrawlResult:
    """Outcome of crawling a single URL."""

    url: str
    title: str
    markdown: str
    success: bool
    error_message: str | None = None

    def to_dict(self):
        return {
            "url": self.url,
            "title": self.title,
            "markdown": self.markdown,
            "success": self.success,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

FETCH_TIMEOUT_SECONDS = 15.0

# Common sitemap locations
SITEMAP_PATHS: tuple[str, ...] = (
    "/sitemap.xml",
    "/sitemap_index.xml",
)

# Realistic browser User-Agent to avoid being blocked by anti-bot protections
DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# URLs that are known to require JavaScript rendering or are not supported
UNSUPPORTED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("github.com", "GitHub repository pages require JavaScript rendering. Try crawling a specific file or README via the raw URL."),
    ("youtube.com", "YouTube requires JavaScript rendering. This crawler fetches static HTML only."),
    ("youtu.be", "YouTube requires JavaScript rendering. This crawler fetches static HTML only."),
    ("twitter.com", "Twitter/X requires JavaScript rendering. This crawler fetches static HTML only."),
    ("x.com", "Twitter/X requires JavaScript rendering. This crawler fetches static HTML only."),
    ("facebook.com", "Facebook requires JavaScript rendering. This crawler fetches static HTML only."),
    ("instagram.com", "Instagram requires JavaScript rendering. This crawler fetches static HTML only."),
    ("tiktok.com", "TikTok requires JavaScript rendering. This crawler fetches static HTML only."),
)


def _check_unsupported_url(url: str) -> str | None:
    """Return an error message if the URL is known to require JS, else None."""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().removeprefix("www.")
    for pattern, message in UNSUPPORTED_PATTERNS:
        if netloc == pattern or netloc.endswith(f".{pattern}"):
            return message
    return None


def _is_same_domain(url: str, base_netloc: str) -> bool:
    parsed = urlparse(url)
    if not parsed.netloc:
        return False
    return parsed.netloc.lower() == base_netloc.lower()


def _normalize_url(url: str) -> str:
    """Strip trailing slash and query params for comparison."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract page title from <title> or og:title."""
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    title = soup.find("title")
    if title and title.string:
        return title.string.strip()
    return ""


def _extract_main_content(soup: BeautifulSoup) -> str:
    """Extract the main readable content from the page.

    Strategy: remove noise (nav, footer, script, style, ads) then
    extract text from the largest text block. <article> and <main>
    are preferred; falls back to body.
    """
    # Remove noise elements
    for tag in soup.find_all(
        ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
    ):
        tag.decompose()

    # Try semantic containers first
    for tag in ["article", "main", "div"]:
        container = soup.find(tag)
        if container:
            return container.get_text(separator="\n\n", strip=True)

    # Fall back to body
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n\n", strip=True)

    return soup.get_text(separator="\n\n", strip=True)


def _to_markdown(text: str) -> str:
    """Lightweight text-to-markdown (preserves paragraphs)."""
    # Collapse excessive blank lines
    lines = text.splitlines()
    cleaned = []
    prev_blank = False
    for line in lines:
        blank = not line.strip()
        if blank:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).strip()


def _crawl_single_url(
    url: str,
    timeout: float = FETCH_TIMEOUT_SECONDS,
) -> CrawlResult:
    """Fetch one URL and extract readable markdown."""
    # Check for known unsupported sites first
    if unsupported := _check_unsupported_url(url):
        return CrawlResult(
            url=url, title="", markdown="", success=False, error_message=unsupported
        )

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=DEFAULT_HEADERS) as client:
            response = client.get(url)
    except httpx.HTTPError as e:
        return CrawlResult(
            url=url, title="", markdown="", success=False, error_message=str(e)
        )

    if response.status_code != 200:
        return CrawlResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error_message=f"HTTP {response.status_code}",
        )

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return CrawlResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error_message=f"Non-HTML content-type: {content_type}",
        )

    soup = BeautifulSoup(response.text, "html.parser")
    title = _extract_title(soup)
    text = _extract_main_content(soup)
    markdown = _to_markdown(text)

    if not markdown.strip():
        return CrawlResult(
            url=url,
            title=title or url,
            markdown="",
            success=False,
            error_message="Empty page content",
        )

    return CrawlResult(
        url=url,
        title=title or url,
        markdown=markdown,
        success=True,
    )


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------


def _parse_sitemap(xml_text: str) -> list[str]:
    """Extract <loc> entries from a sitemap XML."""
    if not xml_text or "<" not in xml_text:
        return []

    urls: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag == "sitemapindex":
        for child in root:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_tag != "sitemap":
                continue
            for sub in child:
                sub_tag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                if sub_tag == "loc" and sub.text:
                    urls.append(sub.text.strip())
    else:
        for elem in root.iter():
            elem_tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if elem_tag == "loc" and elem.text:
                urls.append(elem.text.strip())

    return urls


def _fetch_sitemap(client: httpx.Client, sitemap_url: str) -> list[str]:
    """Fetch one sitemap and return listed URLs."""
    try:
        response = client.get(sitemap_url, timeout=FETCH_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return []
    if response.status_code != 200:
        return []
    return _parse_sitemap(response.text)


async def discover_urls(base_url: str) -> list[str]:
    """Discover all URLs via sitemap (synchronous, used via asyncio.to_thread)."""
    # Check for known unsupported sites first
    if unsupported := _check_unsupported_url(base_url):
        return []

    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return [base_url]

    origin = f"{parsed.scheme}://{parsed.netloc}"
    found: list[str] = []

    with httpx.Client(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True, headers=DEFAULT_HEADERS) as client:
        for path in SITEMAP_PATHS:
            sitemap_url = urljoin(origin, path)
            urls = _fetch_sitemap(client, sitemap_url)
            if urls:
                found.extend(urls)
                break  # One working sitemap is enough

    if not found:
        return [base_url]

    # Dedupe
    seen: set[str] = set()
    deduped: list[str] = []
    for u in found:
        norm = _normalize_url(u)
        if norm not in seen and _is_same_domain(u, parsed.netloc):
            seen.add(norm)
            deduped.append(u)

    return deduped


# ---------------------------------------------------------------------------
# Main async entry point (matches crawl_full_website interface)
# ---------------------------------------------------------------------------


async def crawl_full_website(
    base_url: str,
    *,
    max_pages: int = 20,
    semaphore_count: int = 4,
) -> list[CrawlResult]:
    """Crawl a website using lightweight HTML fetcher (no GPU needed).

    Falls back to ``crawl4ai`` when available for JavaScript-heavy sites.

    Parameters
    ----------
    base_url:
        Any URL on the target site.
    max_pages:
        Maximum number of pages to crawl (capped at sitemap size or this).
    semaphore_count:
        Concurrency limit for async fetches.
    """
    # Try simple crawler first (CPU, no GPU)
    try:
        urls = await discover_urls(base_url)
    except Exception:  # noqa: BLE001
        urls = [base_url]

    if not urls:
        urls = [base_url]

    # Dedupe to max_pages
    if len(urls) > max_pages:
        # Always include the base URL first
        base_norm = _normalize_url(base_url)
        ordered = [u for u in urls if _normalize_url(u) != base_norm]
        urls = [base_url] + ordered[: max_pages - 1]

    semaphore = asyncio.Semaphore(semaphore_count)

    async def fetch_one(url: str) -> CrawlResult:
        async with semaphore:
            # Run sync httpx in thread pool to avoid blocking
            return await asyncio.to_thread(_crawl_single_url, url)

    results: list[CrawlResult] = await asyncio.gather(*[fetch_one(u) for u in urls])
    return results


# ---------------------------------------------------------------------------
# Markdown chunking (shared, used by rag.py)
# ---------------------------------------------------------------------------


def chunk_markdown(
    markdown: str,
    *,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[str]:
    """Split a markdown document into overlapping character chunks.

    Simple paragraph-based splitter with configurable overlap so retrieval
    does not lose context at chunk boundaries.
    """
    if chunk_size < 50:
        raise ValueError("chunk_size must be >= 50")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    text = markdown.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            sentences = [s.strip() for s in paragraph.split(". ") if s.strip()]
            for sentence in sentences:
                candidate = (current + "\n\n" + sentence).strip() if current else sentence
                if len(candidate) > chunk_size and current:
                    chunks.append(current)
                    tail = current[-overlap:] if overlap else ""
                    current = (tail + "\n\n" + sentence).strip()
                else:
                    current = candidate
            continue

        candidate = (current + "\n\n" + paragraph).strip() if current else paragraph
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + paragraph).strip()
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Auto-detection: use crawl4ai when available
# ---------------------------------------------------------------------------

_CRAWL4AI_AVAILABLE = False
try:
    from src.crawler import crawl_full_website as _crawl4ai_crawl

    _CRAWL4AI_AVAILABLE = True
except ImportError:
    _CRAWL4AI_AVAILABLE = False


if _CRAWL4AI_AVAILABLE:
    crawl_full_website = _crawl4ai_crawl  # type: ignore[assignment]
