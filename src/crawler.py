"""Website crawler built on top of crawl4ai.

The crawler turns any public URL into clean markdown that can be
chunked, embedded, and indexed in Qdrant. The output is intentionally
small: just enough metadata to rebuild a citation later plus the
content the LLM will actually read.

Design notes
------------
- We use ``fit_markdown`` instead of ``raw_markdown`` because it is
  already filtered for boilerplate (nav, footer, cookie banners) and
  gives a tighter signal for retrieval. The trade-off is occasional
  over-pruning on sites that put the content inside navigation.
- crawl4ai is heavy (one Chromium instance per browser) so we open a
  single ``AsyncWebCrawler`` for the whole batch and close it cleanly
  in a ``finally`` block.
- Failures are isolated per URL: a 404 on one page does not abort the
  rest of the batch. Each result carries its own ``success`` flag and
  ``error_message``.
- We retry twice on transient errors (network, 5xx) before giving up.
  The retry policy is intentionally simple - exponential backoff is
  delegated to the caller (RAG layer) so this module stays focused.
"""

from __future__ import annotations

import asyncio
import uuid
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from loguru import logger


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


class CrawlResult:
    """Outcome of crawling a single URL.

    Attributes
    ----------
    url:
        The original URL that was crawled.
    title:
        Page title extracted by crawl4ai (or the URL itself if absent).
    markdown:
        ``fit_markdown`` content. Empty on failure.
    success:
        Whether the crawl produced usable content.
    error_message:
        Human-readable error description on failure; ``None`` on success.
    """

    __slots__ = ("url", "title", "markdown", "success", "error_message")

    def __init__(
        self,
        url: str,
        title: str,
        markdown: str,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        self.url = url
        self.title = title
        self.markdown = markdown
        self.success = success
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "markdown": self.markdown,
            "success": self.success,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Headless Chromium with text-only mode and no images, so crawls are
#: fast and cheap. ``text_mode`` strips <img>/<video>; ``light_mode``
#: disables background features that bloat memory.
DEFAULT_BROWSER_CONFIG = BrowserConfig(
    headless=True,
    browser_type="chromium",
    text_mode=True,
    light_mode=True,
)

#: One config for every URL. We keep this small so the public API
#: stays simple; if a caller needs per-URL config, they can extend
#: later without breaking the existing signature.
DEFAULT_RUN_CONFIG = CrawlerRunConfig(
    cache_mode="bypass",
    check_robots_txt=False,
    semaphore_count=4,
    word_count_threshold=10,
)

#: How many times to retry a single URL before giving up. crawl4ai
#: already retries internally for some failure modes; this is a
#: second safety net on top.
DEFAULT_MAX_RETRIES = 2

#: How long to wait between retries on a single URL.
RETRY_BACKOFF_SECONDS = 1.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_result_from_crawl4ai(raw: Any, url: str) -> CrawlResult:
    """Translate a crawl4ai ``CrawlResult`` into our ``CrawlResult``.

    crawl4ai's result object exposes ``success`` (bool), ``markdown``
    (``MarkdownContent`` with ``raw_markdown`` and ``fit_markdown``),
    ``metadata`` (a dict that usually contains the page title), and
    ``error_message``.
    """
    if not getattr(raw, "success", False):
        return CrawlResult(
            url=url,
            title="",
            markdown="",
            success=False,
            error_message=getattr(raw, "error_message", "Unknown crawl4ai error"),
        )

    markdown_obj = getattr(raw, "markdown", None)
    fit_markdown = getattr(markdown_obj, "fit_markdown", "") if markdown_obj else ""
    raw_markdown = getattr(markdown_obj, "raw_markdown", "") if markdown_obj else ""

    # Prefer fit_markdown when it has substance, otherwise fall back to
    # raw_markdown so we never return empty content for a successful crawl.
    markdown = fit_markdown.strip() or raw_markdown.strip()

    metadata = getattr(raw, "metadata", None) or {}
    title = ""
    if isinstance(metadata, dict):
        title = str(metadata.get("title") or metadata.get("og:title") or "").strip()

    return CrawlResult(
        url=url,
        title=title or url,
        markdown=markdown,
        success=True,
    )


def _make_failure(url: str, message: str) -> CrawlResult:
    logger.warning("Crawl failed", url=url, error=message)
    return CrawlResult(
        url=url,
        title="",
        markdown="",
        success=False,
        error_message=message,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def crawl_website(
    url: str,
    *,
    browser_config: BrowserConfig | None = None,
    run_config: CrawlerRunConfig | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> CrawlResult:
    """Crawl a single URL and return its markdown.

    Parameters
    ----------
    url:
        Absolute URL to crawl. The crawler does not resolve redirects
        across hosts, so prefer the canonical URL.
    browser_config, run_config:
        Optional overrides. When ``None`` we use the module-level
        defaults that prioritise speed over fidelity.
    max_retries:
        Number of additional attempts after the first failure. Set to
        ``0`` to disable retries.
    """
    bc = browser_config or DEFAULT_BROWSER_CONFIG
    rc = run_config or DEFAULT_RUN_CONFIG
    attempts = max_retries + 1

    last_error = "Unknown error"
    for attempt in range(1, attempts + 1):
        try:
            async with AsyncWebCrawler(config=bc) as crawler:
                raw = await crawler.arun(url=url, config=rc)
            result = _build_result_from_crawl4ai(raw, url)
            if result.success and result.markdown:
                logger.info(
                    "Crawl succeeded",
                    url=url,
                    attempt=attempt,
                    chars=len(result.markdown),
                )
                return result
            last_error = result.error_message or "Empty markdown"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "Crawl attempt raised",
                url=url,
                attempt=attempt,
            )

        if attempt < attempts:
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return _make_failure(url, last_error)


async def crawl_websites(
    urls: list[str],
    *,
    browser_config: BrowserConfig | None = None,
    run_config: CrawlerRunConfig | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[CrawlResult]:
    """Crawl multiple URLs in one batch using a shared browser.

    One ``AsyncWebCrawler`` is opened for the whole batch. crawl4ai
    handles per-URL concurrency internally via ``semaphore_count``.
    """
    if not urls:
        return []

    bc = browser_config or DEFAULT_BROWSER_CONFIG
    rc = run_config or DEFAULT_RUN_CONFIG

    async with AsyncWebCrawler(config=bc) as crawler:
        results: list[CrawlResult] = []
        for url in urls:
            attempt_results: list[CrawlResult] = []
            for attempt in range(1, max_retries + 2):
                try:
                    raw = await crawler.arun(url=url, config=rc)
                    result = _build_result_from_crawl4ai(raw, url)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Crawl raised", url=url, attempt=attempt)
                    result = _make_failure(url, f"{type(exc).__name__}: {exc}")

                attempt_results.append(result)
                if result.success and result.markdown:
                    break
                if attempt <= max_retries:
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

            # Pick the last successful attempt; if all failed, keep the
            # last failure so callers can see the most recent error.
            final = next(
                (r for r in reversed(attempt_results) if r.success and r.markdown),
                attempt_results[-1],
            )
            results.append(final)
        return results


def chunk_markdown(
    markdown: str,
    *,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[str]:
    """Split a markdown document into overlapping character chunks.

    This is a deliberately simple, dependency-free splitter. It is
    good enough for the demo: each chunk is roughly one paragraph
    wide, with a configurable overlap so retrieval does not lose
    context at chunk boundaries.

    The function collapses runs of blank lines so we do not index
    empty chunks, then packs paragraphs greedily up to ``chunk_size``
    characters.
    """
    if chunk_size < 50:
        raise ValueError("chunk_size must be >= 50")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    text = markdown.strip()
    if not text:
        return []

    # Collapse blank lines but keep paragraph structure.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        # If the paragraph itself is larger than the chunk size, split
        # it on sentence boundaries.
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


def new_crawl_id() -> str:
    """Generate a short identifier for a crawl batch (used as Qdrant namespace)."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Sitemap discovery and full-website crawl
# ---------------------------------------------------------------------------

# Common sitemap locations, tried in order. Most modern sites use
# ``/sitemap.xml`` but some keep the older ``sitemap_index.xml`` name.
SITEMAP_PATHS: tuple[str, ...] = (
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
)

#: HTTP timeout when fetching a sitemap.
SITEMAP_FETCH_TIMEOUT_SECONDS = 10.0


def _normalize_url_for_compare(url: str) -> str:
    """Lowercase scheme + host and strip trailing slashes for dedupe.

    Path is kept as-is because ``/models`` and ``/models/`` are not
    always the same resource.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = parsed.netloc.lower()
    if not scheme or not host:
        return url
    return f"{scheme}://{host}{parsed.path}"


def _same_domain(url: str, base_netloc: str) -> bool:
    """Return True when ``url`` belongs to the same host as ``base_netloc``.

    Both are compared case-insensitively. The scheme is ignored so an
    ``https://`` sitemap entry still matches an ``http://`` base URL.
    """
    parsed = urlparse(url)
    if not parsed.netloc:
        return False
    return parsed.netloc.lower() == base_netloc.lower()


def _parse_sitemap_xml(xml_text: str) -> list[str]:
    """Extract ``<loc>`` entries from a sitemap XML body.

    Supports both the bare ``<urlset>`` form and the nested
    ``<sitemapindex><sitemap><loc>`` form by recursing into sitemap
    index documents. Non-URL elements are ignored.
    """
    if not xml_text or "<" not in xml_text:
        return []

    urls: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("Failed to parse sitemap XML")
        return urls

    # Strip namespace to make tag matching simple.
    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

    if tag == "sitemapindex":
        # A sitemap index references other sitemaps. Recurse into each.
        for child in root:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_tag != "sitemap":
                continue
            for sub in child:
                sub_tag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                if sub_tag == "loc" and sub.text:
                    urls.append(sub.text.strip())
    else:
        # Bare <urlset>: collect every <loc> regardless of nesting depth.
        for elem in root.iter():
            elem_tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if elem_tag == "loc" and elem.text:
                urls.append(elem.text.strip())

    return urls


async def _fetch_sitemap(client: httpx.AsyncClient, sitemap_url: str) -> list[str]:
    """Download one sitemap and return the URLs it lists."""
    try:
        response = await client.get(
            sitemap_url, timeout=SITEMAP_FETCH_TIMEOUT_SECONDS
        )
    except httpx.HTTPError:
        logger.warning("Sitemap request failed", sitemap_url=sitemap_url)
        return []
    if response.status_code != 200:
        logger.info(
            "Sitemap not available",
            sitemap_url=sitemap_url,
            status_code=response.status_code,
        )
        return []
    return _parse_sitemap_xml(response.text)


async def discover_urls(
    base_url: str,
    *,
    extra_paths: tuple[str, ...] = SITEMAP_PATHS,
) -> list[str]:
    """Discover all indexable URLs of a website via its sitemap.

    The function tries the standard sitemap locations in order. When a
    sitemap index is found, every nested sitemap is fetched and merged.
    The result is deduplicated and sorted.

    Parameters
    ----------
    base_url:
        Any URL on the target site. The scheme + host are used to build
        sitemap candidates (``{scheme}://{host}/sitemap.xml``).
    extra_paths:
        Override the default path list. Useful for robots.txt
        ``Sitemap:`` directives that point at a custom location.

    Returns
    -------
    list[str]
        Unique URLs from the sitemap, sorted. Empty list if the site
        does not expose a sitemap or none of the locations responded.
    """
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        logger.warning("discover_urls: base_url is not absolute", base_url=base_url)
        return []

    origin = f"{parsed.scheme}://{parsed.netloc}"
    sitemap_urls = [urljoin(origin, path) for path in extra_paths]

    found: list[str] = []
    nested_sitemaps: list[str] = []

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Try the explicit paths first.
        for sitemap_url in sitemap_urls:
            urls = await _fetch_sitemap(client, sitemap_url)
            if not urls:
                continue
            # If every entry points at another sitemap, treat this as
            # a sitemap index and recurse. Otherwise treat it as the
            # final list of pages.
            if all(url.rstrip("/").lower().endswith(".xml") for url in urls):
                nested_sitemaps.extend(urls)
            else:
                found.extend(urls)
            # One working sitemap is enough; bail out to avoid hammering
            # the site with multiple downloads.
            break

        for nested in nested_sitemaps:
            nested_urls = await _fetch_sitemap(client, nested)
            found.extend(nested_urls)

        # Also honour the ``Sitemap:`` directive in robots.txt.
        robots_url = urljoin(origin, "/robots.txt")
        try:
            r = await client.get(robots_url, timeout=SITEMAP_FETCH_TIMEOUT_SECONDS)
        except httpx.HTTPError:
            r = None
        if r is not None and r.status_code == 200:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    custom = line.split(":", 1)[1].strip()
                    if custom:
                        custom_urls = await _fetch_sitemap(client, custom)
                        found.extend(custom_urls)

    # Dedupe + sort by host-aware normalised URL.
    deduped = sorted(
        {
            _normalize_url_for_compare(u): u
            for u in found
            if u
        }.values()
    )
    logger.info(
        "Sitemap discovery complete",
        base_url=base_url,
        discovered=len(deduped),
    )
    return deduped


def filter_same_domain(urls: list[str], base_url: str) -> list[str]:
    """Keep only URLs that share the host of ``base_url``.

    External links (Twitter, LinkedIn, blog platforms, etc.) are
    dropped. The base URL itself is kept even if it appears in the
    list so the caller can always include the home page.
    """
    parsed = urlparse(base_url)
    if not parsed.netloc:
        return list(urls)

    base_netloc = parsed.netloc
    kept = [u for u in urls if _same_domain(u, base_netloc)]
    dropped = len(urls) - len(kept)
    if dropped:
        logger.info(
            "Filtered external URLs",
            base_url=base_url,
            kept=len(kept),
            dropped=dropped,
        )
    return kept


async def crawl_full_website(
    base_url: str,
    *,
    max_pages: int = 50,
    browser_config: BrowserConfig | None = None,
    run_config: CrawlerRunConfig | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[CrawlResult]:
    """Crawl every page of a site using its sitemap.

    Steps
    ----
    1. Fetch the sitemap (with the standard fallbacks) to discover URLs.
    2. Drop anything that points at another host.
    3. Cap the list to ``max_pages`` so a huge site does not exhaust
       the browser or hit the LLM budget.
    4. Hand the surviving URLs to ``crawl_websites``.

    When no sitemap is found, we fall back to crawling just the
    supplied ``base_url`` so the call is still useful.
    """
    discovered = await discover_urls(base_url)
    if not discovered:
        logger.warning(
            "No sitemap found, falling back to single page",
            base_url=base_url,
        )
        discovered = [base_url]

    same_domain = filter_same_domain(discovered, base_url)
    if not same_domain:
        logger.warning(
            "No same-domain URLs after filtering, falling back to base_url",
            base_url=base_url,
        )
        same_domain = [base_url]

    if max_pages > 0 and len(same_domain) > max_pages:
        logger.info(
            "Capping sitemap URLs to max_pages",
            max_pages=max_pages,
            original=len(same_domain),
        )
        # Keep the base URL first so the home page is always present,
        # then fill the remainder with the rest of the sitemap.
        ordered = [base_url] + [
            u for u in same_domain if u != _normalize_url_for_compare(base_url)
        ]
        same_domain = ordered[:max_pages]

    return await crawl_websites(
        same_domain,
        browser_config=browser_config,
        run_config=run_config,
        max_retries=max_retries,
    )
