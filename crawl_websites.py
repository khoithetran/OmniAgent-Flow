#!/usr/bin/env python
"""One-shot script to crawl and index website(s) into Qdrant.

Usage
-----
    python crawl_websites.py                    # interactive: prompted for URLs
    python crawl_websites.py https://example.com
    python crawl_websites.py https://a.com https://b.com --max-pages 10

The script:
  1. Prompts / reads URL(s) from the command line.
  2. For each URL: discovers pages via sitemap, crawls up to --max-pages.
  3. Indexes the resulting markdown into Qdrant.
  4. Reports per-site stats (pages crawled, chunks indexed, failures).

Requirements
------------
    Redis, Qdrant, and OpenAI must be running (or configured via .env).
    Run ``uvicorn src.main:app`` first, or rely on the inline init below.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add src/ to path so we can import from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

# Configure loguru to show timestamps and human-readable levels.
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level: <8} | {message}")

from src import crawler, rag
from src.config import get_settings


# ---------------------------------------------------------------------------
# Websites to crawl (add/remove as needed).
# ---------------------------------------------------------------------------

DEFAULT_WEBSITES: list[str] = []  # Interactive by default.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl one or more websites and index them into Qdrant.",
    )
    parser.add_argument(
        "urls",
        nargs="*",
        default=[],
        help="One or more URLs to crawl. Omit for interactive mode.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum pages to crawl per site (default: 10).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Drop the Qdrant collection before indexing.",
    )
    return parser.parse_args()


async def crawl_and_index(
    url: str,
    *,
    max_pages: int,
    replace: bool,
) -> dict[str, int | str]:
    """Crawl one URL (full site) and index into Qdrant.

    Returns a summary dict with keys: pages, chunks, failures, elapsed.
    """
    start = time.monotonic()

    logger.info("Discovering pages via sitemap for {url}", url=url)
    results = await crawler.crawl_full_website(
        url,
        max_pages=max_pages,
    )

    pages_ok = sum(1 for r in results if r.success)
    pages_fail = len(results) - pages_ok
    logger.info(
        "Crawl complete: {ok}/{total} pages OK, {fail} failures",
        ok=pages_ok,
        total=len(results),
        fail=pages_fail,
    )

    if pages_ok == 0:
        elapsed = time.monotonic() - start
        return {
            "url": url,
            "pages": 0,
            "chunks": 0,
            "failures": pages_fail,
            "elapsed": round(elapsed, 1),
        }

    # Index all results into Qdrant.
    summary = await rag.index_crawl_results(
        results,
        replace=replace,
    )
    elapsed = time.monotonic() - start

    logger.info(
        "Indexed {url}: {pages} pages, {chunks} chunks in {elapsed}s",
        url=url,
        pages=summary["pages"],
        chunks=summary["chunks"],
        elapsed=round(elapsed, 1),
    )

    return {
        "url": url,
        "pages": summary["pages"],
        "chunks": summary["chunks"],
        "failures": summary["failures"] + pages_fail,
        "elapsed": round(elapsed, 1),
    }


async def interactive_prompt() -> list[str]:
    """Read URLs from stdin, one per line, blank line to finish."""
    print("\nEnter URLs to crawl (one per line, blank line to finish):")
    urls: list[str] = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        urls.append(line)
    return urls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    args = parse_args()

    urls = args.urls
    if not urls:
        urls = DEFAULT_WEBSITES
    if not urls:
        urls = await interactive_prompt()

    if not urls:
        print("No URLs provided. Nothing to do.")
        return

    # Validate URLs look reasonable.
    for url in urls:
        if not url.startswith(("http://", "https://")):
            logger.error(
                "URL must start with http:// or https://: {url}",
                url=url,
            )
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  OmniAgent Flow — Website Crawler")
    print(f"{'='*60}")
    print(f"  URLs     : {len(urls)}")
    print(f"  Max pages: {args.max_pages} per site")
    print(f"  Replace  : {args.replace}")
    print(f"{'='*60}\n")

    # Initialise Qdrant and OpenAI clients inline (no FastAPI lifespan needed).
    try:
        rag.init_qdrant()
        rag.init_openai()
    except Exception:
        logger.exception("Failed to initialise Qdrant or OpenAI clients.")
        sys.exit(1)

    settings = get_settings()
    print(f"  Qdrant   : {settings.qdrant_url}")
    print(f"  Collection: {settings.qdrant_collection}\n")

    # Replace collection only on the first site; subsequent sites upsert into it.
    results: list[dict[str, int | str]] = []
    for i, url in enumerate(urls):
        summary = await crawl_and_index(
            url,
            max_pages=args.max_pages,
            replace=(args.replace and i == 0),
        )
        results.append(summary)

    # Summary table.
    total_pages = sum(r["pages"] for r in results)
    total_chunks = sum(r["chunks"] for r in results)
    total_fail = sum(r["failures"] for r in results)
    total_elapsed = sum(r["elapsed"] for r in results)  # type: ignore[arg-type]

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'URL':<40} {'Pages':>6} {'Chunks':>7} {'Failed':>7} {'Elapsed':>8}")
    print(f"  {'-'*40} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")
    for r in results:
        print(
            f"  {r['url']:<40} {r['pages']:>6} {r['chunks']:>7} "
            f"{r['failures']:>7} {r['elapsed']:>7}s"
        )
    print(f"  {'-'*40} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")
    print(
        f"  {'TOTAL':<40} {total_pages:>6} {total_chunks:>7} "
        f"{total_fail:>7} {total_elapsed:>7}s"
    )
    print(f"{'='*60}\n")

    if total_pages > 0:
        print("Crawl and index complete!")
    else:
        print("No pages were successfully crawled. Check the URLs and try again.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
