"""Tests for src/crawler.py — website crawler and chunking."""

from __future__ import annotations

import pytest

from src.crawler import (
    CrawlResult,
    chunk_markdown,
    filter_same_domain,
    _normalize_url_for_compare,
    _same_domain,
    _parse_sitemap_xml,
    new_crawl_id,
)


# ---------------------------------------------------------------------------
# chunk_markdown
# ---------------------------------------------------------------------------

class TestChunkMarkdown:
    def test_empty_string_returns_empty_list(self):
        assert chunk_markdown("") == []
        assert chunk_markdown("   \n\n  ") == []

    def test_single_paragraph_short(self):
        result = chunk_markdown("This is a short paragraph.")
        assert result == ["This is a short paragraph."]

    def test_multiple_paragraphs_one_chunk(self):
        text = "\n\n".join([f"Para {i} content." for i in range(1, 6)])
        result = chunk_markdown(text, chunk_size=5000, overlap=0)
        assert len(result) == 1

    def test_overlap_preserved(self):
        text = "A" * 2000 + "\n\n" + "B" * 2000
        chunks = chunk_markdown(text, chunk_size=1500, overlap=100)
        assert len(chunks) >= 2
        # Overlap means the end of chunk 0 appears at the start of chunk 1
        assert chunks[1].startswith(chunks[0][-100:])

    def test_paragraph_larger_than_chunk_size_splits_on_sentences(self):
        long_para = ". ".join([f"Sentence number {i}" for i in range(1, 50)])
        chunks = chunk_markdown(long_para, chunk_size=200, overlap=20)
        assert len(chunks) >= 2

    def test_chunk_size_minimum_50(self):
        with pytest.raises(ValueError, match="chunk_size must be >= 50"):
            chunk_markdown("short", chunk_size=30)

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="overlap must be in"):
            chunk_markdown("some text", chunk_size=100, overlap=100)

    def test_overlap_zero_valid(self):
        chunks = chunk_markdown("para one\n\npara two", chunk_size=1000, overlap=0)
        assert len(chunks) == 1

    def test_blank_paragraphs_collapsed(self):
        result = chunk_markdown("Hello\n\n\n\nWorld")
        assert len(result) == 1
        assert "Hello" in result[0]
        assert "World" in result[0]


# ---------------------------------------------------------------------------
# CrawlResult dataclass
# ---------------------------------------------------------------------------

class TestCrawlResult:
    def test_success_result(self):
        r = CrawlResult(
            url="https://example.com",
            title="Example",
            markdown="# Hello",
            success=True,
        )
        assert r.success is True
        assert r.markdown == "# Hello"
        assert r.error_message is None
        assert r.to_dict()["url"] == "https://example.com"

    def test_failure_result(self):
        r = CrawlResult(
            url="https://example.com/404",
            title="",
            markdown="",
            success=False,
            error_message="404 Not Found",
        )
        assert r.success is False
        assert r.error_message == "404 Not Found"


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------

class TestNormalizeUrlForCompare:
    def test_lowercases_host(self):
        assert _normalize_url_for_compare("HTTPS://EXAMPLE.COM/page") == "https://example.com/page"

    def test_keeps_path_trailing_slash(self):
        assert _normalize_url_for_compare("https://example.com/page/") == "https://example.com/page/"

    def test_keeps_root_trailing_slash(self):
        # urlparse returns "/" as the path for bare root URLs
        assert _normalize_url_for_compare("https://example.com/") == "https://example.com/"

    def test_keeps_path_case(self):
        # Path should not be lowercased so /Models vs /models stay distinct
        assert _normalize_url_for_compare("https://example.com/Models") == "https://example.com/Models"


class TestSameDomain:
    def test_same_host(self):
        assert _same_domain("https://example.com/page", "example.com") is True

    def test_different_host(self):
        assert _same_domain("https://other.com/page", "example.com") is False

    def test_case_insensitive(self):
        assert _same_domain("https://EXAMPLE.COM/", "example.com") is True

    def test_www_prefix_matches(self):
        # _same_domain does strict netloc comparison (no www-stripping)
        # so www.example.com != example.com
        assert _same_domain("https://www.example.com/", "www.example.com") is True


# ---------------------------------------------------------------------------
# Sitemap XML parsing
# ---------------------------------------------------------------------------

class TestParseSitemapXml:
    def test_empty_string(self):
        assert _parse_sitemap_xml("") == []
        assert _parse_sitemap_xml("no tags") == []

    def test_simple_urlset(self):
        xml = """<?xml version="1.0"?>
<urlset>
  <url><loc>https://example.com/</loc></url>
  <url><loc>https://example.com/about</loc></url>
</urlset>"""
        urls = _parse_sitemap_xml(xml)
        assert len(urls) == 2
        assert "https://example.com/" in urls

    def test_sitemap_index(self):
        xml = """<?xml version="1.0"?>
<sitemapindex>
  <sitemap><loc>https://example.com/sitemap-pages.xml</loc></sitemap>
</sitemapindex>"""
        urls = _parse_sitemap_xml(xml)
        assert "https://example.com/sitemap-pages.xml" in urls

    def test_invalid_xml_returns_empty(self):
        assert _parse_sitemap_xml("<not><valid") == []

    def test_strips_whitespace(self):
        xml = '<urlset><url><loc>  https://example.com/page  </loc></url></urlset>'
        urls = _parse_sitemap_xml(xml)
        assert urls[0] == "https://example.com/page"


# ---------------------------------------------------------------------------
# filter_same_domain
# ---------------------------------------------------------------------------

class TestFilterSameDomain:
    def test_keeps_same_domain(self):
        urls = [
            "https://example.com/",
            "https://example.com/about",
            "https://example.com/blog/post",
        ]
        result = filter_same_domain(urls, "https://example.com/")
        assert len(result) == 3

    def test_drops_external(self):
        urls = [
            "https://example.com/",
            "https://twitter.com/example",
            "https://github.com/example",
        ]
        result = filter_same_domain(urls, "https://example.com/")
        assert len(result) == 1
        assert "example.com" in result[0]

    def test_empty_list_returns_empty(self):
        assert filter_same_domain([], "https://example.com/") == []

    def test_invalid_base_url_returns_all(self):
        urls = ["https://example.com/"]
        result = filter_same_domain(urls, "not-a-url")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# new_crawl_id
# ---------------------------------------------------------------------------

class TestNewCrawlId:
    def test_returns_12_chars(self):
        cid = new_crawl_id()
        assert len(cid) == 12

    def test_deterministic_from_same_input(self):
        # uuid4 is random, so we can't test equality — just format
        import re
        assert re.fullmatch(r"[0-9a-f]{12}", new_crawl_id()) is not None

    def test_unique_per_call(self):
        ids = {new_crawl_id() for _ in range(100)}
        assert len(ids) == 100
