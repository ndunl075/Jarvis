"""Tests for DuckDuckGo snippet fetcher."""

from jarvis.tools.local.web_search_fetch import fetch_search_snippets


def test_fetch_search_snippets_returns_results():
    results = fetch_search_snippets("Python programming language", max_results=3)
    assert len(results) >= 1
    assert results[0]["url"].startswith("http")
    assert results[0]["title"]
