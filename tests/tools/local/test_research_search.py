"""Tests for research_search (Brave + DDG fallback)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.tools.local.research_search import fetch_search_snippets


def test_brave_provider_parses_results():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "web": {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "description": "A site",
                },
            ],
        },
    }
    with patch("jarvis.tools.local.research_search.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.get.return_value = (
            mock_response
        )
        hits = fetch_search_snippets(
            "test query",
            max_results=5,
            provider="brave",
            brave_api_key="test-key",
        )
    assert len(hits) == 1
    assert hits[0]["url"] == "https://example.com"
    assert hits[0]["title"] == "Example"


def test_brave_failure_falls_back_to_ddg():
    with patch(
        "jarvis.tools.local.research_search._fetch_brave",
        side_effect=RuntimeError("quota"),
    ), patch(
        "jarvis.tools.local.research_search.fetch_ddg",
        return_value=[{"title": "DDG", "url": "https://d.com", "content": "x"}],
    ) as ddg:
        hits = fetch_search_snippets(
            "q",
            provider="brave",
            brave_api_key="key",
        )
    ddg.assert_called_once()
    assert hits[0]["title"] == "DDG"
