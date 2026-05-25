"""Tests for Jina fallback in web_page_fetch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.tools.local.web_page_fetch import fetch_page_text


def test_jina_fallback_when_direct_fetch_short():
    direct_resp = MagicMock()
    direct_resp.raise_for_status = MagicMock()
    direct_resp.headers = {"content-type": "text/html"}
    direct_resp.text = "<html><body><p>Hi</p></body></html>"

    jina_resp = MagicMock()
    jina_resp.raise_for_status = MagicMock()
    jina_resp.text = "x" * 600

    def fake_get(url, **kwargs):
        if "r.jina.ai" in url:
            return jina_resp
        return direct_resp

    with patch("jarvis.tools.local.web_page_fetch.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.get.side_effect = fake_get
        text = fetch_page_text(
            "https://example.com/article",
            max_chars=1000,
            use_jina_fallback=True,
        )
    assert len(text) >= 600
