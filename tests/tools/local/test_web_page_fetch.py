"""Tests for web_page_fetch (HTML text extraction)."""

from __future__ import annotations

from unittest.mock import patch

from jarvis.tools.local.web_page_fetch import _extract_main_text, fetch_page_text


def test_extract_main_text_strips_chrome():
    html = """
    <html><head><script>var x=1;</script><style>body{}</style></head>
    <body>
      <nav>Home About Contact</nav>
      <article>
        <h1>Solar Panels</h1>
        <p>Solar panels convert sunlight to electricity via the photovoltaic effect.</p>
        <p>Efficiency of typical commercial panels ranges from 17 to 22 percent.</p>
      </article>
      <footer>Copyright 2025</footer>
    </body></html>
    """
    text = _extract_main_text(html)
    assert "photovoltaic" in text
    assert "Copyright" not in text
    assert "Home About Contact" not in text


def test_fetch_page_text_returns_empty_for_non_http():
    assert fetch_page_text("javascript:void(0)") == ""
    assert fetch_page_text("") == ""


def test_fetch_page_text_truncates_to_max_chars():
    long_html = (
        "<html><body><article>"
        + ("Sentence number one is here. " * 1000)
        + "</article></body></html>"
    )

    class FakeResp:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = long_html

        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, _url):
            return FakeResp()

    with patch("jarvis.tools.local.web_page_fetch.httpx.Client", FakeClient):
        out = fetch_page_text("https://example.com", max_chars=200)

    assert 0 < len(out) <= 220  # 200 + " …" + small slack
