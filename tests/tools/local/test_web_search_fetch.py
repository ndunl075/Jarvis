"""Tests for the DuckDuckGo snippet fetcher.

These are hermetic: the HTTP transport is replaced with `httpx.MockTransport`
so the assertions exercise the *parsing* of DuckDuckGo's HTML endpoint rather
than DuckDuckGo's live availability. A real-network smoke test lives at the
bottom behind `@pytest.mark.manual` (skipped by default, see pyproject.toml).
"""

from __future__ import annotations

import httpx
import pytest

from jarvis.tools.local import web_search_fetch
from jarvis.tools.local.web_search_fetch import (
    fetch_search_snippets,
    format_snippets_for_prompt,
)

# ---------------------------------------------------------------------------
# Fixture HTML
#
# Modelled on a real https://html.duckduckgo.com/html/ response. The shape
# that matters to the parser:
#   - the outbound link is wrapped in //duckduckgo.com/l/?uddg=<percent-encoded>
#   - `class="result__a"` precedes `href=` on the anchor
#   - both anchors sit on a single (long) line, as DuckDuckGo emits them
#   - snippets carry inline <b> highlighting and HTML entities
#
# Composed from parts rather than pasted verbatim purely so the source stays
# inside the 100-column limit; the assembled string is byte-for-byte the
# single-line markup DuckDuckGo actually serves.
# ---------------------------------------------------------------------------

_PY_ORG = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2F&amp;rut=6f1"
_WIKIPEDIA = (
    "//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2F"
    "Python_%28programming_language%29&amp;rut=9c2"
)
_DOCS = "//docs.python.org/3/tutorial/"


def _result_block(href: str, title: str, snippet: str) -> str:
    """One `div.result` exactly as the HTML endpoint lays it out."""
    return (
        '  <div class="result results_links results_links_deep web-result">\n'
        '    <div class="links_main links_deep result__body">\n'
        '      <h2 class="result__title">\n'
        f'        <a rel="nofollow" class="result__a" href="{href}">{title}</a>\n'
        "      </h2>\n"
        f'      <a class="result__snippet" href="{href}">{snippet}</a>\n'
        '      <div class="clear"></div>\n'
        "    </div>\n"
        "  </div>\n"
    )


_DDG_HTML_FIXTURE = (
    "<!DOCTYPE html>\n"
    "<html><body>\n"
    '<div class="serp__results">\n'
    + _result_block(
        _PY_ORG,
        "Welcome to Python.org",
        "The official home of the <b>Python Programming Language</b>.",
    )
    + _result_block(
        _WIKIPEDIA,
        "Python (programming language) - Wikipedia",
        # DuckDuckGo wraps long snippets, sometimes mid-<b>.
        "Python is a high-level, general-purpose <b>programming\n"
        "language</b>. Guido &amp; friends released it in 1991.",
    )
    + _result_block(
        _DOCS,
        "The Python Tutorial",
        "Python is an easy to learn, powerful language.",
    )
    + "</div>\n"
    "</body></html>\n"
)

_NO_RESULTS_HTML = """<!DOCTYPE html>
<html><body>
<div class="no-results">No results found for that query.</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ddg(monkeypatch):
    """Route `httpx.Client` inside web_search_fetch through a MockTransport.

    Returns a callable: `mock_ddg(handler)` where `handler` is the usual
    `httpx.MockTransport` request handler. Every request the module makes is
    also appended to the returned list-like `.requests` for contract checks.
    """
    real_client_cls = httpx.Client

    def install(handler):
        seen: list[httpx.Request] = []

        def recording_handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        def client_factory(**kwargs):
            kwargs.pop("transport", None)
            return real_client_cls(
                transport=httpx.MockTransport(recording_handler), **kwargs
            )

        monkeypatch.setattr(web_search_fetch.httpx, "Client", client_factory)
        return seen

    return install


def _ok(body: str, status: int = 200):
    return lambda request: httpx.Response(status, text=body)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_fetch_search_snippets_returns_results(mock_ddg):
    mock_ddg(_ok(_DDG_HTML_FIXTURE))

    results = fetch_search_snippets("Python programming language", max_results=3)

    assert len(results) == 3
    assert results[0]["url"].startswith("http")
    assert results[0]["title"]
    assert all(set(r) == {"title", "url", "content"} for r in results)


def test_uddg_redirect_wrapper_is_unwrapped(mock_ddg):
    """Outbound links come back wrapped in //duckduckgo.com/l/?uddg=...;
    the real target must be percent-decoded out of the query string."""
    mock_ddg(_ok(_DDG_HTML_FIXTURE))

    results = fetch_search_snippets("python", max_results=3)

    assert results[0]["url"] == "https://www.python.org/"
    assert results[1]["url"] == (
        "https://en.wikipedia.org/wiki/Python_(programming_language)"
    )


def test_protocol_relative_href_without_uddg_gets_https_scheme(mock_ddg):
    mock_ddg(_ok(_DDG_HTML_FIXTURE))

    results = fetch_search_snippets("python", max_results=3)

    assert results[2]["url"] == "https://docs.python.org/3/tutorial/"


def test_snippet_markup_is_stripped_and_entities_unescaped(mock_ddg):
    mock_ddg(_ok(_DDG_HTML_FIXTURE))

    results = fetch_search_snippets("python", max_results=3)

    assert results[0]["content"] == (
        "The official home of the Python Programming Language."
    )
    # Second snippet spans a newline inside <b>...</b> and carries an entity.
    assert "<b>" not in results[1]["content"]
    assert "Guido & friends released it in 1991." in results[1]["content"]


def test_titles_are_parsed_verbatim(mock_ddg):
    mock_ddg(_ok(_DDG_HTML_FIXTURE))

    results = fetch_search_snippets("python", max_results=3)

    assert [r["title"] for r in results] == [
        "Welcome to Python.org",
        "Python (programming language) - Wikipedia",
        "The Python Tutorial",
    ]


def test_max_results_caps_the_returned_list(mock_ddg):
    mock_ddg(_ok(_DDG_HTML_FIXTURE))

    assert len(fetch_search_snippets("python", max_results=1)) == 1
    assert len(fetch_search_snippets("python", max_results=2)) == 2
    # Asking for more than the page holds is not an error.
    assert len(fetch_search_snippets("python", max_results=99)) == 3


def test_unparseable_page_returns_empty_list(mock_ddg):
    """A results page we cannot parse is not an exception — the caller
    surfaces a friendly 'no results' message."""
    mock_ddg(_ok(_NO_RESULTS_HTML))

    assert fetch_search_snippets("asdkjhaskdjh", max_results=5) == []


# ---------------------------------------------------------------------------
# Request contract / short-circuits
# ---------------------------------------------------------------------------


def test_blank_query_short_circuits_without_any_request(mock_ddg):
    seen = mock_ddg(_ok(_DDG_HTML_FIXTURE))

    assert fetch_search_snippets("", max_results=3) == []
    assert fetch_search_snippets("   ", max_results=3) == []
    assert seen == []


def test_query_is_posted_to_the_ddg_html_endpoint(mock_ddg):
    seen = mock_ddg(_ok(_DDG_HTML_FIXTURE))

    fetch_search_snippets("  python programming  ", max_results=1)

    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "https://html.duckduckgo.com/html/"
    # The query is trimmed before it goes on the wire.
    assert b"q=python+programming" in request.content
    assert request.headers["User-Agent"] == "Jarvis/1.0 (local research)"


def test_http_error_propagates(mock_ddg):
    """raise_for_status() is deliberate: the research tool distinguishes
    'DuckDuckGo is unhappy' from 'no results'."""
    mock_ddg(_ok("rate limited", status=429))

    with pytest.raises(httpx.HTTPStatusError):
        fetch_search_snippets("python", max_results=3)


def test_transport_error_propagates(mock_ddg):
    def boom(request):
        raise httpx.ConnectError("offline", request=request)

    mock_ddg(boom)

    with pytest.raises(httpx.ConnectError):
        fetch_search_snippets("python", max_results=3)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_format_snippets_for_prompt_numbers_and_labels_each_hit(mock_ddg):
    mock_ddg(_ok(_DDG_HTML_FIXTURE))

    block = format_snippets_for_prompt(fetch_search_snippets("python", max_results=2))

    assert block.startswith("[1] Welcome to Python.org\n")
    assert "URL: https://www.python.org/" in block
    assert "[2] Python (programming language) - Wikipedia" in block
    # Hits are separated by a blank line.
    assert "\n\n[2] " in block


def test_format_snippets_for_prompt_empty_input():
    assert format_snippets_for_prompt([]) == ""


# ---------------------------------------------------------------------------
# Live smoke test (operator-run)
# ---------------------------------------------------------------------------


@pytest.mark.manual
def test_live_duckduckgo_still_matches_our_parser():
    """Real-network canary: DuckDuckGo can change its HTML at any time, which
    would silently break the parser above without breaking the mocked tests.

    Skipped by default (`addopts = -m 'not manual'`). Run by hand with:
        pytest -m manual tests/tools/local/test_web_search_fetch.py
    """
    results = fetch_search_snippets("Python programming language", max_results=3)
    assert len(results) >= 1
    assert results[0]["url"].startswith("http")
    assert results[0]["title"]


# ---------------------------------------------------------------------------
# Regression: highlighted titles and link/snippet pairing
#
# The title group used to be `[^<]+`, which cannot match across the <b> tags
# DuckDuckGo wraps around query terms in a title. Such an anchor did not match
# at all, so the result disappeared — and because links and snippets were
# collected into two independent lists and paired by index, the missing link
# shifted every later snippet up by one. The visible symptom was a result that
# kept its own title and URL but carried the NEXT site's description, which
# then reached the summarizer as a numbered citation.
# ---------------------------------------------------------------------------


_HIGHLIGHTED_HTML = (
    "<!DOCTYPE html>\n<html><body>\n"
    '<div class="serp__results">\n'
    + _result_block("//alpha.example/", "Plain Alpha", "snippet about ALPHA")
    + _result_block(
        "//beta.example/",
        "Beta <b>highlighted</b> title",
        "snippet about BETA",
    )
    + _result_block("//gamma.example/", "Plain Gamma", "snippet about GAMMA")
    + "</div>\n</body></html>\n"
)


def test_title_with_inline_highlighting_is_kept(mock_ddg):
    mock_ddg(_ok(_HIGHLIGHTED_HTML))
    results = fetch_search_snippets("q")

    assert len(results) == 3, "a <b>-highlighted title must not drop the result"
    assert results[1]["title"] == "Beta highlighted title"
    assert "<b>" not in results[1]["title"]


def test_highlighted_title_does_not_shift_later_snippets(mock_ddg):
    mock_ddg(_ok(_HIGHLIGHTED_HTML))
    results = fetch_search_snippets("q")

    # Each result must carry its OWN snippet, not the next site's.
    for r in results:
        host = r["url"].split("//")[-1].split(".")[0].upper()
        assert r["content"] == f"snippet about {host}", (
            f"{r['url']} carries {r['content']!r}"
        )


def test_link_without_a_snippet_does_not_shift_the_rest(mock_ddg):
    """A link whose snippet anchor is absent must yield an empty snippet
    rather than borrowing the following result's."""
    html = (
        "<!DOCTYPE html>\n<html><body>\n"
        '<div class="serp__results">\n'
        '  <a rel="nofollow" class="result__a" href="//alpha.example/">Alpha</a>\n'
        + _result_block("//beta.example/", "Beta", "snippet about BETA")
        + "</div>\n</body></html>\n"
    )
    mock_ddg(_ok(html))
    results = fetch_search_snippets("q")

    assert [r["url"] for r in results] == ["https://alpha.example/", "https://beta.example/"]
    assert results[1]["content"] == "snippet about BETA"
    # Alpha had no snippet; content falls back to its own title, never Beta's.
    assert results[0]["content"] == "Alpha"


def test_max_results_still_caps_after_pairing(mock_ddg):
    mock_ddg(_ok(_HIGHLIGHTED_HTML))
    assert len(fetch_search_snippets("q", max_results=2)) == 2
