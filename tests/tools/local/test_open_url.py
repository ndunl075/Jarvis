"""Tests for jarvis.tools.local.open_url.OpenUrlTool."""

from __future__ import annotations

from unittest.mock import patch

from jarvis.tools.local.open_url import OpenUrlArgs, OpenUrlTool


async def test_opens_http_url_through_platform_seam():
    """The FULL url (with path + query) is what's passed to webbrowser;
    the spoken response is the friendly site name, not the URL."""
    with patch("jarvis.tools.local.open_url.winplat.open_url") as op:
        result = await OpenUrlTool().execute(
            OpenUrlArgs(url="http://example.com/page?q=1")
        )
    assert result.success
    op.assert_called_once_with("http://example.com/page?q=1")
    # Spoken output uses the friendly name (capitalised first label
    # fallback for unknown sites), not the URL.
    assert (result.output or "").startswith("Opening Example")


async def test_opens_https_url():
    with patch("jarvis.tools.local.open_url.winplat.open_url") as op:
        result = await OpenUrlTool().execute(OpenUrlArgs(url="https://anthropic.com"))
    assert result.success
    op.assert_called_once_with("https://anthropic.com")


async def test_google_search_url_speaks_query_not_url():
    """Live bug: 'Google how to make pasta' was being spoken as
    'Opening google dot com, sir' (or worse, markdown). The friendly
    response decodes the search URL's q= parameter."""
    url = "https://www.google.com/search?q=how+to+make+pasta"
    with patch("jarvis.tools.local.open_url.winplat.open_url") as op:
        result = await OpenUrlTool().execute(OpenUrlArgs(url=url))
    assert result.success
    op.assert_called_once_with(url)
    assert result.output == "Searching the web for how to make pasta, sir."


async def test_known_site_uses_friendly_name():
    """youtube.com -> 'YouTube' (not 'Youtube.com'). Curated map."""
    with patch("jarvis.tools.local.open_url.winplat.open_url"):
        result = await OpenUrlTool().execute(
            OpenUrlArgs(url="https://www.youtube.com/watch?v=abc")
        )
    assert result.output == "Opening YouTube, sir."


async def test_url_passed_verbatim_includes_query_string():
    """webbrowser receives the full URL with query string intact —
    defends against the live bug where it sounded like only the homepage
    opened (which was actually the spoken response, not the URL)."""
    url = "https://www.google.com/search?q=python+regex"
    with patch("jarvis.tools.local.open_url.winplat.open_url") as op:
        await OpenUrlTool().execute(OpenUrlArgs(url=url))
    assert op.call_args.args[0] == url
    assert "search?q=" in op.call_args.args[0]


async def test_rejects_non_http_scheme():
    with patch("jarvis.tools.local.open_url.winplat.open_url") as op:
        result = await OpenUrlTool().execute(OpenUrlArgs(url="file:///etc/passwd"))
    assert not result.success
    assert "scheme" in (result.error or "")
    op.assert_not_called()


async def test_rejects_url_without_host():
    with patch("jarvis.tools.local.open_url.winplat.open_url") as op:
        result = await OpenUrlTool().execute(OpenUrlArgs(url="https://"))
    assert not result.success
    op.assert_not_called()


def test_requires_confirmation_false():
    """Phase 4 contract: no tool sets the flag (UX deferred)."""
    assert OpenUrlTool().requires_confirmation is False
