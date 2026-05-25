"""YouTube search URL redirect in open_url."""

from __future__ import annotations

from unittest.mock import patch

from jarvis.tools.local.open_url import OpenUrlArgs, OpenUrlTool


async def test_open_url_youtube_search_resolves_to_watch():
    search = "https://www.youtube.com/results?search_query=hotel+california"
    watch = "https://www.youtube.com/watch?v=abc&autoplay=1"
    with (
        patch(
            "jarvis.tools.local.open_url.youtube_watch_url_from_url",
            return_value=watch,
        ),
        patch("jarvis.tools.local.open_url.winplat.open_url") as open_url,
    ):
        result = await OpenUrlTool().execute(OpenUrlArgs(url=search))
    assert result.success
    open_url.assert_called_once_with(watch)
    assert "Playing" in (result.output or "")
