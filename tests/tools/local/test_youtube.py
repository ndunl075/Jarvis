"""Tests for jarvis.tools.local.youtube."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.tools.local.youtube import (
    normalize_music_search_query,
    youtube_watch_url_for_query,
    youtube_watch_url_from_url,
)


def test_normalize_music_search_query_strips_command_words():
    raw = "hey jarvis can you play the song Hotel California on YouTube please"
    assert normalize_music_search_query(raw) == "Hotel California"


def test_youtube_watch_url_parses_first_video_id():
    html = 'prefix "videoId":"dQw4w9WgXcQ" suffix'
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = MagicMock(
        json=lambda: {},
        raise_for_status=MagicMock(),
    )
    mock_client.get.return_value = mock_resp
    with patch("jarvis.tools.local.youtube.httpx.Client", return_value=mock_client):
        with patch(
            "jarvis.tools.local.youtube._search_via_innertube",
            return_value=None,
        ):
            url = youtube_watch_url_for_query("never gonna give you up")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&autoplay=1"


def test_youtube_watch_url_empty_query():
    assert youtube_watch_url_for_query("   ") is None


def test_youtube_watch_url_from_search_results_page():
    search_url = "https://www.youtube.com/results?search_query=hotel+california"
    with patch(
        "jarvis.tools.local.youtube.youtube_watch_url_for_query",
        return_value="https://www.youtube.com/watch?v=abc&autoplay=1",
    ) as resolve:
        url = youtube_watch_url_from_url(search_url)
    assert url == "https://www.youtube.com/watch?v=abc&autoplay=1"
    resolve.assert_called_once()
