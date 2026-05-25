"""Tests for jarvis.tools.local.play_youtube_music."""

from __future__ import annotations

from unittest.mock import patch

from jarvis.core.request_context import current_user_transcription
from jarvis.tools.local.play_youtube_music import PlayYoutubeMusicArgs, PlayYoutubeMusicTool


async def test_play_youtube_music_opens_watch_url():
    with (
        patch(
            "jarvis.tools.local.play_youtube_music.youtube_watch_url_for_query",
            return_value="https://www.youtube.com/watch?v=abc&autoplay=1",
        ) as resolve,
        patch("jarvis.tools.local.play_youtube_music.winplat.open_url") as open_url,
    ):
        result = await PlayYoutubeMusicTool().execute(
            PlayYoutubeMusicArgs(query="bohemian rhapsody queen")
        )
    assert result.success
    resolve.assert_called_once_with("bohemian rhapsody queen")
    open_url.assert_called_once_with(
        "https://www.youtube.com/watch?v=abc&autoplay=1"
    )
    assert "YouTube" in (result.output or "")


async def test_play_youtube_falls_back_to_full_utterance():
    token = current_user_transcription.set(
        "play the song Hotel California by the Eagles on YouTube"
    )
    try:
        with (
            patch(
                "jarvis.tools.local.play_youtube_music.youtube_watch_url_for_query",
                return_value="https://www.youtube.com/watch?v=x&autoplay=1",
            ) as resolve,
            patch("jarvis.tools.local.play_youtube_music.winplat.open_url"),
        ):
            result = await PlayYoutubeMusicTool().execute(
                PlayYoutubeMusicArgs(query="")
            )
    finally:
        current_user_transcription.reset(token)
    assert result.success
    resolve.assert_called_once_with("Hotel California by the Eagles")


async def test_play_youtube_music_empty_without_context():
    result = await PlayYoutubeMusicTool().execute(PlayYoutubeMusicArgs(query="  "))
    assert not result.success
