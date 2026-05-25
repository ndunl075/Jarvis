"""Play music by searching YouTube and opening the top video."""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from jarvis.core.request_context import current_user_transcription
from jarvis.platform import windows as winplat
from jarvis.tools.local.youtube import (
    normalize_music_search_query,
    youtube_watch_url_for_query,
)
from jarvis.tools.registry import ToolResult

log = logging.getLogger(__name__)


class PlayYoutubeMusicArgs(BaseModel):
    query: str = Field(
        description=(
            "YouTube search terms ONLY — song title and/or artist extracted "
            "from what the user said. Example: user says 'play Hotel "
            "California by the Eagles on YouTube' -> query='Hotel California "
            "Eagles'. Do NOT include command words (play, youtube, please)."
        ),
    )


def _still_has_command_words(q: str) -> bool:
    low = q.lower()
    return any(
        token in low
        for token in ("play ", "youtube", "jarvis", "please put on", "listen to")
    )


def _resolve_search_query(args_query: str) -> str:
    """Prefer a clean LLM `query`; fall back to the full voice utterance."""
    utterance = current_user_transcription.get()
    cleaned = normalize_music_search_query(args_query)
    if cleaned and len(cleaned) >= 2 and not _still_has_command_words(cleaned):
        return cleaned
    if utterance:
        from_utterance = normalize_music_search_query(utterance)
        if from_utterance:
            log.info(
                "play_youtube_music: derived search %r from utterance",
                from_utterance,
            )
            return from_utterance
    return cleaned


class PlayYoutubeMusicTool:
    name: str = "play_youtube_music"
    description: str = (
        "Find and open a specific YouTube video in the browser with autoplay. "
        "Use when the user wants to hear a song, track, artist, or music. "
        "Pass `query` as the distilled song/artist search string from their "
        "full request — the tool resolves the top video and opens it directly "
        "(not a search results page). Do NOT use open_url for music playback."
    )
    args_schema = PlayYoutubeMusicArgs
    requires_confirmation: bool = False

    async def execute(self, args: PlayYoutubeMusicArgs) -> ToolResult:
        search_q = _resolve_search_query(args.query)
        if not search_q:
            return ToolResult(
                success=False,
                error="could not determine what to play on YouTube",
            )

        watch_url = await asyncio.to_thread(youtube_watch_url_for_query, search_q)
        if watch_url is None:
            return ToolResult(
                success=False,
                error=f"could not find a YouTube video for {search_q!r}",
            )
        try:
            await asyncio.to_thread(winplat.open_url, watch_url)
        except OSError as e:
            return ToolResult(success=False, error=f"could not open YouTube: {e}")
        log.info("play_youtube_music: opened %s for query %r", watch_url, search_q)
        return ToolResult(
            success=True,
            output=f"Playing {search_q} on YouTube, sir.",
        )
