"""Open a URL in the user's default browser.

The spoken response (ToolResult.output) is deliberately *not* the URL.
URLs read poorly through TTS ("h-t-t-p-s colon slash slash w-w-w dot
google dot com slash search question mark q equals…") and the LLM was
also generating markdown link syntax that bled into playback. The
output for a Google search URL is "Searching the web for <query>, sir.";
for any other URL it's "Opening <FriendlyName>, sir." where the friendly
name comes from a small site-name map (with a capitalised-first-label
fallback for unknown sites)."""

from __future__ import annotations

import asyncio
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field

from jarvis.platform import windows as winplat
from jarvis.tools.local.youtube import youtube_watch_url_from_url
from jarvis.tools.registry import ToolResult

# Hand-curated readable names for sites that come up in conversational
# voice use. Extend as needed; unknown domains fall back to the
# capitalised first label.
_FRIENDLY_NAMES: dict[str, str] = {
    "google.com": "Google",
    "google.co.uk": "Google",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "github.com": "GitHub",
    "stackoverflow.com": "Stack Overflow",
    "wikipedia.org": "Wikipedia",
    "reddit.com": "Reddit",
    "twitter.com": "Twitter",
    "x.com": "Twitter",
    "amazon.com": "Amazon",
    "netflix.com": "Netflix",
    "spotify.com": "Spotify",
    "discord.com": "Discord",
    "steamcommunity.com": "Steam",
    "steampowered.com": "Steam",
}


def _registered_domain(netloc: str) -> str:
    host = netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _friendly_name(netloc: str) -> str:
    domain = _registered_domain(netloc)
    if domain in _FRIENDLY_NAMES:
        return _FRIENDLY_NAMES[domain]
    label = domain.split(".")[0]
    return label.title() if label else domain


def _spoken_response_for(url: str) -> str:
    """Compose the TTS line for a successfully opened URL. Hides the URL
    entirely; reads as natural English."""
    parsed = urlparse(url)
    domain = _registered_domain(parsed.netloc)
    if domain.startswith("google.") and parsed.path == "/search":
        q = parse_qs(parsed.query).get("q", [""])[0]
        if q:
            return f"Searching the web for {q}, sir."
    return f"Opening {_friendly_name(parsed.netloc)}, sir."


class OpenUrlArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL to open.")


class OpenUrlTool:
    name: str = "open_url"
    description: str = (
        "Opens a URL in the default browser, or performs a web search. "
        "Only use when the user explicitly says 'open <url>', "
        "'search <query>', or 'google <query>'. Do NOT use for playing "
        "music — use play_youtube_music instead (it opens the video directly)."
    )
    args_schema = OpenUrlArgs
    requires_confirmation: bool = False

    async def execute(self, args: OpenUrlArgs) -> ToolResult:
        parsed = urlparse(args.url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                success=False,
                error=f"only http(s) URLs are accepted, got scheme {parsed.scheme!r}",
            )
        if not parsed.netloc:
            return ToolResult(success=False, error="URL has no host")
        open_target = args.url
        spoken = _spoken_response_for(args.url)
        # LLM sometimes opens a YouTube search page instead of play_youtube_music.
        watch = await asyncio.to_thread(youtube_watch_url_from_url, args.url)
        if watch:
            open_target = watch
            q = parse_qs(urlparse(args.url).query).get("search_query", [""])[0]
            if q:
                spoken = f"Playing {q}, sir."
            else:
                spoken = "Playing that on YouTube, sir."
        await asyncio.to_thread(winplat.open_url, open_target)
        return ToolResult(success=True, output=spoken)
