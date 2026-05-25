"""Resolve a YouTube search query to a watch URL (top video result)."""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import parse_qs, quote_plus, unquote_plus, urlparse

import httpx

log = logging.getLogger(__name__)

_SEARCH_URL = "https://www.youtube.com/results?search_query={query}"
_INNERTUBE_SEARCH_URL = "https://www.youtube.com/youtubei/v1/search"
_INNERTUBE_CLIENT = {
    "clientName": "WEB",
    "clientVersion": "2.20240214.00.00",
}
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_TIMEOUT = 10.0
_VIDEO_ID_LEN = 11
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Leading fillers — peeled iteratively (same idea as intent_router).
_LEADING_FILLER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^hey\s+jarvis,?\s+", re.IGNORECASE),
    re.compile(r"^(?:(?:can|could|would)\s+you\s+(?:please\s+)?)", re.IGNORECASE),
    re.compile(r"^please\s+", re.IGNORECASE),
    re.compile(r"^(?:i\s+)?(?:want\s+(?:you\s+)?to\s+)", re.IGNORECASE),
    re.compile(r"^(?:play|put\s+on|queue|start)\s+", re.IGNORECASE),
    re.compile(r"^(?:the\s+)?(?:song|track|music|video)\s+", re.IGNORECASE),
    re.compile(r"^listen\s+to\s+", re.IGNORECASE),
)
_TRAILING_FILLER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\s+on\s+youtube(?:\s+music)?$", re.IGNORECASE),
    re.compile(r"\s+in\s+(?:the\s+)?browser$", re.IGNORECASE),
    re.compile(r"\s+for\s+me$", re.IGNORECASE),
    re.compile(r"\s+please$", re.IGNORECASE),
    re.compile(r"\s+sir$", re.IGNORECASE),
)
_TRAILING_PUNCT = re.compile(r"[\s.!?,]+$")

_HTML_VIDEO_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"'),
    re.compile(r"watch\?v=([A-Za-z0-9_-]{11})"),
)


def _peel_patterns(s: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    while True:
        new = s
        for pat in patterns:
            new = pat.sub("", new, count=1)
        if new == s:
            break
        s = new.strip()
    return s


def normalize_music_search_query(text: str) -> str:
    """Distill a voice utterance or LLM arg down to YouTube search terms."""
    s = text.strip()
    if not s:
        return ""
    s = _TRAILING_PUNCT.sub("", s)
    s = _peel_patterns(s, _LEADING_FILLER_PATTERNS)
    s = _peel_patterns(s, _TRAILING_FILLER_PATTERNS)
    s = _TRAILING_PUNCT.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _valid_video_id(vid: str) -> bool:
    return bool(_VIDEO_ID_RE.match(vid))


def _watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&autoplay=1"


def _first_video_id_from_html(html: str) -> str | None:
    seen: set[str] = set()
    for pat in _HTML_VIDEO_ID_PATTERNS:
        for m in pat.finditer(html):
            vid = m.group(1)
            if _valid_video_id(vid) and vid not in seen:
                # Skip YouTube's placeholder / ad ids that sometimes appear first.
                if vid not in ("undefined",):
                    return vid
            seen.add(vid)
    # ytInitialData JSON blob (more reliable on modern YouTube pages).
    marker = "var ytInitialData = "
    idx = html.find(marker)
    if idx >= 0:
        start = idx + len(marker)
        end = html.find(";</script>", start)
        if end > start:
            try:
                data = json.loads(html[start:end])
                vid = _video_id_from_yt_initial_data(data)
                if vid:
                    return vid
            except json.JSONDecodeError:
                pass
    return None


def _video_id_from_yt_initial_data(data: object) -> str | None:
    """Walk ytInitialData for the first watch endpoint videoId."""
    if isinstance(data, dict):
        if "videoId" in data:
            vid = data["videoId"]
            if isinstance(vid, str) and _valid_video_id(vid):
                return vid
        for v in data.values():
            found = _video_id_from_yt_initial_data(v)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _video_id_from_yt_initial_data(item)
            if found:
                return found
    return None


def _search_via_innertube(client: httpx.Client, query: str) -> str | None:
    payload = {
        "context": {"client": _INNERTUBE_CLIENT},
        "query": query,
    }
    try:
        resp = client.post(
            _INNERTUBE_SEARCH_URL,
            params={"prettyPrint": "false"},
            json=payload,
            headers={
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/json",
                "X-YouTube-Client-Name": "1",
                "X-YouTube-Client-Version": _INNERTUBE_CLIENT["clientVersion"],
            },
        )
        resp.raise_for_status()
        return _video_id_from_yt_initial_data(resp.json())
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        log.debug("Innertube search failed: %s", e)
        return None


def _search_via_html(client: httpx.Client, query: str) -> str | None:
    url = _SEARCH_URL.format(query=quote_plus(query))
    resp = client.get(url)
    resp.raise_for_status()
    return _first_video_id_from_html(resp.text)


def youtube_watch_url_for_query(query: str) -> str | None:
    """Return an autoplay watch URL for the top search hit, or None."""
    q = normalize_music_search_query(query)
    if not q:
        return None
    headers = {"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers=headers,
        ) as client:
            vid = _search_via_innertube(client, q)
            if not vid:
                vid = _search_via_html(client, q)
    except httpx.HTTPError as e:
        log.warning("YouTube search fetch failed for %r: %s", q, e)
        return None
    if not vid or not _valid_video_id(vid):
        return None
    return _watch_url(vid)


def youtube_watch_url_from_url(url: str) -> str | None:
    """If `url` is already a watch link or a YouTube search URL, return a
    direct autoplay watch URL."""
    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    if "youtu.be" in host:
        vid = parsed.path.lstrip("/").split("/")[0]
        if _valid_video_id(vid):
            return _watch_url(vid)
    if "youtube.com" not in host:
        return None
    if parsed.path in ("/watch", "/watch/"):
        qs = parse_qs(parsed.query)
        vid = (qs.get("v") or [""])[0]
        if _valid_video_id(vid):
            return _watch_url(vid)
    if "/results" in parsed.path:
        qs = parse_qs(parsed.query)
        raw_q = (qs.get("search_query") or [""])[0]
        if raw_q:
            return youtube_watch_url_for_query(unquote_plus(raw_q))
    return None
