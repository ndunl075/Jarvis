"""Fetch web search snippets without a paid API key.

Uses DuckDuckGo's HTML endpoint (no account required). Results are passed
to the local Ollama model for summarization in the research panel.
"""

from __future__ import annotations

import logging
import re
from html import unescape
from urllib.parse import parse_qs, unquote, urlparse

import httpx

log = logging.getLogger(__name__)

_TIMEOUT = 12.0
_USER_AGENT = "Jarvis/1.0 (local research)"
# DuckDuckGo HTML results page (no API key).
_DDG_HTML = "https://html.duckduckgo.com/html/"


def fetch_search_snippets(query: str, *, max_results: int = 5) -> list[dict[str, str]]:
    """Return [{title, url, content}, ...] for a search query.

    Raises httpx.HTTPError on network failure. Returns an empty list when
    DuckDuckGo returns no parseable hits (caller should surface a friendly error).
    """
    q = query.strip()
    if not q:
        return []

    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = client.post(_DDG_HTML, data={"q": q})
        response.raise_for_status()
        html = response.text

    results: list[dict[str, str]] = []
    # Each result block: <a class="result__a" href="...">title</a> ... snippet
    link_pat = re.compile(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
        re.IGNORECASE,
    )
    snippet_pat = re.compile(
        r'class="result__snippet"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    links = list(link_pat.finditer(html))
    snippets = list(snippet_pat.finditer(html))

    for i, m in enumerate(links[:max_results]):
        href = unescape(m.group(1))
        title = _strip_tags(unescape(m.group(2))).strip()
        url = _resolve_ddg_redirect(href)
        content = ""
        if i < len(snippets):
            content = _strip_tags(unescape(snippets[i].group(1))).strip()
        if not title and not content:
            continue
        results.append({
            "title": title or url,
            "url": url,
            "content": content or title,
        })

    return results


def _resolve_ddg_redirect(href: str) -> str:
    """DuckDuckGo HTML wraps outbound links in //duckduckgo.com/l/?uddg=..."""
    if "uddg=" in href:
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg", [""])[0]
        if uddg:
            return unquote(uddg)
    if href.startswith("//"):
        return "https:" + href
    return href


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def format_snippets_for_prompt(snippets: list[dict[str, str]]) -> str:
    """Build a compact context block for the summarization prompt."""
    parts: list[str] = []
    for i, s in enumerate(snippets, 1):
        parts.append(
            f"[{i}] {s.get('title', '')}\n"
            f"URL: {s.get('url', '')}\n"
            f"{s.get('content', '')}"
        )
    return "\n\n".join(parts)
