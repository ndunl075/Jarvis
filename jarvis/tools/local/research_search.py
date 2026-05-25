"""Web search for deep research (DuckDuckGo default; Brave when ultra + key)."""

from __future__ import annotations

import logging
from typing import Literal

import httpx

from jarvis.tools.local.web_search_fetch import fetch_search_snippets as fetch_ddg

log = logging.getLogger(__name__)

SearchProvider = Literal["ddg", "brave"]

_BRAVE_WEB = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_TIMEOUT = 12.0
_USER_AGENT = "Jarvis/1.0 (deep research ultra)"


def fetch_search_snippets(
    query: str,
    *,
    max_results: int = 5,
    provider: SearchProvider = "ddg",
    brave_api_key: str | None = None,
) -> list[dict[str, str]]:
    """Return ``[{title, url, content}, ...]`` for a search query."""
    q = query.strip()
    if not q:
        return []

    if provider == "brave" and brave_api_key:
        try:
            hits = _fetch_brave(q, max_results=max_results, api_key=brave_api_key)
            if hits:
                return hits
        except Exception as exc:
            log.warning("Brave search failed for %r, falling back to DDG: %s", q, exc)

    return fetch_ddg(q, max_results=max_results)


def _fetch_brave(query: str, *, max_results: int, api_key: str) -> list[dict[str, str]]:
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": max_results}
    with httpx.Client(
        timeout=_BRAVE_TIMEOUT,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = client.get(_BRAVE_WEB, params=params)
        response.raise_for_status()
        data = response.json()

    results: list[dict[str, str]] = []
    web = data.get("web") or {}
    for item in (web.get("results") or [])[:max_results]:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        desc = (item.get("description") or "").strip()
        if not url:
            continue
        results.append({
            "title": title or url,
            "url": url,
            "content": desc or title,
        })
    return results
