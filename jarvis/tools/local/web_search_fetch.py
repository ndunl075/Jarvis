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
    for href, title, content in _iter_result_pairs(html):
        if not title and not content:
            continue
        url = _resolve_ddg_redirect(href)
        results.append({
            "title": title or url,
            "url": url,
            "content": content or title,
        })
        if len(results) >= max_results:
            break

    return results


# Each result block is an <a class="result__a"> (title + outbound href)
# followed by an <a class="result__snippet"> (the description).
#
# The title group is `.*?` rather than `[^<]+` on purpose: DuckDuckGo
# highlights query terms inside the title with <b>, and `[^<]+` cannot match
# across that, so the anchor did not match AT ALL and the whole result
# vanished. _strip_tags below is what removes the markup — it was already
# being called, but could never see a tag.
_LINK_RE = re.compile(
    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def _iter_result_pairs(html: str) -> list[tuple[str, str, str]]:
    """Return (href, title, snippet) triples in document order.

    Links and snippets are walked as one position-ordered stream and each
    snippet is attached to the link it follows, rather than collecting two
    independent lists and pairing them by index.

    The index pairing was silently wrong whenever the two lists came out
    different lengths: one unmatched title shifted every later snippet up by
    one, so a result kept its own title and URL but carried the NEXT site's
    description. Those pairs feed the research summarizer with numbered
    citations, so a mismatch there attributes text to a source it did not
    come from. Position pairing cannot drift that way — a link with no
    snippet simply yields an empty one.
    """
    events: list[tuple[int, bool, re.Match[str]]] = []
    events += [(m.start(), True, m) for m in _LINK_RE.finditer(html)]
    events += [(m.start(), False, m) for m in _SNIPPET_RE.finditer(html)]
    events.sort(key=lambda e: e[0])

    pending: list[list[str]] = []
    for _pos, is_link, m in events:
        if is_link:
            pending.append([
                unescape(m.group(1)),
                _strip_tags(unescape(m.group(2))),
                "",
            ])
        elif pending and not pending[-1][2]:
            # Snippets belong to the most recent link. A stray snippet before
            # any link (or a second one for the same link) is ignored rather
            # than shifting everything after it.
            pending[-1][2] = _strip_tags(unescape(m.group(1)))

    return [(href, title, snippet) for href, title, snippet in pending]


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
