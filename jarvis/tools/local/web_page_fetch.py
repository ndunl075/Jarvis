"""Fetch a web page and extract clean readable text (no JS).

Used by deep research to give the worker LLM full article text instead of
short DDG snippets. Pure-stdlib + httpx; no headless browser, so very fast
but skips JS-heavy sites (acceptable trade-off for local research).
"""

from __future__ import annotations

import logging
import re
from html import unescape

import httpx

log = logging.getLogger(__name__)

_TIMEOUT = 15.0
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36 Jarvis/1.0"
)

# Tags whose body text is useless / harmful for summarization.
_DROP_BLOCK = re.compile(
    r"<(script|style|nav|footer|header|aside|form|noscript|iframe|svg)\b[^>]*>"
    r".*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_INLINE_LINK = re.compile(r"\[(?:\d+|[a-z])\]", re.IGNORECASE)


_JINA_PREFIX = "https://r.jina.ai/"
_JINA_MIN_USEFUL_CHARS = 500


def fetch_page_text(
    url: str,
    *,
    max_chars: int = 6000,
    use_jina_fallback: bool = False,
) -> str:
    """Return cleaned plain text from a URL, truncated to ``max_chars``.

    Returns "" on any failure (404, timeout, binary content, etc.). The
    caller is expected to fall back to the DDG snippet for that URL.

    When ``use_jina_fallback`` is True (Deep Research Ultra), retries via
    Jina Reader if direct fetch yields little or no text (JS-heavy sites).
    """
    if not url.startswith(("http://", "https://")):
        return ""

    text = _fetch_direct(url, max_chars=max_chars)
    if len(text) >= _JINA_MIN_USEFUL_CHARS or not use_jina_fallback:
        return text

    jina_text = _fetch_jina(url, max_chars=max_chars)
    if len(jina_text) > len(text):
        return jina_text
    return text


def _fetch_direct(url: str, *, max_chars: int) -> str:
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html"},
        ) as client:
            r = client.get(url)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "").lower()
            if "html" not in ctype and "text" not in ctype:
                return ""
            html = r.text
    except Exception as exc:
        log.debug("fetch_page_text(%s) failed: %s", url, exc)
        return ""

    text = _extract_main_text(html)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"
    return text


def _fetch_jina(url: str, *, max_chars: int) -> str:
    """Jina Reader — free markdown extraction for JS-rendered pages."""
    jina_url = _JINA_PREFIX + url
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/plain"},
        ) as client:
            r = client.get(jina_url)
            r.raise_for_status()
            text = (r.text or "").strip()
    except Exception as exc:
        log.debug("jina fetch(%s) failed: %s", url, exc)
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"
    return text


def _extract_main_text(html: str) -> str:
    # Strip blocks that never contribute to article content.
    cleaned = _DROP_BLOCK.sub(" ", html)

    # Prefer <article>, <main>, or large prose containers if present —
    # this drops nav chrome and most ad text.
    body = _largest_article_body(cleaned) or cleaned

    text = _TAG.sub("\n", body)
    text = unescape(text)
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    text = _INLINE_LINK.sub("", text)

    lines = [ln.strip() for ln in text.splitlines()]
    # Drop short navigation-y lines (cookies, share, etc.).
    lines = [ln for ln in lines if len(ln) >= 40 or ln.endswith((".", "?", "!"))]
    return "\n".join(lines).strip()


_ARTICLE = re.compile(r"<(article|main)\b[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)


def _largest_article_body(html: str) -> str | None:
    matches = _ARTICLE.findall(html)
    if not matches:
        return None
    return max((body for _tag, body in matches), key=len)
