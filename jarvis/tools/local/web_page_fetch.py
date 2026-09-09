"""Fetch a web page and extract clean readable text (no JS).

Used by deep research to give the worker LLM full article text instead of
short DDG snippets. Pure-stdlib + httpx; no headless browser, so very fast
but skips JS-heavy sites (acceptable trade-off for local research).

The URLs handled here come from search-engine results or from the LLM, i.e.
they are influenced by third parties. Two guards follow from that:

* responses are streamed under a byte budget so a huge (hostile or merely
  broken) body cannot exhaust memory, and
* destinations that resolve to loopback / private / link-local addresses are
  refused, so the research fetcher cannot be pointed at services on the
  user's own machine or LAN (Ollama on 127.0.0.1:11434, a router admin page,
  a NAS…). Jarvis is a desktop app rather than a server, so this is a
  smaller problem than a classic server-side SSRF — but a research fetcher
  still has no business reaching the private network.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from html import unescape
from urllib.parse import urljoin, urlsplit

import httpx

log = logging.getLogger(__name__)

_TIMEOUT = 15.0
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36 Jarvis/1.0"
)

# Hard ceiling on how many response bytes we will hold in memory, per fetch.
# Sizing: callers ask for 6,000 chars (default) or 16,000 (Deep Research
# Ultra) of *extracted* text. Even a very heavy news/wiki page is ~1-2 MB of
# HTML for that much prose, so 5 MiB comfortably fits any real article while
# capping the damage from a multi-gigabyte body. Anything past the cap is
# never buffered: we stop pulling from the socket and use the prefix we have.
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024

# Redirects are followed by hand (see _fetch_guarded) so every hop can be
# validated; this bounds the chain.
_MAX_REDIRECTS = 5

# We ask for an undecoded body on purpose. httpx decompresses transparently,
# so a byte budget applied to the *decoded* stream can still be attacked with
# a decompression bomb: one small compressed chunk can inflate to hundreds of
# MB inside a single decoder step, before our accounting ever sees it. Asking
# for `identity` keeps the budget measuring what actually crosses the socket.
# The cost is bandwidth on ordinary pages, which is acceptable for the handful
# of fetches a research run makes. A server may ignore the header and gzip
# anyway; the decoded-byte budget below still stops us within roughly one
# chunk's worth of expansion in that case.
_BASE_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Encoding": "identity",
}

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

    Returns "" on any failure (404, timeout, binary content, oversized body,
    private/loopback destination, etc.). The caller is expected to fall back
    to the DDG snippet for that URL.

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


# ---------------------------------------------------------------------------
# Destination guard
# ---------------------------------------------------------------------------


def _resolve_host_ips(host: str, port: int | None) -> list[str]:
    """Return every IP ``host`` resolves to, or [] if it cannot be resolved.

    An IP literal resolves to itself. A name is resolved through
    ``socket.getaddrinfo``; *all* answers are returned because a hostile name
    can hand back one public address and one loopback address, and checking
    only the first would wave the second through.
    """
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return []
    return [info[4][0] for info in infos if info[4]]


def _is_blocked_ip(raw: str) -> bool:
    """True if ``raw`` is an address a research fetch must not reach."""
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])  # drop any zone id
    except ValueError:
        return True  # unparseable -> refuse
    # ::ffff:127.0.0.1 and friends: judge the embedded IPv4 address.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(
        ip.is_private  # RFC1918 + IPv6 unique-local (fc00::/7)
        or ip.is_loopback
        or ip.is_link_local  # 169.254.0.0/16, fe80::/10
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified  # 0.0.0.0 / :: (routes to localhost on some OSes)
    )


def _destination_allowed(url: str) -> bool:
    """True if ``url`` is http(s) and resolves only to public addresses.

    Note the honest limit: resolution here and the connect that httpx makes a
    moment later are separate events, so a name whose DNS answer flips between
    them (DNS rebinding) can still slip past. This narrows the hole a long way
    — literal private IPs, public names pointing at private space, and private
    redirect targets are all refused — but it does not close it. Closing it
    properly means pinning the checked IP into the connection, which needs a
    custom transport we do not have here.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return False
    host = parts.hostname
    if not host:
        return False
    try:
        port = parts.port
    except ValueError:
        return False
    ips = _resolve_host_ips(host, port)
    if not ips:
        return False  # fail closed: cannot verify -> do not connect
    return not any(_is_blocked_ip(ip) for ip in ips)


# ---------------------------------------------------------------------------
# Guarded, size-capped fetch
# ---------------------------------------------------------------------------


def _read_capped(response: httpx.Response) -> str:
    """Read at most ``_MAX_RESPONSE_BYTES`` of a streamed body, as text."""
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > _MAX_RESPONSE_BYTES:
        log.debug("body declares %s bytes, over cap — skipping", declared)
        return ""

    chunks: list[bytes] = []
    remaining = _MAX_RESPONSE_BYTES
    for chunk in response.iter_bytes():
        if len(chunk) >= remaining:
            # Keep only what fits and stop pulling; the socket is closed by
            # the caller's `with` block, so the rest is never transferred.
            chunks.append(chunk[:remaining])
            log.debug("response body hit the %d byte cap — truncated", _MAX_RESPONSE_BYTES)
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    try:
        return raw.decode(response.encoding or "utf-8", errors="replace")
    except LookupError:  # server declared a charset Python does not know
        return raw.decode("utf-8", errors="replace")


def _fetch_guarded(client: httpx.Client, url: str) -> tuple[str, str] | None:
    """GET ``url`` with the destination guard applied to every hop.

    Returns ``(body_text, content_type)``, or None if the fetch was refused
    or the redirect chain was too long.

    Redirects are followed manually (``follow_redirects=False`` on the client)
    rather than through an httpx event hook. With ``follow_redirects=True``
    httpx resolves the chain inside ``send()``, so the guard would have to
    live in a hook and signal refusal by raising through httpx's redirect
    loop. Doing the loop here keeps validation and the request that follows it
    adjacent, makes the hop budget explicit, and is straightforward to test.
    """
    current = url
    for _hop in range(_MAX_REDIRECTS + 1):
        if not _destination_allowed(current):
            log.debug("refusing fetch of %s: blocked destination", current)
            return None
        with client.stream("GET", current) as response:
            if response.is_redirect:
                location = response.headers.get("location", "").strip()
                if not location:
                    return None
                # Resolve relative Locations, then re-validate at the top of
                # the loop — this is what stops a redirect to loopback, and
                # what keeps a redirect from escaping to file:// or similar.
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            ctype = response.headers.get("content-type", "").lower()
            return _read_capped(response), ctype
    log.debug("refusing fetch of %s: too many redirects", url)
    return None


def _fetch_direct(url: str, *, max_chars: int) -> str:
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=False,
            headers={**_BASE_HEADERS, "Accept": "text/html"},
        ) as client:
            got = _fetch_guarded(client, url)
            if got is None:
                return ""
            html, ctype = got
            if "html" not in ctype and "text" not in ctype:
                return ""
    except Exception as exc:
        log.debug("fetch_page_text(%s) failed: %s", url, exc)
        return ""

    text = _extract_main_text(html)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"
    return text


def _fetch_jina(url: str, *, max_chars: int) -> str:
    """Jina Reader — free markdown extraction for JS-rendered pages."""
    # This path hands `url` to a third party, so check it before sending it:
    # a private or loopback URL must not leak out of the machine even as a
    # string, and Jina would happily be told about it otherwise.
    if not _destination_allowed(url):
        log.debug("refusing to send %s to Jina: blocked destination", url)
        return ""
    jina_url = _JINA_PREFIX + url
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=False,
            headers={**_BASE_HEADERS, "Accept": "text/plain"},
        ) as client:
            got = _fetch_guarded(client, jina_url)
            if got is None:
                return ""
            text = (got[0] or "").strip()
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
