"""Tests for web_page_fetch (HTML text extraction, size cap, SSRF guard).

No test here may touch the real network: every fetch goes through an
``httpx.MockTransport`` and DNS is stubbed out.
"""

from __future__ import annotations

import socket
from contextlib import contextmanager
from unittest.mock import patch

import httpx

from jarvis.tools.local import web_page_fetch
from jarvis.tools.local.web_page_fetch import _extract_main_text, fetch_page_text

_PUBLIC_IP = "93.184.216.34"
_REAL_CLIENT = httpx.Client  # captured before any patching of httpx.Client


def _client_factory(handler):
    """Return a drop-in for ``httpx.Client`` bound to a MockTransport."""
    transport = httpx.MockTransport(handler)

    def make_client(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_CLIENT(*args, **kwargs)

    return make_client


@contextmanager
def _mocked_web(handler, dns: dict[str, list[str]] | None = None):
    """Patch httpx transport + DNS. Unknown hosts resolve to a public IP.

    Only the resolver is stubbed, not ``_resolve_host_ips`` itself, so the
    IP-literal handling under test still runs.
    """
    table = dict(dns or {})
    table.setdefault("localhost", ["127.0.0.1"])

    def fake_getaddrinfo(host, port, *args, **kwargs):
        addrs = table.get(host, [_PUBLIC_IP])
        if not addrs:
            raise socket.gaierror(f"stubbed NXDOMAIN for {host}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port or 80)) for a in addrs]

    with (
        patch.object(web_page_fetch.httpx, "Client", _client_factory(handler)),
        patch.object(web_page_fetch.socket, "getaddrinfo", fake_getaddrinfo),
    ):
        yield


def _html_response(body: str, ctype: str = "text/html") -> httpx.Response:
    return httpx.Response(200, headers={"content-type": ctype}, text=body)


# ---------------------------------------------------------------------------
# Extraction (unchanged behavior)
# ---------------------------------------------------------------------------


def test_extract_main_text_strips_chrome():
    html = """
    <html><head><script>var x=1;</script><style>body{}</style></head>
    <body>
      <nav>Home About Contact</nav>
      <article>
        <h1>Solar Panels</h1>
        <p>Solar panels convert sunlight to electricity via the photovoltaic effect.</p>
        <p>Efficiency of typical commercial panels ranges from 17 to 22 percent.</p>
      </article>
      <footer>Copyright 2025</footer>
    </body></html>
    """
    text = _extract_main_text(html)
    assert "photovoltaic" in text
    assert "Copyright" not in text
    assert "Home About Contact" not in text


def test_fetch_page_text_returns_empty_for_non_http():
    assert fetch_page_text("javascript:void(0)") == ""
    assert fetch_page_text("") == ""


def test_fetch_page_text_truncates_to_max_chars():
    long_html = (
        "<html><body><article>"
        + ("Sentence number one is here. " * 1000)
        + "</article></body></html>"
    )

    with _mocked_web(lambda _req: _html_response(long_html)):
        out = fetch_page_text("https://example.com", max_chars=200)

    assert 0 < len(out) <= 220  # 200 + " …" + small slack


def test_fetch_page_text_parses_public_page():
    html = """
    <html><body>
      <nav>Home About Contact</nav>
      <article>
        <p>Photovoltaic cells convert sunlight into direct-current electricity.</p>
        <p>Commercial panel efficiency typically ranges from 17 to 22 percent.</p>
      </article>
    </body></html>
    """
    seen: list[str] = []

    def handler(request):
        seen.append(str(request.url))
        return _html_response(html)

    with _mocked_web(handler):
        out = fetch_page_text("https://example.com/solar")

    assert seen == ["https://example.com/solar"]
    assert "Photovoltaic cells" in out
    assert "Home About Contact" not in out


def test_fetch_page_text_rejects_non_text_content_type():
    with _mocked_web(lambda _req: _html_response("%PDF-1.7 binary", ctype="application/pdf")):
        assert fetch_page_text("https://example.com/paper.pdf") == ""


def test_fetch_page_text_requests_identity_encoding():
    """Compression is declined so the byte budget measures socket bytes."""
    seen: list[str] = []

    def handler(request):
        seen.append(request.headers.get("accept-encoding", ""))
        return _html_response("<html><body><p>ok</p></body></html>")

    with _mocked_web(handler):
        fetch_page_text("https://example.com")

    assert seen == ["identity"]


# ---------------------------------------------------------------------------
# Bug 1 — response body size cap
# ---------------------------------------------------------------------------


def test_oversized_body_stops_reading_at_cap():
    """The stream is abandoned at the cap instead of being buffered whole."""
    chunk = b"<p>" + b"a" * 1021 + b"</p>"  # 1 KiB per chunk
    yielded = 0

    def body():
        nonlocal yielded
        for _ in range(100_000):  # would be ~100 MB if read to the end
            yielded += 1
            yield chunk

    def handler(_request):
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body())

    # Shrink the cap so the test proves the mechanism without allocating
    # anything large.
    with (
        patch.object(web_page_fetch, "_MAX_RESPONSE_BYTES", 8 * 1024),
        _mocked_web(handler),
    ):
        out = fetch_page_text("https://example.com/huge", max_chars=500)

    assert yielded <= 9  # 8 KiB cap / 1 KiB chunks, plus the one that trips it
    assert len(out) <= 502


def test_response_byte_cap_is_sane():
    """Generous enough for any real article, small enough to bound memory."""
    assert 1_000_000 <= web_page_fetch._MAX_RESPONSE_BYTES <= 16 * 1024 * 1024


def test_declared_oversize_content_length_is_not_read():
    read_attempted = False

    def body():
        nonlocal read_attempted
        read_attempted = True
        yield b"x" * 1024

    def handler(_request):
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html",
                "content-length": str(web_page_fetch._MAX_RESPONSE_BYTES + 1),
            },
            content=body(),
        )

    with _mocked_web(handler):
        assert fetch_page_text("https://example.com/huge") == ""
    assert read_attempted is False


# ---------------------------------------------------------------------------
# Bug 2 — SSRF guard
# ---------------------------------------------------------------------------


def _refuses(url: str, dns: dict[str, list[str]] | None = None) -> bool:
    """True if the fetch returned "" without any request being issued."""
    seen: list[str] = []

    def handler(request):
        seen.append(str(request.url))
        return _html_response("<html><body><p>secret</p></body></html>")

    with _mocked_web(handler, dns=dns):
        out = fetch_page_text(url)
    return out == "" and seen == []


def test_loopback_destinations_are_refused():
    assert _refuses("http://127.0.0.1:11434/")  # the user's own Ollama
    assert _refuses("http://localhost:8080/")
    assert _refuses("http://[::1]:11434/")


def test_private_and_link_local_destinations_are_refused():
    assert _refuses("http://192.168.1.1/")  # router admin page
    assert _refuses("http://10.0.0.5/admin")
    assert _refuses("http://172.16.4.4/")
    assert _refuses("http://169.254.169.254/")  # link-local metadata address
    assert _refuses("http://[fd00::1]/")  # IPv6 unique-local
    assert _refuses("http://0.0.0.0/")


def test_hostname_resolving_to_private_ip_is_refused():
    """A public-looking name pointing at loopback is caught by resolving it."""
    assert _refuses("http://localtest.me/", dns={"localtest.me": ["127.0.0.1"]})


def test_mixed_dns_answer_is_refused():
    """One public + one loopback answer must still be refused."""
    assert _refuses("http://rebind.example/", dns={"rebind.example": [_PUBLIC_IP, "127.0.0.1"]})


def test_unresolvable_host_is_refused():
    assert _refuses("http://nope.invalid/", dns={"nope.invalid": []})


def test_redirect_to_loopback_is_refused():
    seen: list[str] = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:11434/api/tags"})
        return _html_response("<html><body><p>ollama models</p></body></html>")

    with _mocked_web(handler):
        out = fetch_page_text("https://example.com/redirect")

    assert out == ""
    assert seen == ["https://example.com/redirect"]  # the loopback hop never ran


def test_redirect_to_private_hostname_is_refused():
    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://nas.local/"})
        return _html_response("<html><body><p>private</p></body></html>")

    with _mocked_web(handler, dns={"nas.local": ["10.1.2.3"]}):
        assert fetch_page_text("https://example.com/redirect") == ""


def test_redirect_cannot_escape_http_scheme():
    def handler(request):
        if request.url.scheme in ("http", "https"):
            return httpx.Response(302, headers={"location": "file:///etc/passwd"})
        raise AssertionError("non-http scheme was requested")

    with _mocked_web(handler):
        assert fetch_page_text("https://example.com/redirect") == ""


def test_public_redirect_is_followed():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/article"})
        return _html_response(
            "<html><body><article>"
            "<p>Photovoltaic cells convert sunlight into electricity.</p>"
            "</article></body></html>"
        )

    with _mocked_web(handler):
        out = fetch_page_text("https://example.com/start")

    assert "Photovoltaic cells" in out


def test_redirect_loop_is_bounded():
    hops: list[str] = []

    def handler(request):
        hops.append(str(request.url))
        return httpx.Response(302, headers={"location": f"/hop{len(hops)}"})

    with _mocked_web(handler):
        assert fetch_page_text("https://example.com/start") == ""

    assert len(hops) <= web_page_fetch._MAX_REDIRECTS + 1


# ---------------------------------------------------------------------------
# Bug 3 — Jina must not be told about private URLs
# ---------------------------------------------------------------------------


def test_private_url_is_never_sent_to_jina():
    seen: list[str] = []

    def handler(request):
        seen.append(str(request.url))
        return _html_response("x" * 600, ctype="text/plain")

    with _mocked_web(handler):
        out = fetch_page_text("http://192.168.1.1/", max_chars=1000, use_jina_fallback=True)

    assert out == ""
    assert seen == []  # neither the direct fetch nor r.jina.ai was contacted


def test_jina_response_is_size_capped():
    def handler(request):
        if "r.jina.ai" in str(request.url):
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                content=(b"y" * 1024 for _ in range(1000)),
            )
        return _html_response("<html><body><p>Hi</p></body></html>")

    with (
        patch.object(web_page_fetch, "_MAX_RESPONSE_BYTES", 4 * 1024),
        _mocked_web(handler),
    ):
        out = fetch_page_text("https://example.com/a", max_chars=50_000, use_jina_fallback=True)

    assert 0 < len(out) <= 4 * 1024
