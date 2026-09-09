"""Tests for Jina fallback in web_page_fetch."""

from __future__ import annotations

import socket
from unittest.mock import patch

import httpx

from jarvis.tools.local import web_page_fetch
from jarvis.tools.local.web_page_fetch import fetch_page_text

_PUBLIC_IP = "93.184.216.34"
_REAL_CLIENT = httpx.Client  # captured before any patching of httpx.Client


def _fake_getaddrinfo(host, port, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (_PUBLIC_IP, port or 80))]


def _make_client(handler):
    transport = httpx.MockTransport(handler)

    def make(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_CLIENT(*args, **kwargs)

    return make


def test_jina_fallback_when_direct_fetch_short():
    def handler(request):
        if "r.jina.ai" in str(request.url):
            return httpx.Response(200, headers={"content-type": "text/plain"}, text="x" * 600)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body><p>Hi</p></body></html>",
        )

    with (
        patch.object(web_page_fetch.httpx, "Client", _make_client(handler)),
        patch.object(web_page_fetch.socket, "getaddrinfo", _fake_getaddrinfo),
    ):
        text = fetch_page_text(
            "https://example.com/article",
            max_chars=1000,
            use_jina_fallback=True,
        )
    assert len(text) >= 600
