"""Synchronous Ollama HTTP helpers for QThread workers (research panel).

The main app uses async OllamaClient on the audio loop; Qt research workers
run on QThreads and call these blocking helpers instead.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx

log = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:11434"


def stream_chat(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    on_chunk: Callable[[str], None],
    max_tokens: int = 1024,
    temperature: float = 0.4,
) -> str:
    """Stream a chat completion; invoke on_chunk per token. Returns full text."""
    url = endpoint.rstrip("/") + "/api/chat"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    parts: list[str] = []
    timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message") or {}
                chunk = msg.get("content") or ""
                if chunk:
                    parts.append(chunk)
                    on_chunk(chunk)
                if data.get("done"):
                    break
    return "".join(parts)


def chat_once(
    *,
    endpoint: str,
    model: str,
    user_content: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
) -> str:
    """Single non-streaming chat turn; returns assistant text."""
    url = endpoint.rstrip("/") + "/api/chat"
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=body)
        response.raise_for_status()
        data = response.json()
    return (data.get("message") or {}).get("content") or ""
