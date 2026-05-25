"""LLM routing for deep research (Ollama local + Groq free tier).

Model prefixes:
  - ``groq/<model-id>`` → Groq OpenAI-compatible API (requires API key)
  - anything else → Ollama ``/api/chat`` at ``endpoint``
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from jarvis.tools.local.ollama_sync import chat_once as ollama_chat_once

log = logging.getLogger(__name__)

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_PREFIX = "groq/"


def resolve_api_key(config_value: str, env_var: str) -> str:
    """Prefer environment variable over persisted config value."""
    env = os.environ.get(env_var, "").strip()
    if env:
        return env
    return (config_value or "").strip()


def is_groq_model(model: str) -> bool:
    return model.strip().lower().startswith(_GROQ_PREFIX)


def groq_model_id(model: str) -> str:
    """Strip ``groq/`` prefix for the Groq API model field."""
    m = model.strip()
    if m.lower().startswith(_GROQ_PREFIX):
        return m[len(_GROQ_PREFIX) :]
    return m


def chat_once(
    *,
    endpoint: str,
    model: str,
    user_content: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
    groq_api_key: str | None = None,
) -> str:
    """Single non-streaming chat turn via Ollama or Groq."""
    if is_groq_model(model):
        key = (groq_api_key or "").strip()
        if not key:
            raise ValueError(
                "Groq API key required for ultra planner (set JARVIS_GROQ_API_KEY "
                "or research.groq_api_key in config)"
            )
        return _groq_chat_once(
            api_key=key,
            model=groq_model_id(model),
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return ollama_chat_once(
        endpoint=endpoint,
        model=model,
        user_content=user_content,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _groq_chat_once(
    *,
    api_key: str,
    model: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(_GROQ_CHAT_URL, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()
