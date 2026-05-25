"""Tests for research_llm (Groq routing + key resolution)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.tools.local.research_llm import (
    chat_once,
    groq_model_id,
    is_groq_model,
    resolve_api_key,
)


def test_is_groq_model():
    assert is_groq_model("groq/llama-3.3-70b-versatile")
    assert not is_groq_model("qwen2.5:7b-instruct")


def test_groq_model_id_strips_prefix():
    assert groq_model_id("groq/llama-3.3-70b-versatile") == "llama-3.3-70b-versatile"


def test_resolve_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("JARVIS_GROQ_API_KEY", "from-env")
    assert resolve_api_key("from-file", "JARVIS_GROQ_API_KEY") == "from-env"


def test_chat_once_groq_requires_key():
    with pytest.raises(ValueError, match="Groq API key"):
        chat_once(
            endpoint="http://localhost:11434",
            model="groq/llama-3.3-70b-versatile",
            user_content="hi",
            groq_api_key=None,
        )


def test_chat_once_groq_calls_api():
    with patch(
        "jarvis.tools.local.research_llm._groq_chat_once",
        return_value="answer",
    ) as groq:
        out = chat_once(
            endpoint="http://localhost:11434",
            model="groq/llama-3.3-70b-versatile",
            user_content="hi",
            groq_api_key="sk-test",
        )
    assert out == "answer"
    groq.assert_called_once()


def test_chat_once_ollama_delegates():
    with patch(
        "jarvis.tools.local.research_llm.ollama_chat_once",
        return_value="local",
    ) as ollama:
        out = chat_once(
            endpoint="http://localhost:11434",
            model="qwen2.5:7b-instruct",
            user_content="hi",
        )
    assert out == "local"
    ollama.assert_called_once()
