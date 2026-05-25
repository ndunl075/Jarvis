"""Tests for build_deep_research_config (Ultra mode)."""

from __future__ import annotations

from jarvis.core.config import ResearchConfig
from jarvis.tools.local.deep_research_runner import (
    ULTRA_DEFAULT_PLANNER,
    ULTRA_GAP_FILL_ITERATIONS,
    ULTRA_MAX_PAGE_CHARS,
    build_deep_research_config,
)


def test_standard_config_defaults():
    rcfg = ResearchConfig()
    cfg = build_deep_research_config(
        research=rcfg,
        main_llm_model="qwen2.5:7b-instruct",
    )
    assert cfg.ultra is False
    assert cfg.search_provider == "ddg"
    assert cfg.use_jina_reader is False
    assert cfg.gap_fill_max_iterations == 1


def test_ultra_without_keys_uses_jina_and_multi_gap():
    rcfg = ResearchConfig(ultra_enabled=True, max_page_chars=6000)
    cfg = build_deep_research_config(
        research=rcfg,
        main_llm_model="qwen2.5:7b-instruct",
    )
    assert cfg.ultra is True
    assert cfg.use_jina_reader is True
    assert cfg.gap_fill_max_iterations == ULTRA_GAP_FILL_ITERATIONS
    assert cfg.max_page_chars >= ULTRA_MAX_PAGE_CHARS
    assert cfg.search_provider == "ddg"
    assert cfg.planner_model == "qwen2.5:7b-instruct"


def test_ultra_with_groq_key_uses_groq_planner(monkeypatch):
    monkeypatch.setenv("JARVIS_GROQ_API_KEY", "gk-test")
    rcfg = ResearchConfig(ultra_enabled=True)
    cfg = build_deep_research_config(
        research=rcfg,
        main_llm_model="qwen2.5:7b-instruct",
    )
    assert cfg.planner_model == ULTRA_DEFAULT_PLANNER
    assert cfg.groq_api_key == "gk-test"


def test_ultra_with_brave_key_uses_brave_search(monkeypatch):
    monkeypatch.setenv("JARVIS_BRAVE_API_KEY", "bk-test")
    rcfg = ResearchConfig(ultra_enabled=True)
    cfg = build_deep_research_config(
        research=rcfg,
        main_llm_model="main",
    )
    assert cfg.search_provider == "brave"
    assert cfg.brave_api_key == "bk-test"
