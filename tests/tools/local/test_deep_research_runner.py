"""Tests for deep research runner (planner + worker, mocked network + LLM)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.tools.local.deep_research_runner import (
    DeepResearchConfig,
    DeepResearchPaused,
    _diversify_snippets,
    _register_sources,
    run_deep_research,
)
from jarvis.tools.local.deep_research_store import (
    CitedClaim,
    create_session,
    load_state,
)


def _cfg(**overrides) -> DeepResearchConfig:
    base = dict(
        planner_model="planner",
        worker_model="worker",
        depth=2,
        breadth=3,
        fetch_pages=False,
        max_page_chars=2000,
        enable_gap_fill=False,
    )
    base.update(overrides)
    return DeepResearchConfig(**base)


def test_run_deep_research_pauses_midway(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    state = create_session("Solar power")
    counter = {"n": 0}

    def should_pause() -> bool:
        counter["n"] += 1
        return counter["n"] > 4

    with patch(
        "jarvis.tools.local.deep_research_runner.plan_sub_questions",
        return_value=["Q1", "Q2"],
    ), patch(
        "jarvis.tools.local.deep_research_runner.expand_search_queries",
        return_value=["q-a", "q-b"],
    ), patch(
        "jarvis.tools.local.deep_research_runner.fetch_search_snippets",
        return_value=[{"title": "T", "url": "https://x.com", "content": "c"}],
    ), patch(
        "jarvis.tools.local.deep_research_runner.extract_claims",
        return_value=([CitedClaim(text="Fact 1", citations=[1])], "summary"),
    ), patch(
        "jarvis.tools.local.deep_research_runner.synthesize_executive_overview",
        return_value="Overview text",
    ):
        with pytest.raises(DeepResearchPaused):
            run_deep_research(state, cfg=_cfg(), should_pause=should_pause)

    loaded = load_state(state.id)
    assert loaded is not None
    assert loaded.status == "paused"


def test_run_deep_research_completes_with_citations(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    state = create_session("Wind energy")
    state.sub_questions = ["Sub A"]

    with patch(
        "jarvis.tools.local.deep_research_runner.expand_search_queries",
        return_value=["wind a", "wind b"],
    ), patch(
        "jarvis.tools.local.deep_research_runner.fetch_search_snippets",
        return_value=[
            {"title": "T1", "url": "https://y.com/a", "content": "body a"},
            {"title": "T2", "url": "https://z.com/b", "content": "body b"},
        ],
    ), patch(
        "jarvis.tools.local.deep_research_runner.extract_claims",
        return_value=(
            [
                CitedClaim(text="Wind is renewable", citations=[1]),
                CitedClaim(text="Capacity grows yearly", citations=[1, 2]),
            ],
            "Wind power is growing.",
        ),
    ), patch(
        "jarvis.tools.local.deep_research_runner.synthesize_executive_overview",
        return_value="Final overview with [1] citation",
    ):
        out = run_deep_research(state, cfg=_cfg(), should_pause=lambda: False)

    assert out.status == "completed"
    assert len(out.sections) == 1
    assert len(out.citations) >= 1
    section = out.sections[0]
    assert section.cited_claims
    assert section.cited_claims[0].citations == [1]


def test_run_deep_research_gap_fill_adds_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    state = create_session("EV batteries")
    state.sub_questions = ["What is the state of solid-state batteries?"]

    search_calls = {"n": 0}

    def fake_search(q, max_results=5, **kwargs):
        search_calls["n"] += 1
        if search_calls["n"] == 1:
            return [{"title": "Base", "url": "https://base.com", "content": "x"}]
        return [{"title": "Gap", "url": "https://gap.com", "content": "y"}]

    with patch(
        "jarvis.tools.local.deep_research_runner.expand_search_queries",
        return_value=["base q"],
    ), patch(
        "jarvis.tools.local.deep_research_runner.fetch_search_snippets",
        side_effect=fake_search,
    ), patch(
        "jarvis.tools.local.deep_research_runner.identify_gap_queries",
        return_value=["gap q"],
    ), patch(
        "jarvis.tools.local.deep_research_runner.extract_claims",
        return_value=(
            [CitedClaim(text="thing", citations=[1])],
            "s",
        ),
    ), patch(
        "jarvis.tools.local.deep_research_runner.synthesize_executive_overview",
        return_value="ov",
    ):
        out = run_deep_research(
            state,
            cfg=_cfg(enable_gap_fill=True),
            should_pause=lambda: False,
        )

    urls = [c["url"] for c in out.citations]
    assert "https://base.com" in urls
    assert "https://gap.com" in urls


def test_diversify_snippets_prefers_one_per_domain():
    snippets = [
        {"url": "https://example.com/a", "title": "A"},
        {"url": "https://example.com/b", "title": "B"},
        {"url": "https://other.com/x", "title": "X"},
        {"url": "https://third.com/y", "title": "Y"},
    ]
    out = _diversify_snippets(snippets, limit=3)
    domains = {url.split("/")[2] for url in (s["url"] for s in out)}
    assert "example.com" in domains
    assert "other.com" in domains
    assert "third.com" in domains
    assert len(out) == 3


def test_register_sources_dedupes_and_indexes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    state = create_session("X")
    a = [{"title": "A", "url": "https://a.com"}, {"title": "B", "url": "https://b.com"}]
    e1 = _register_sources(state, a)
    assert [s["index"] for s in e1] == [1, 2]
    # Duplicate URL keeps existing index, new one gets next index.
    b = [{"title": "A again", "url": "https://a.com"}, {"title": "C", "url": "https://c.com"}]
    e2 = _register_sources(state, b)
    assert e2[0]["index"] == 1
    assert e2[1]["index"] == 3
    assert len(state.citations) == 3
