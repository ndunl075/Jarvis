"""Tests for deep research persistence."""

from __future__ import annotations

from jarvis.tools.local.deep_research_store import (
    CitedClaim,
    DeepResearchSection,
    create_session,
    delete_all_sessions,
    delete_session,
    find_session_by_query,
    format_references_markdown,
    format_section_markdown,
    latest_paused_session,
    list_sessions,
    load_state,
    read_report,
    save_state,
)


def test_create_session_writes_report_and_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    state = create_session("Quantum dots")
    assert state.query == "Quantum dots"
    assert state.status == "running"
    assert "Quantum dots" in read_report(state.id)
    loaded = load_state(state.id)
    assert loaded is not None
    assert loaded.id == state.id


def test_save_state_updates_status_in_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    state = create_session("AI safety")
    state.status = "paused"
    save_state(state)
    assert "paused" in read_report(state.id)


def test_list_and_latest_paused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    s1 = create_session("Topic A")
    s1.status = "paused"
    save_state(s1)
    create_session("Topic B")
    sessions = list_sessions()
    assert len(sessions) >= 2
    latest = latest_paused_session()
    assert latest is not None
    assert latest.status == "paused"


def test_format_section_markdown_cited_claims():
    md = format_section_markdown(
        DeepResearchSection(
            sub_question="How does it work?",
            summary="Short summary [1].",
            cited_claims=[
                CitedClaim(text="Point one", citations=[1]),
                CitedClaim(text="Point two", citations=[2, 3]),
            ],
            queries_used=["alpha", "beta"],
        ),
        1,
    )
    assert "How does it work?" in md
    assert "Point one [1]" in md
    assert "Point two [2] [3]" in md
    assert "Short summary" in md
    assert "alpha" in md and "beta" in md


def test_format_section_markdown_legacy_key_points():
    md = format_section_markdown(
        DeepResearchSection(
            sub_question="Legacy section",
            key_points=["bullet"],
        ),
        2,
    )
    assert "Legacy section" in md
    assert "bullet" in md


def test_delete_session_removes_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    state = create_session("To delete")
    assert load_state(state.id) is not None
    assert delete_session(state.id) is True
    assert load_state(state.id) is None


def test_delete_session_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    assert delete_session("../evil") is False
    assert delete_session("a/b") is False
    assert delete_session("") is False


def test_delete_all_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    create_session("A")
    create_session("B")
    create_session("C")
    n = delete_all_sessions()
    assert n == 3
    assert list_sessions() == []


def test_find_session_by_query_substring(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.deep_research_store.deep_research_root",
        lambda: tmp_path,
    )
    create_session("Quantum computing breakthroughs")
    create_session("Wind energy")
    found = find_session_by_query("quantum")
    assert found is not None
    assert "Quantum" in found.query
    assert find_session_by_query("nonexistent") is None
    assert find_session_by_query("") is None


def test_format_references_markdown():
    md = format_references_markdown([
        {"title": "Example", "url": "https://example.com"},
        {"title": "Other", "url": "https://other.com"},
    ])
    assert "References" in md
    assert "1. [Example](https://example.com)" in md
    assert "2. [Other](https://other.com)" in md
