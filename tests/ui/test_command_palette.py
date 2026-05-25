"""Tests for the command palette's pure fuzzy ranking + entry building."""

from __future__ import annotations

from jarvis.ui.command_palette import _build_entries, _fuzzy_score


def test_build_entries_covers_every_capability_example():
    from jarvis.ui.capabilities import CAPABILITY_CATEGORIES

    expected = sum(
        len(cap.examples) for cat in CAPABILITY_CATEGORIES for cap in cat.capabilities
    )
    entries = _build_entries()
    assert len(entries) == expected
    # Spot-check a known entry.
    phrases = [e.phrase for e in entries]
    assert any("play some jazz" in p.lower() for p in phrases)


def test_fuzzy_score_substring_outranks_subsequence():
    sub = _fuzzy_score("dash", "show dashboard")
    seq = _fuzzy_score("dash", "drink a smash")
    assert sub > seq > 0


def test_fuzzy_score_no_match():
    assert _fuzzy_score("xyzqq", "show dashboard") == 0


def test_fuzzy_score_empty_query_matches_everything():
    assert _fuzzy_score("", "anything") > 0


def test_fuzzy_score_case_insensitive():
    a = _fuzzy_score("DASH", "show dashboard")
    b = _fuzzy_score("dash", "show dashboard")
    assert a == b


def test_earlier_substring_position_scores_higher():
    early = _fuzzy_score("note", "note taking")
    late = _fuzzy_score("note", "open the note panel later")
    assert early > late
