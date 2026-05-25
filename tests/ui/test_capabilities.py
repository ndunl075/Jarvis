"""Tests for the capability catalog used by the Help panel and Help tab."""

from __future__ import annotations

from jarvis.ui.capabilities import (
    CAPABILITY_CATEGORIES,
    all_capabilities,
    search_capabilities,
)


def test_categories_non_empty_and_each_has_examples():
    assert CAPABILITY_CATEGORIES
    for cat in CAPABILITY_CATEGORIES:
        assert cat.capabilities
        for cap in cat.capabilities:
            assert cap.name
            assert cap.description
            assert cap.examples
            for ex in cap.examples:
                assert ex.strip()


def test_all_capabilities_flat_count_matches():
    flat = all_capabilities()
    total = sum(len(c.capabilities) for c in CAPABILITY_CATEGORIES)
    assert len(flat) == total


def test_search_substring_match():
    results = search_capabilities("note")
    assert results
    names = [cap.name.lower() for _cat, cap in results]
    assert any("note" in n for n in names)


def test_search_empty_returns_everything():
    full = search_capabilities("")
    flat = all_capabilities()
    assert len(full) == len(flat)


def test_search_case_insensitive():
    a = search_capabilities("WEATHER")
    b = search_capabilities("weather")
    assert len(a) == len(b)
    assert a == b


def test_search_no_match():
    assert search_capabilities("zzz-not-a-real-thing") == []
