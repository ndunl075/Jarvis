"""Tests for the capability catalog used by the Help panel and Help tab."""

from __future__ import annotations

from jarvis.tools.catalogue import local_tool_names, voice_pattern_catalogue
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


# --- linkage to the tool catalogue -----------------------------------
#
# capabilities.py stays hand-written — see its module docstring for why a
# generated catalogue was rejected. What these tests remove is the drift
# that motivated generating it: a Help card describing a capability the
# registry does not have, or a voice-reachable tool nobody documents.


# Tools deliberately absent from the Help panel, with the reason. Keep
# this list short and argued; it is the escape hatch, not the norm.
_UNDOCUMENTED_TOOLS: dict[str, str] = {
    # LLM-only utility with no voice pattern and no user-facing phrasing:
    # the model calls it while answering "what's in my downloads folder",
    # which is not itself a capability a user would look up in Help.
    "list_directory": "LLM-only helper, no voice phrasing to document",
}


def _documented_tools() -> set[str]:
    return {
        name
        for cat in CAPABILITY_CATEGORIES
        for cap in cat.capabilities
        for name in cap.tools
    }


def test_every_capability_names_only_real_tools():
    known = set(local_tool_names())
    for cat in CAPABILITY_CATEGORIES:
        for cap in cat.capabilities:
            for name in cap.tools:
                assert name in known, (
                    f"capability {cap.name!r} claims tool {name!r}, which is "
                    "not in jarvis.tools.catalogue"
                )


def test_every_tool_is_documented_by_some_capability():
    missing = set(local_tool_names()) - _documented_tools() - set(_UNDOCUMENTED_TOOLS)
    assert not missing, (
        f"tools with no Help entry: {sorted(missing)}. Add them to a "
        "Capability's `tools`, or to _UNDOCUMENTED_TOOLS with a reason."
    )


def test_undocumented_allow_list_has_no_stale_entries():
    stale = set(_UNDOCUMENTED_TOOLS) - set(local_tool_names())
    assert not stale, f"_UNDOCUMENTED_TOOLS names tools that no longer exist: {stale}"


def test_every_voice_reachable_tool_is_documented():
    """Stricter than the check above: if a user can *say* something and
    reach a tool, the phrase belongs in Help. A voice pattern with no Help
    card is a feature nobody can discover."""
    reachable = {name for name, _vp in voice_pattern_catalogue()}
    undocumented = reachable - _documented_tools()
    assert not undocumented, (
        f"tools reachable by voice but absent from Help: {sorted(undocumented)}"
    )


def test_capability_categories_and_order_are_stable():
    """The Help panel is read top to bottom and the command palette groups
    by these names, so reordering or renaming a category is a user-visible
    change that should surface as a deliberate diff here."""
    assert [c.name for c in CAPABILITY_CATEGORIES] == [
        "Getting started",
        "Apps and workspace",
        "Music and media",
        "Web and search",
        "Research",
        "Notes",
        "System",
        "Clipboard and typing",
        "Power-user tools",
    ]
    assert all(c.icon_glyph for c in CAPABILITY_CATEGORIES)
