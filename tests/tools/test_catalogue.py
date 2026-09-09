"""Tests for the static local-tool catalogue.

The catalogue is what stops "add a tool" from meaning "remember to also
edit the router's pattern table and the settings tab's checkbox list".
These tests hold that promise up.
"""

from __future__ import annotations

import re

import pytest

from jarvis.tools.catalogue import (
    local_tool_classes,
    local_tool_names,
    voice_pattern_catalogue,
)
from jarvis.tools.registry import (
    TOOL_NAME_REGEX,
    Tool,
    VoicePattern,
    VoiceRoutable,
)


def test_every_catalogue_class_satisfies_the_tool_contract():
    """Checked at the class level, which is how the catalogue consumes
    these. `description` is deliberately not asserted here:
    LaunchWorkspaceTool builds its description from the configured
    workspace apps, so it only exists on the instance."""
    for cls in local_tool_classes():
        assert isinstance(cls.name, str) and cls.name
        assert TOOL_NAME_REGEX.match(cls.name), cls
        assert hasattr(cls, "args_schema"), cls
        assert callable(cls.execute), cls


def test_tool_names_are_unique():
    names = [cls.name for cls in local_tool_classes()]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate tool names in the catalogue: {sorted(dupes)}"


def test_local_tool_names_is_sorted_and_matches_the_classes():
    names = local_tool_names()
    assert list(names) == sorted(names)
    assert set(names) == {cls.name for cls in local_tool_classes()}


# --- voice patterns --------------------------------------------------


def test_catalogue_is_sorted_by_priority():
    priorities = [vp.priority for _name, vp in voice_pattern_catalogue()]
    assert priorities == sorted(priorities)


def test_every_pattern_priority_is_unique():
    """Not strictly required — the sort has a deterministic tiebreak — but
    a duplicate priority means two patterns' relative order was decided by
    a tiebreak rather than by a person, which for a first-match-wins table
    is exactly the situation this refactor exists to prevent."""
    priorities = [vp.priority for _name, vp in voice_pattern_catalogue()]
    dupes = {p for p in priorities if priorities.count(p) > 1}
    assert not dupes, f"patterns sharing a priority: {sorted(dupes)}"


def test_every_pattern_compiles_and_is_start_anchored():
    """`_try_pattern` uses re.match, so a pattern without `^` still only
    matches at the start — but writing it explicitly is what makes the
    precedence comments readable, and an unanchored pattern is usually a
    sign someone meant `search`."""
    for name, vp in voice_pattern_catalogue():
        re.compile(vp.regex)  # raises on a bad regex
        assert vp.regex.startswith("^"), f"{name}: pattern is not anchored"


def test_every_pattern_regex_is_lowercase():
    """Patterns are matched against the normalized (lowercased)
    transcription, so an uppercase literal can never match."""
    for name, vp in voice_pattern_catalogue():
        literals = re.sub(r"\\[a-zA-Z]|\(\?:|[\\^$(){}\[\]|?*+.,\d-]", "", vp.regex)
        assert literals == literals.lower(), f"{name}: {vp.regex!r} has uppercase"


def test_every_pattern_names_a_catalogue_tool():
    known = set(local_tool_names())
    for name, _vp in voice_pattern_catalogue():
        assert name in known


def test_declared_patterns_are_real_voice_pattern_instances():
    declaring = [c for c in local_tool_classes() if getattr(c, "voice_patterns", ())]
    assert declaring, "no tool declares voice patterns — catalogue is broken"
    for cls in declaring:
        assert isinstance(cls.voice_patterns, tuple), cls
        assert all(isinstance(p, VoicePattern) for p in cls.voice_patterns), cls


def test_args_builder_returns_a_dict_for_a_matching_utterance():
    """Guards against an args lambda that captures a group the regex does
    not have, which would only surface at dispatch time."""
    samples = {
        "^volume\\s+(up|down|mute|unmute)$": "volume up",
        "^paste\\s+item\\s+(\\d{1,2})$": "paste item 3",
        "^open\\s+(.+)$": "open spotify",
        "^research\\s+(.+)$": "research black holes",
    }
    checked = 0
    for _name, vp in voice_pattern_catalogue():
        utterance = samples.get(vp.regex)
        if utterance is None:
            continue
        m = re.compile(vp.regex).match(utterance)
        assert m is not None
        assert isinstance(vp.args(m), dict)
        checked += 1
    assert checked == len(samples)


# --- MCP tools stay valid --------------------------------------------


class _FakeMCPTool:
    """Structural stand-in for an MCP-adapted tool: satisfies `Tool`,
    declares no voice patterns, no help entry."""

    name = "fs_read_file"
    description = "read a file"
    requires_confirmation = False

    def __init__(self) -> None:
        from jarvis.tools.registry import EmptyArgs

        self.args_schema = EmptyArgs

    async def execute(self, args):  # noqa: ANN001, ARG002
        from jarvis.tools.registry import ToolResult

        return ToolResult(success=True)


def test_mcp_shaped_tool_still_satisfies_the_tool_protocol():
    """The point of keeping VoicePattern out of `Tool`: isinstance()
    against a runtime_checkable Protocol is a hasattr() sweep over every
    declared member, so adding voice_patterns to `Tool` would have made
    every MCP tool stop being a `Tool`."""
    assert isinstance(_FakeMCPTool(), Tool)


def test_mcp_shaped_tool_is_not_voice_routable():
    assert not isinstance(_FakeMCPTool(), VoiceRoutable)


def test_a_tool_declaring_patterns_is_voice_routable():
    class _Routable(_FakeMCPTool):
        name = "routable"
        voice_patterns = (VoicePattern(regex=r"^ping$", priority=1),)

    assert isinstance(_Routable(), VoiceRoutable)
    assert isinstance(_Routable(), Tool)


@pytest.mark.parametrize("cls", list(local_tool_classes()))
def test_catalogue_classes_are_importable_and_named(cls):
    assert isinstance(cls.name, str)


# --- the settings tab is driven from here ----------------------------


def test_settings_tools_tab_shows_every_catalogue_tool():
    """The Settings → Tools checkbox list used to be a hand-maintained
    tuple that could silently omit a tool (leaving it permanently
    un-disableable in the UI). It is derived now; this asserts the split
    across the three group boxes stays a partition of the catalogue."""
    from jarvis.ui.settings.tabs.tools import (
        _DEEP_RESEARCH_TOOL_NAMES,
        _LOCAL_TOOL_NAMES,
        _RESEARCH_TOOL_NAMES,
    )

    shown = list(_LOCAL_TOOL_NAMES) + list(_RESEARCH_TOOL_NAMES) + list(
        _DEEP_RESEARCH_TOOL_NAMES
    )
    assert len(shown) == len(set(shown)), "a tool appears in two group boxes"
    assert set(shown) == set(local_tool_names())


def test_settings_research_group_names_are_real_tools():
    """The two research group boxes are still hand-listed (which tools sit
    under which explanatory blurb is a layout choice). This stops one of
    them naming a tool that no longer exists."""
    from jarvis.ui.settings.tabs.tools import (
        _DEEP_RESEARCH_TOOL_NAMES,
        _RESEARCH_TOOL_NAMES,
    )

    known = set(local_tool_names())
    for name in (*_RESEARCH_TOOL_NAMES, *_DEEP_RESEARCH_TOOL_NAMES):
        assert name in known, f"settings tab lists unknown tool {name!r}"
