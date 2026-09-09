"""Voice tools to enable/disable Deep Research Ultra mode."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ClassVar

from jarvis.tools.registry import EmptyArgs, ToolResult, VoicePattern

log = logging.getLogger(__name__)


class EnableDeepResearchUltraTool:
    name: str = "enable_deep_research_ultra"
    description: str = (
        "Turns on Deep Research Ultra: Brave search (if API key set), Jina page "
        "reader, Groq 70B planner (if key set), and multi-pass gap-fill. "
        "Call when the user says 'enable deep research ultra', 'turn on ultra "
        "research', or 'use deep research ultra mode'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False
    # 90: ahead of DeepResearchTool (120) so "use ultra research" is a
    # mode toggle, not a research query.
    voice_patterns: ClassVar[tuple[VoicePattern, ...]] = (
        VoicePattern(
            regex=r"^(?:jarvis,?\s+)?(?:enable|turn\s+on|use)\s+"
            r"(?:(?:deep\s+research\s+)?ultra(?:\s+(?:research|mode))?"
            r"|ultra\s+research)$",
            priority=90,
        ),
    )

    def __init__(self, *, set_ultra: Callable[[bool], str]) -> None:
        self._set_ultra = set_ultra

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        message = self._set_ultra(True)
        return ToolResult(success=True, output=message)


class DisableDeepResearchUltraTool:
    name: str = "disable_deep_research_ultra"
    description: str = (
        "Turns off Deep Research Ultra and returns to standard local deep research. "
        "Call when the user says 'disable deep research ultra', 'turn off ultra "
        "research', or 'use normal deep research'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False
    voice_patterns: ClassVar[tuple[VoicePattern, ...]] = (
        VoicePattern(
            regex=r"^(?:jarvis,?\s+)?(?:disable|turn\s+off)\s+"
            r"(?:(?:deep\s+research\s+)?ultra(?:\s+(?:research|mode))?"
            r"|ultra\s+research)$",
            priority=100,
        ),
        VoicePattern(
            regex=r"^(?:jarvis,?\s+)?(?:use\s+)?normal\s+deep\s+research$",
            priority=110,
        ),
    )

    def __init__(self, *, set_ultra: Callable[[bool], str]) -> None:
        self._set_ultra = set_ultra

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        message = self._set_ultra(False)
        return ToolResult(success=True, output=message)
