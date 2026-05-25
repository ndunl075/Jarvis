"""Voice tools for the Help / capabilities panel."""

from __future__ import annotations

from collections.abc import Callable

from jarvis.tools.registry import EmptyArgs, ToolResult


class OpenHelpTool:
    name: str = "open_help"
    description: str = (
        "Opens the 'What can I say?' help panel — a searchable, plain-English "
        "list of every voice command Jarvis understands, grouped by topic. "
        "Call when the user says 'what can you do', 'show help', 'open help', "
        "'show capabilities', or 'what can I say'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, on_open: Callable[[], None]) -> None:
        self._on_open = on_open

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._on_open()
        return ToolResult(
            success=True,
            output=(
                "Here's everything I can do, sir. Use the search box to "
                "narrow it down."
            ),
        )
