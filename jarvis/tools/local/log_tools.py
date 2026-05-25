"""Voice tools for the live log viewer panel."""

from __future__ import annotations

from collections.abc import Callable

from jarvis.tools.registry import EmptyArgs, ToolResult


class ShowLogsTool:
    name: str = "show_logs"
    description: str = (
        "Opens the live log viewer panel showing recent entries from "
        "jarvis.log with a level filter and search. Call for 'show logs', "
        "'open logs', 'show me the log', or 'show errors'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, on_open: Callable[[], None]) -> None:
        self._on_open = on_open

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._on_open()
        return ToolResult(success=True, output="Showing logs, sir.")


class CloseLogsTool:
    name: str = "close_logs"
    description: str = (
        "Closes the log viewer panel. Call for 'close logs' or 'hide logs'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, on_close: Callable[[], None]) -> None:
        self._on_close = on_close

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._on_close()
        return ToolResult(success=True, output="Log viewer closed.")
