"""Voice tools for the live system dashboard."""

from __future__ import annotations

from collections.abc import Callable

from jarvis.tools.registry import EmptyArgs, ToolResult


class ShowDashboardTool:
    name: str = "show_dashboard"
    description: str = (
        "Opens the live system dashboard panel showing CPU, RAM, mic level, "
        "current model, mode, foreground app, and saved notes / deep research "
        "counts. Call when the user says 'show dashboard', 'open dashboard', "
        "'show system stats', or 'how's my computer doing'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, on_open: Callable[[], None]) -> None:
        self._on_open = on_open

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._on_open()
        return ToolResult(success=True, output="Showing your dashboard, sir.")


class CloseDashboardTool:
    name: str = "close_dashboard"
    description: str = (
        "Closes the dashboard panel. Call for 'close dashboard' or 'hide dashboard'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, on_close: Callable[[], None]) -> None:
        self._on_close = on_close

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._on_close()
        return ToolResult(success=True, output="Dashboard closed.")
