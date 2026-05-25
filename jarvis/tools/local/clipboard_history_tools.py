"""Voice tools for the clipboard history panel."""

from __future__ import annotations

import logging
from collections.abc import Callable

from pydantic import BaseModel, Field

from jarvis.tools.registry import EmptyArgs, ToolResult

log = logging.getLogger(__name__)


class ShowClipboardHistoryTool:
    name: str = "show_clipboard_history"
    description: str = (
        "Opens the clipboard history panel showing recent items you've "
        "copied. Call when the user says 'show clipboard history', "
        "'open clipboard history', or 'what have I copied'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, on_open: Callable[[], None]) -> None:
        self._on_open = on_open

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._on_open()
        return ToolResult(
            success=True,
            output="Here's your clipboard history, sir.",
        )


class CloseClipboardHistoryTool:
    name: str = "close_clipboard_history"
    description: str = (
        "Closes the clipboard history panel. Call for 'close clipboard "
        "history' or 'hide clipboard history'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, on_close: Callable[[], None]) -> None:
        self._on_close = on_close

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._on_close()
        return ToolResult(success=True, output="Clipboard history closed.")


class PasteClipboardItemArgs(BaseModel):
    index: int = Field(
        default=1,
        ge=1,
        description=(
            "1-based position in the history (1 = most recent). Use 1 for "
            "'paste my last copy', 2 for 'second-to-last', etc."
        ),
    )


class PasteClipboardItemTool:
    name: str = "paste_clipboard_item"
    description: str = (
        "Loads a previous clipboard item back onto the live clipboard so the "
        "next paste (Ctrl+V) uses it. Call for 'paste item 3', 'use my "
        "second-to-last copy', or 'paste my last copy'."
    )
    args_schema = PasteClipboardItemArgs
    requires_confirmation: bool = False

    def __init__(self, *, on_paste: Callable[[int], str | None]) -> None:
        self._on_paste = on_paste

    async def execute(self, args: PasteClipboardItemArgs) -> ToolResult:
        preview = self._on_paste(args.index)
        if preview is None:
            return ToolResult(
                success=False,
                error=f"No item at position {args.index} in your clipboard history.",
            )
        return ToolResult(
            success=True,
            output=f"Item {args.index} loaded. Press Ctrl+V to paste: {preview}",
        )


class ClearClipboardHistoryTool:
    name: str = "clear_clipboard_history"
    description: str = (
        "Empties the clipboard history (pinned items survive by default). "
        "Call for 'clear my clipboard history' or 'clear clipboard history'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, on_clear: Callable[[], int]) -> None:
        self._on_clear = on_clear

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        n = self._on_clear()
        return ToolResult(
            success=True,
            output=f"Cleared {n} item(s) from your clipboard history, sir.",
        )
