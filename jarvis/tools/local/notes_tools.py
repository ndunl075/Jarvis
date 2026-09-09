"""Voice tools for the notes panel."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, Field

from jarvis.tools.registry import EmptyArgs, ToolResult, VoicePattern

log = logging.getLogger(__name__)


class TakeNoteArgs(BaseModel):
    content: str = Field(
        description=(
            "What to write down. From 'take a note about the meeting being "
            "delayed', use 'meeting delayed'. From 'jot down buy milk', use "
            "'buy milk'."
        )
    )
    title: str = Field(
        default="",
        description=(
            "Optional short title. If empty, the first 8 words of content "
            "become the title."
        ),
    )


class TakeNoteTool:
    name: str = "take_note"
    description: str = (
        "Saves a new markdown note. Call when the user says 'take a note', "
        "'jot this down', 'write this down', 'note that', or 'remember this'. "
        "Pass the actual content (NOT the trigger phrase) as 'content'."
    )
    args_schema = TakeNoteArgs
    requires_confirmation: bool = False
    voice_patterns: ClassVar[tuple[VoicePattern, ...]] = (
        VoicePattern(
            regex=r"^(?:take\s+a\s+note|jot\s+(?:this\s+)?down"
            r"|write\s+(?:this\s+)?down|note\s+that|remember\s+this)"
            r"(?:\s+(?:that|about|saying))?[:\s]+(.+)$",
            priority=440,
            args=lambda m: {"content": m.group(1)},
        ),
    )

    def __init__(self, *, on_create: Callable[[str, str], str]) -> None:
        self._on_create = on_create

    async def execute(self, args: TakeNoteArgs) -> ToolResult:
        content = args.content.strip()
        if not content:
            return ToolResult(success=False, error="empty note content")
        title = args.title.strip()
        if not title:
            words = content.split()
            title = " ".join(words[:8]) + ("…" if len(words) > 8 else "")
        try:
            stored_title = self._on_create(title, content)
        except Exception as exc:  # noqa: BLE001
            log.exception("take_note failed")
            return ToolResult(success=False, error=f"could not save note: {exc}")
        return ToolResult(
            success=True,
            output=f"Note saved, sir. Titled '{stored_title}'.",
        )


class AppendToNoteArgs(BaseModel):
    content: str = Field(description="Text to append to the note.")
    title: str = Field(
        default="",
        description=(
            "Title of the note to append to (case-insensitive substring). "
            "Empty means the currently active note in the notes panel."
        ),
    )


class AppendToNoteTool:
    name: str = "append_to_note"
    description: str = (
        "Appends text to an existing note. Call when the user says "
        "'add to my [topic] note', 'append [text] to the [topic] note', or "
        "'add this' while the notes panel shows a specific note."
    )
    args_schema = AppendToNoteArgs
    requires_confirmation: bool = False
    voice_patterns: ClassVar[tuple[VoicePattern, ...]] = (
        VoicePattern(
            regex=r"^(?:add|append)\s+(?:this\s+)?to\s+(?:my\s+|the\s+)?"
            r"(.+?)\s+note[:\s]+(.+)$",
            priority=490,
            args=lambda m: {"title": m.group(1), "content": m.group(2)},
        ),
    )

    def __init__(
        self,
        *,
        on_append_active: Callable[[str], str | None],
        on_append_by_title: Callable[[str, str], str | None],
    ) -> None:
        self._on_append_active = on_append_active
        self._on_append_by_title = on_append_by_title

    async def execute(self, args: AppendToNoteArgs) -> ToolResult:
        content = args.content.strip()
        if not content:
            return ToolResult(success=False, error="empty append content")
        title = args.title.strip()
        if not title:
            updated = self._on_append_active(content)
            if updated is None:
                return ToolResult(
                    success=False,
                    error="No active note. Open a note first or specify a title.",
                )
            return ToolResult(
                success=True,
                output=f"Added to '{updated}', sir.",
            )
        updated = self._on_append_by_title(title, content)
        if updated is None:
            return ToolResult(
                success=False,
                error=f"No note matching {title!r} found.",
            )
        return ToolResult(success=True, output=f"Added to '{updated}', sir.")


class ReadNoteArgs(BaseModel):
    title: str = Field(
        default="",
        description=(
            "Title to find (case-insensitive substring). Empty reads the "
            "currently active note."
        ),
    )


class ReadNoteTool:
    name: str = "read_note"
    description: str = (
        "Reads a note aloud. Call for 'read my notes about X', "
        "'read the [topic] note', or 'read this note' (active note)."
    )
    args_schema = ReadNoteArgs
    requires_confirmation: bool = False
    # 450 MUST precede 460: "read this note" would otherwise be captured
    # by the <title> form and read a note titled "this".
    voice_patterns: ClassVar[tuple[VoicePattern, ...]] = (
        VoicePattern(
            regex=r"^read\s+(?:this|the\s+current|my\s+current)\s+note$",
            priority=450,
            args=lambda m: {"title": ""},
        ),
        VoicePattern(
            regex=r"^read\s+(?:me\s+)?(?:my\s+|the\s+)?(.+?)\s+notes?$",
            priority=460,
            args=lambda m: {"title": m.group(1)},
        ),
    )

    def __init__(
        self,
        *,
        on_read_active: Callable[[], tuple[str, str] | None],
        on_read_by_title: Callable[[str], tuple[str, str] | None],
    ) -> None:
        self._on_read_active = on_read_active
        self._on_read_by_title = on_read_by_title

    async def execute(self, args: ReadNoteArgs) -> ToolResult:
        title = args.title.strip()
        if not title:
            result = self._on_read_active()
        else:
            result = self._on_read_by_title(title)
        if result is None:
            msg = (
                "No active note to read, sir."
                if not title
                else f"No note matching {title!r} found."
            )
            return ToolResult(success=False, error=msg)
        note_title, body = result
        spoken = body if body else f"The note '{note_title}' is empty."
        # Cap spoken length so TTS doesn't read an essay.
        if len(spoken) > 600:
            spoken = spoken[:600].rsplit(" ", 1)[0] + " — that's the first portion, sir."
        return ToolResult(success=True, output=spoken)


class OpenNotesTool:
    name: str = "open_notes"
    description: str = (
        "Opens the notes panel. Call when the user says 'open my notes', "
        "'show my notes', 'open notes', or 'show notes'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False
    # 420: must beat open_app's catch-all for "open my notes".
    voice_patterns: ClassVar[tuple[VoicePattern, ...]] = (
        VoicePattern(
            regex=r"^(?:open|show|bring\s+up)\s+(?:my\s+|the\s+)?notes$",
            priority=420,
        ),
    )

    def __init__(self, *, on_open: Callable[[], None]) -> None:
        self._on_open = on_open

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._on_open()
        return ToolResult(success=True, output="Opening your notes, sir.")


class CloseNotesTool:
    name: str = "close_notes"
    description: str = (
        "Closes the notes panel. Call for 'close notes', 'hide notes'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False
    voice_patterns: ClassVar[tuple[VoicePattern, ...]] = (
        VoicePattern(
            regex=r"^close\s+(?:my\s+|the\s+)?notes$", priority=430
        ),
    )

    def __init__(self, *, on_close: Callable[[], None]) -> None:
        self._on_close = on_close

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._on_close()
        return ToolResult(success=True, output="Notes panel closed.")


class DeleteNoteArgs(BaseModel):
    title: str = Field(
        default="",
        description="Title to match (case-insensitive substring). Empty deletes active note.",
    )


class DeleteNoteTool:
    name: str = "delete_note"
    description: str = (
        "Deletes a note. Call for 'delete the [topic] note' or 'delete this note' "
        "(active note)."
    )
    args_schema = DeleteNoteArgs
    requires_confirmation: bool = False
    # 470 MUST precede 480, for the same reason as read_note above.
    voice_patterns: ClassVar[tuple[VoicePattern, ...]] = (
        VoicePattern(
            regex=r"^delete\s+(?:this|the\s+current)\s+note$",
            priority=470,
            args=lambda m: {"title": ""},
        ),
        VoicePattern(
            regex=r"^delete\s+(?:the\s+)?(.+?)\s+note$",
            priority=480,
            args=lambda m: {"title": m.group(1)},
        ),
    )

    def __init__(
        self,
        *,
        on_delete_active: Callable[[], str | None],
        on_delete_by_title: Callable[[str], str | None],
    ) -> None:
        self._on_delete_active = on_delete_active
        self._on_delete_by_title = on_delete_by_title

    async def execute(self, args: DeleteNoteArgs) -> ToolResult:
        title = args.title.strip()
        if not title:
            removed = self._on_delete_active()
            if removed is None:
                return ToolResult(success=False, error="No active note to delete.")
            return ToolResult(success=True, output=f"Deleted note '{removed}', sir.")
        removed = self._on_delete_by_title(title)
        if removed is None:
            return ToolResult(success=False, error=f"No note matching {title!r}.")
        return ToolResult(success=True, output=f"Deleted note '{removed}', sir.")
