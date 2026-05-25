"""Voice tools for deep research (panel + pause/resume)."""

from __future__ import annotations

import asyncio
import logging
import queue
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from jarvis.tools.registry import EmptyArgs, ToolResult

log = logging.getLogger(__name__)


class DeepResearchArgs(BaseModel):
    query: str = Field(
        description=(
            "Topic for in-depth research. From 'deep research quantum computing' "
            "use 'quantum computing'."
        )
    )


class DeepResearchTool:
    name: str = "deep_research"
    description: str = (
        "Starts in-depth web research on a topic. Saves key points to a markdown "
        "report under Jarvis data folder; shows progress in the deep research panel. "
        "User can pause and resume later. Call for 'deep research [topic]', "
        "'do deep research on [topic]', not for quick 'research [topic]'."
    )
    args_schema = DeepResearchArgs
    requires_confirmation: bool = False

    def __init__(
        self,
        *,
        on_start: Callable[[str, queue.Queue], None],
        on_speak: Callable[[str], Awaitable[None]] | None = None,
        ultra_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._on_start = on_start
        self._on_speak = on_speak
        self._ultra_enabled_fn = ultra_enabled

    async def execute(self, args: DeepResearchArgs) -> ToolResult:
        query = args.query.strip()
        if not query:
            return ToolResult(success=False, error="empty query")

        ultra = self._ultra_enabled_fn() if self._ultra_enabled_fn else False
        if self._on_speak:
            try:
                msg = (
                    "Starting deep research in Ultra mode, sir. "
                    "This may take several minutes."
                    if ultra
                    else "Starting deep research, sir. This may take several minutes."
                )
                await self._on_speak(msg)
            except Exception:
                log.debug("deep research pre-speak failed", exc_info=True)

        result_q: queue.Queue = queue.Queue(maxsize=1)
        self._on_start(query, result_q)

        try:
            result = await asyncio.to_thread(lambda: result_q.get(timeout=600.0))
        except queue.Empty:
            return ToolResult(
                success=False,
                error="deep research timed out after 10 minutes",
            )

        if isinstance(result, Exception):
            return ToolResult(success=False, error=str(result))

        message, _session_id = result
        return ToolResult(success=True, output=message)


class PauseDeepResearchTool:
    name: str = "pause_deep_research"
    description: str = (
        "Pauses the active deep research job and saves progress to disk. "
        "Call when the user says 'pause deep research' or 'pause research' "
        "while deep research is running."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, on_pause: Callable[[], None]) -> None:
        self._on_pause = on_pause

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._on_pause()
        return ToolResult(
            success=True,
            output="Pausing deep research after the current step, sir.",
        )


class ResumeDeepResearchTool:
    name: str = "resume_deep_research"
    description: str = (
        "Resumes the most recently paused deep research session. "
        "Call when the user says 'resume deep research' or 'continue deep research'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(
        self,
        *,
        on_resume_latest: Callable[[queue.Queue | None], str | None],
    ) -> None:
        self._on_resume_latest = on_resume_latest

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        result_q: queue.Queue = queue.Queue(maxsize=1)
        session_id = self._on_resume_latest(result_q)
        if session_id is None:
            return ToolResult(
                success=False,
                error="No paused deep research session found.",
            )

        try:
            result = await asyncio.to_thread(lambda: result_q.get(timeout=600.0))
        except queue.Empty:
            return ToolResult(
                success=True,
                output="Resumed deep research, sir. Check the panel for progress.",
            )

        if isinstance(result, Exception):
            return ToolResult(success=False, error=str(result))

        message, _ = result
        return ToolResult(success=True, output=message)


class CloseDeepResearchTool:
    name: str = "close_deep_research"
    description: str = (
        "Closes the deep research panel. Call when the user says "
        "'close deep research' or 'hide deep research'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, close_callback: Callable[[], None]) -> None:
        self._close = close_callback

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._close()
        return ToolResult(success=True, output="Deep research panel closed.")


class DeleteDeepResearchArgs(BaseModel):
    query: str = Field(
        default="",
        description=(
            "Topic to match against existing session queries (case-insensitive "
            "substring). Empty string deletes the currently active session."
        ),
    )


class DeleteDeepResearchTool:
    name: str = "delete_deep_research"
    description: str = (
        "Deletes one deep research session (its markdown report + saved state). "
        "Call for 'delete deep research [topic]', 'delete the [topic] research', "
        "or 'remove deep research [topic]'. Empty topic deletes the currently "
        "active session in the panel."
    )
    args_schema = DeleteDeepResearchArgs
    requires_confirmation: bool = False

    def __init__(
        self,
        *,
        delete_by_query: Callable[[str], str | None],
        delete_active: Callable[[], str | None],
    ) -> None:
        self._delete_by_query = delete_by_query
        self._delete_active = delete_active

    async def execute(self, args: DeleteDeepResearchArgs) -> ToolResult:
        topic = args.query.strip()
        if not topic:
            removed = self._delete_active()
            if removed is None:
                return ToolResult(
                    success=False,
                    error="No active deep research session to delete.",
                )
            return ToolResult(
                success=True,
                output=f"Deleted the active deep research on {removed}, sir.",
            )
        removed = self._delete_by_query(topic)
        if removed is None:
            return ToolResult(
                success=False,
                error=f"No deep research session matching {topic!r}.",
            )
        return ToolResult(
            success=True,
            output=f"Deleted deep research on {removed}, sir.",
        )


class DeleteAllDeepResearchTool:
    name: str = "delete_all_deep_research"
    description: str = (
        "Deletes every saved deep research session. Call only when the user "
        "explicitly says 'delete all deep research', 'clear all deep research', "
        "or 'wipe deep research history'."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, delete_all: Callable[[], int]) -> None:
        self._delete_all = delete_all

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        n = self._delete_all()
        if n == 0:
            return ToolResult(success=True, output="No deep research sessions to delete, sir.")
        return ToolResult(
            success=True,
            output=f"Deleted {n} deep research session{'s' if n != 1 else ''}, sir.",
        )
