"""Research tool — triggers ResearchPanel's QThread worker and awaits result.

Uses DuckDuckGo for web snippets (no API key) and local Ollama for the
summary. Requires Ollama to be running and a network connection for search.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from jarvis.tools.registry import EmptyArgs, ToolResult

log = logging.getLogger(__name__)


class ResearchArgs(BaseModel):
    query: str = Field(
        description=(
            "The topic or question to research. Extract the core subject, e.g. "
            "'quantum computing' from 'research quantum computing'."
        )
    )


def _first_n_sentences(text: str, n: int) -> str:
    """Return the first n sentences from text, stripping bullet markers."""
    text = text.replace("•", "").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in parts if s.strip()]
    return " ".join(sentences[:n])


class ResearchTool:
    name: str = "research"
    description: str = (
        "Searches the web and shows a research panel with a summary and sources. "
        "Uses local Ollama to summarize DuckDuckGo results — no cloud LLM API key. "
        "Call when the user says 'research [topic]', 'look up [topic]', or asks "
        "you to find information about something. "
        "The result is shown visually AND the first two sentences are spoken back."
    )
    args_schema = ResearchArgs
    requires_confirmation: bool = False

    def __init__(
        self,
        *,
        on_start: Callable[[str, queue.Queue], None],
        on_speak: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._on_start = on_start
        self._on_speak = on_speak

    async def execute(self, args: ResearchArgs) -> ToolResult:
        query = args.query.strip()
        if not query:
            return ToolResult(success=False, error="empty query")

        if self._on_speak:
            try:
                await self._on_speak("Searching now, sir.")
            except Exception:
                log.debug("pre-search speak failed", exc_info=True)

        result_q: queue.Queue = queue.Queue(maxsize=1)
        self._on_start(query, result_q)

        try:
            result = await asyncio.to_thread(lambda: result_q.get(timeout=90.0))
        except queue.Empty:
            return ToolResult(
                success=False,
                error="research timed out after 90 seconds",
            )

        if isinstance(result, Exception):
            return ToolResult(success=False, error=str(result))

        summary, _sources = result
        spoken = _first_n_sentences(summary, 2)
        return ToolResult(
            success=True,
            output=spoken or f"Research on '{query}' complete.",
        )


class CloseResearchTool:
    name: str = "close_research"
    description: str = (
        "Closes the research panel. Call when the user says 'close research', "
        "'dismiss', 'close panel', 'hide results', or similar phrases."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, close_callback: Callable[[], None]) -> None:
        self._close = close_callback

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._close()
        return ToolResult(success=True, output="Research panel closed.")


class ReadMoreTool:
    name: str = "read_more"
    description: str = (
        "Reads the next two sentences of the current research summary aloud. "
        "Call when the user says 'read more', 'continue', or 'keep going'. "
        "Only useful while the research panel is open."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, get_next: Callable[[], str | None]) -> None:
        self._get_next = get_next

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        text = self._get_next()
        if text is None:
            return ToolResult(
                success=True,
                output="That is the end of the summary, sir.",
            )
        return ToolResult(success=True, output=text)


class CopyResearchTool:
    name: str = "copy_research"
    description: str = (
        "Copies the research summary to clipboard. "
        "Call when the user says 'copy that', 'copy the summary', or similar. "
        "Only useful while the research panel is open."
    )
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, *, copy_callback: Callable[[], None]) -> None:
        self._copy = copy_callback

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        self._copy()
        return ToolResult(success=True, output="Copied to your clipboard, sir.")
