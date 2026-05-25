"""Read-only filesystem helpers.

This is the Phase 4 file surface: list_directory ONLY. Read, write, move,
and delete are intentionally absent — they're the destructive class of
operations that needs the requires_confirmation UX (deferred to Phase
6+) before they ship. Until then, the LLM has visibility into the
filesystem but no ability to modify it through tools."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from jarvis.tools.registry import ToolResult

log = logging.getLogger(__name__)

# How many entry names get spoken back. Voice playback of a full file
# listing is unusable — five names plus a count is the whole UX a user
# can absorb in a single TTS pass. The screen / settings view is the
# place for the full list; this tool is voice-only.
_SPOKEN_PREVIEW_COUNT = 5


class ListDirectoryArgs(BaseModel):
    path: str = Field(
        description=(
            "Directory to list. Accepts ~ for the user home. "
            "Relative paths resolve against the user home."
        ),
    )


class ListDirectoryTool:
    name: str = "list_directory"
    description: str = (
        "Lists files in a directory. Only use when the user explicitly "
        "asks what is in a folder, directory, or path. Read-only — does "
        "not open, modify, or delete files."
    )
    args_schema = ListDirectoryArgs
    requires_confirmation: bool = False

    async def execute(self, args: ListDirectoryArgs) -> ToolResult:
        def _list() -> tuple[Path, list[str]]:
            raw = Path(args.path).expanduser()
            if not raw.is_absolute():
                raw = (Path.home() / raw).resolve()
            else:
                raw = raw.resolve()
            if not raw.exists():
                raise FileNotFoundError(str(raw))
            if not raw.is_dir():
                raise NotADirectoryError(str(raw))
            return raw, sorted(p.name for p in raw.iterdir())

        try:
            resolved, names = await asyncio.to_thread(_list)
        except FileNotFoundError as e:
            return ToolResult(success=False, error=f"no such directory: {e}")
        except NotADirectoryError as e:
            return ToolResult(success=False, error=f"not a directory: {e}")
        except PermissionError as e:
            return ToolResult(success=False, error=f"permission denied: {e}")
        except OSError as e:
            return ToolResult(success=False, error=f"could not list: {e}")

        if not names:
            # Drop the resolved path from spoken output — TTS reading a
            # full Windows path is unintelligible. The path lives in
            # the log (above, via _list raising or returning) and the
            # eventual desktop notification surface (Phase 5).
            log.info("list_directory: %s is empty", resolved)
            return ToolResult(success=True, output="The folder is empty, sir.")
        # Hard cap. Even a 6-item directory is too much to read verbatim
        # over TTS in any natural-sounding cadence; pick the smaller of
        # (count, preview) and surface the full count alongside.
        preview = ", ".join(names[:_SPOKEN_PREVIEW_COUNT])
        suffix = (
            f"First {_SPOKEN_PREVIEW_COUNT}: {preview}."
            if len(names) > _SPOKEN_PREVIEW_COUNT
            else preview + "."
        )
        return ToolResult(
            success=True,
            output=f"{len(names)} items, sir. {suffix}",
        )
