"""Lock the workstation (Win+L equivalent)."""

from __future__ import annotations

import asyncio
from typing import ClassVar

from pydantic import BaseModel

from jarvis.platform import windows as winplat
from jarvis.tools.registry import EmptyArgs, ToolResult, VoicePattern


class LockScreenTool:
    name: str = "lock_screen"
    description: str = (
        "Locks the Windows screen. Only use when the user explicitly "
        "asks to lock the screen or PC."
    )
    args_schema: type[BaseModel] = EmptyArgs
    # NOTE: re-decided when the confirmation gate landed, rather than
    # inherited. The gate exists now, so "stays False until the UX is
    # wired" is no longer a reason for anything; this tool is ungated
    # because it does not qualify.
    #
    # Locking is disruptive but it is not destructive: nothing is lost,
    # nothing leaves the machine, and signing back in undoes all of it
    # in seconds. It is also the one action here whose failure mode is
    # *safe* — a spurious lock leaves the workstation more secure, not
    # less, which is the opposite of type_into_active_window's.
    #
    # And it is reached by an exact voice pattern ("lock the screen"),
    # so the misfire risk the gate is for — the model inventing a call
    # from an ambiguous transcription — is largely bypassed. Gating it
    # would put a modal dialog in front of the deliberate, correct,
    # common path in exchange for softening a two-second annoyance.
    requires_confirmation: bool = False
    voice_patterns: ClassVar[tuple[VoicePattern, ...]] = (
        VoicePattern(
            regex=r"^lock\s+(?:the\s+)?(?:screen|my\s+pc|pc)$", priority=30
        ),
    )

    async def execute(self, args: EmptyArgs) -> ToolResult:
        try:
            await asyncio.to_thread(winplat.lock_screen)
        except NotImplementedError as e:
            return ToolResult(success=False, error=str(e))
        except OSError as e:
            return ToolResult(success=False, error=f"lock failed: {e}")
        return ToolResult(success=True, output="Locking the workstation, sir.")
