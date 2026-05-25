"""Lock the workstation (Win+L equivalent)."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from jarvis.platform import windows as winplat
from jarvis.tools.registry import EmptyArgs, ToolResult


class LockScreenTool:
    name: str = "lock_screen"
    description: str = (
        "Locks the Windows screen. Only use when the user explicitly "
        "asks to lock the screen or PC."
    )
    args_schema: type[BaseModel] = EmptyArgs
    # NOTE: this is the closest thing to a destructive action in the Phase
    # 4 set — interrupts whatever the user is doing — but it is reversible
    # by signing back in. requires_confirmation stays False until Phase 6+
    # wires hotkey-based cancellation (see registry.py header).
    requires_confirmation: bool = False

    async def execute(self, args: EmptyArgs) -> ToolResult:
        try:
            await asyncio.to_thread(winplat.lock_screen)
        except NotImplementedError as e:
            return ToolResult(success=False, error=str(e))
        except OSError as e:
            return ToolResult(success=False, error=f"lock failed: {e}")
        return ToolResult(success=True, output="Locking the workstation, sir.")
