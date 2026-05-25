"""Capture the primary display to a timestamped PNG."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from pydantic import BaseModel

from jarvis.platform import windows as winplat
from jarvis.tools.registry import EmptyArgs, ToolResult

log = logging.getLogger(__name__)


class ScreenshotTool:
    name: str = "screenshot"
    description: str = (
        "Captures the screen and saves the image. Only use when the user "
        "explicitly asks for a screenshot or to capture the screen."
    )
    args_schema: type[BaseModel] = EmptyArgs
    requires_confirmation: bool = False

    async def execute(self, args: EmptyArgs) -> ToolResult:
        def _grab_and_save() -> str:
            # Late import: pyautogui pulls in a heavy chain (mouse/keyboard
            # hooks). Keep that off the cold-import path of consumers that
            # never call screenshot.
            import pyautogui
            target = winplat.screenshots_dir() / (
                datetime.now().strftime("jarvis_%Y%m%d_%H%M%S.png")
            )
            image = pyautogui.screenshot()
            image.save(target)
            return str(target)

        try:
            path = await asyncio.to_thread(_grab_and_save)
        except Exception as e:
            return ToolResult(success=False, error=f"screenshot failed: {e}")
        # Spoken output is intentionally just "Screenshot saved, sir." —
        # TTS reading the full path ("C-colon-backslash-Users-backslash...")
        # is a UX disaster. The path goes to the log for debugging
        # instead, where it's actually useful (the desktop notification
        # surface in Phase 5 will be the proper place to surface it
        # visually).
        log.info("screenshot saved: %s", path)
        return ToolResult(success=True, output="Screenshot saved, sir.")
