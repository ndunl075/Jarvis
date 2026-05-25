"""Tests for jarvis.tools.local.screenshot.ScreenshotTool."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from jarvis.tools.local.screenshot import ScreenshotTool
from jarvis.tools.registry import EmptyArgs


def _install_fake_pyautogui(saver_recorder: list[Path]) -> types.ModuleType:
    """Install a fake `pyautogui` module that returns an object whose
    .save(path) records the saved path. The real pyautogui pulls in
    mouse/keyboard hooks; we don't want that in unit tests."""
    fake = types.ModuleType("pyautogui")
    fake.screenshot = MagicMock()
    image = MagicMock()
    image.save.side_effect = lambda p: saver_recorder.append(Path(p))
    fake.screenshot.return_value = image
    return fake


async def test_saves_png_to_screenshots_dir_with_timestamp(tmp_path):
    saved: list[Path] = []
    fake = _install_fake_pyautogui(saved)
    with patch.dict(sys.modules, {"pyautogui": fake}), \
         patch(
            "jarvis.tools.local.screenshot.winplat.screenshots_dir",
            return_value=tmp_path,
         ):
        result = await ScreenshotTool().execute(EmptyArgs())
    assert result.success
    assert len(saved) == 1
    assert saved[0].parent == tmp_path
    assert saved[0].suffix == ".png"
    assert saved[0].name.startswith("jarvis_")


async def test_spoken_output_does_not_include_file_path(tmp_path):
    """Live-test fix: TTS was reading the full Windows path aloud
    ('C-colon-backslash-Users-backslash...'). Spoken output is now a
    fixed short string; the path goes to the log."""
    saved: list[Path] = []
    fake = _install_fake_pyautogui(saved)
    with patch.dict(sys.modules, {"pyautogui": fake}), \
         patch(
            "jarvis.tools.local.screenshot.winplat.screenshots_dir",
            return_value=tmp_path,
         ):
        result = await ScreenshotTool().execute(EmptyArgs())
    assert result.output == "Screenshot saved, sir."
    out = result.output or ""
    assert str(tmp_path) not in out
    assert ".png" not in out
    assert "\\" not in out
    assert "/" not in out


async def test_failure_returns_error(tmp_path):
    fake = _install_fake_pyautogui([])
    fake.screenshot.side_effect = RuntimeError("display unavailable")
    with patch.dict(sys.modules, {"pyautogui": fake}), \
         patch(
            "jarvis.tools.local.screenshot.winplat.screenshots_dir",
            return_value=tmp_path,
         ):
        result = await ScreenshotTool().execute(EmptyArgs())
    assert not result.success
    assert "display unavailable" in (result.error or "")


def test_requires_confirmation_false():
    assert ScreenshotTool().requires_confirmation is False
