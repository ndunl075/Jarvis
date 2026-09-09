"""Tests for jarvis.tools.local.lock_screen.LockScreenTool.

Critically: these tests MUST mock winplat.lock_screen. A test that
actually calls into ctypes would lock the developer's session running
the suite."""

from __future__ import annotations

from unittest.mock import patch

from jarvis.tools.local.lock_screen import LockScreenTool
from jarvis.tools.registry import EmptyArgs


async def test_calls_platform_lock_screen():
    with patch("jarvis.tools.local.lock_screen.winplat.lock_screen") as lock:
        result = await LockScreenTool().execute(EmptyArgs())
    assert result.success
    lock.assert_called_once_with()


async def test_oserror_surfaces_as_result_error():
    with patch(
        "jarvis.tools.local.lock_screen.winplat.lock_screen",
        side_effect=OSError("LockWorkStation returned 0"),
    ):
        result = await LockScreenTool().execute(EmptyArgs())
    assert not result.success
    assert "LockWorkStation" in (result.error or "")


async def test_non_windows_returns_error_not_raise():
    with patch(
        "jarvis.tools.local.lock_screen.winplat.lock_screen",
        side_effect=NotImplementedError("Windows-only"),
    ):
        result = await LockScreenTool().execute(EmptyArgs())
    assert not result.success
    assert "Windows" in (result.error or "")


def test_requires_confirmation_false():
    """Ungated on purpose, and re-decided once the gate existed.

    Locking is interruptive but not destructive, it is undone by signing
    back in, and its failure mode is a *more* secure workstation. It is
    also pattern-routed, so the LLM-misfire risk the gate exists for
    mostly does not apply. Flipping this True puts a modal dialog in
    front of the common, deliberate path; the comment on the tool says
    the same thing at more length."""
    assert LockScreenTool().requires_confirmation is False
