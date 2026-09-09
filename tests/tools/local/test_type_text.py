"""Tests for jarvis.tools.local.type_into_active_window.TypeIntoActiveWindowTool."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from jarvis.tools.local.type_into_active_window import (
    TypeIntoActiveWindowArgs,
    TypeIntoActiveWindowTool,
)


def _fake_pyautogui() -> tuple[types.ModuleType, MagicMock]:
    fake = types.ModuleType("pyautogui")
    typewrite = MagicMock()
    fake.typewrite = typewrite
    return fake, typewrite


async def test_typewrite_called_with_text_and_interval():
    fake, tw = _fake_pyautogui()
    with patch.dict(sys.modules, {"pyautogui": fake}):
        result = await TypeIntoActiveWindowTool().execute(
            TypeIntoActiveWindowArgs(text="hello world", interval_seconds=0.01)
        )
    assert result.success
    tw.assert_called_once_with("hello world", interval=0.01)


async def test_long_text_rejected_before_typing():
    fake, tw = _fake_pyautogui()
    with patch.dict(sys.modules, {"pyautogui": fake}):
        result = await TypeIntoActiveWindowTool().execute(
            TypeIntoActiveWindowArgs(text="x" * 5000)
        )
    assert not result.success
    assert "too long" in (result.error or "")
    tw.assert_not_called()


async def test_typewrite_exception_surfaces_as_error():
    fake, tw = _fake_pyautogui()
    tw.side_effect = RuntimeError("focus window gone")
    with patch.dict(sys.modules, {"pyautogui": fake}):
        result = await TypeIntoActiveWindowTool().execute(TypeIntoActiveWindowArgs(text="hi"))
    assert not result.success
    assert "focus window gone" in (result.error or "")


def test_interval_bounds_enforced_by_schema():
    with pytest.raises(Exception):  # noqa: B017
        TypeIntoActiveWindowArgs(text="hi", interval_seconds=-0.1)
    with pytest.raises(Exception):  # noqa: B017
        TypeIntoActiveWindowArgs(text="hi", interval_seconds=1.0)


def test_requires_confirmation_true():
    """The one built-in tool behind the confirmation gate.

    It synthesises keystrokes into whatever window has focus, which the
    LLM cannot see. Flipping this back to False silently removes the
    only thing standing between a misheard sentence and a terminal."""
    assert TypeIntoActiveWindowTool().requires_confirmation is True


def test_confirmation_summary_says_what_and_where():
    """The prompt has to answer "what is about to happen", and the
    model-facing description does not."""
    summary = TypeIntoActiveWindowTool().confirmation_summary(
        TypeIntoActiveWindowArgs(text="rm -rf /")
    )
    assert "8 character(s)" in summary
    assert "keyboard focus" in summary
