"""Tests for the live log viewer voice tools + default log path."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from jarvis.tools.local.log_tools import CloseLogsTool, ShowLogsTool
from jarvis.tools.registry import EmptyArgs


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_show_logs_calls_open():
    cb = MagicMock()
    tool = ShowLogsTool(on_open=cb)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    cb.assert_called_once()


def test_close_logs_calls_close():
    cb = MagicMock()
    tool = CloseLogsTool(on_close=cb)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    cb.assert_called_once()


def test_line_level_extraction():
    from jarvis.ui.log_panel import _line_level

    assert _line_level("22:10:11 [INFO] jarvis.app: starting up") == "INFO"
    assert _line_level("22:10:11 [WARNING] x: y") == "WARNING"
    assert _line_level("22:10:11 [ERROR] x: y") == "ERROR"
    assert _line_level("no level at all here") is None
    assert _line_level("") is None
    # Bracket too far in — likely a stack trace line, not a logging header.
    assert _line_level("              [INFO] indented") is None


def test_default_log_path_under_appdata_on_windows(monkeypatch):
    from jarvis.ui import log_panel as _lp

    monkeypatch.setenv("APPDATA", r"C:\Users\test\AppData\Roaming")
    monkeypatch.setattr(_lp, "sys", _lp.sys)  # keep sys reference
    # Force sys.platform to win32 for this check
    monkeypatch.setattr(_lp.sys, "platform", "win32", raising=False)
    p = _lp.default_log_path()
    assert "Jarvis" in str(p)
    assert p.name == "jarvis.log"
