"""Tests for dashboard and help voice tools."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from jarvis.tools.local.dashboard_tools import CloseDashboardTool, ShowDashboardTool
from jarvis.tools.local.help_tools import OpenHelpTool
from jarvis.tools.registry import EmptyArgs


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_show_dashboard_calls_open():
    cb = MagicMock()
    tool = ShowDashboardTool(on_open=cb)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    cb.assert_called_once()


def test_close_dashboard_calls_close():
    cb = MagicMock()
    tool = CloseDashboardTool(on_close=cb)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    cb.assert_called_once()


def test_open_help_calls_open():
    cb = MagicMock()
    tool = OpenHelpTool(on_open=cb)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    cb.assert_called_once()
