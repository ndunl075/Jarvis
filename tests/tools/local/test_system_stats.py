"""Tests for jarvis.tools.local.system_stats.SystemStatsTool."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from jarvis.tools.local.system_stats import SystemStatsTool
from jarvis.tools.registry import EmptyArgs


def _fake_psutil(cpu: float, mem: float) -> types.ModuleType:
    fake = types.ModuleType("psutil")
    fake.cpu_percent = MagicMock(return_value=cpu)
    vm = MagicMock()
    vm.percent = mem
    fake.virtual_memory = MagicMock(return_value=vm)
    return fake


async def test_returns_cpu_and_memory_in_spoken_summary():
    with patch.dict(sys.modules, {"psutil": _fake_psutil(47.3, 62.0)}):
        result = await SystemStatsTool().execute(EmptyArgs())
    assert result.success
    out = result.output or ""
    assert "47" in out
    assert "62" in out
    assert "sir" in out.lower()


async def test_failure_returns_error():
    fake = types.ModuleType("psutil")
    fake.cpu_percent = MagicMock(side_effect=RuntimeError("counter not ready"))
    with patch.dict(sys.modules, {"psutil": fake}):
        result = await SystemStatsTool().execute(EmptyArgs())
    assert not result.success
    assert "counter not ready" in (result.error or "")


def test_requires_confirmation_false():
    assert SystemStatsTool().requires_confirmation is False
