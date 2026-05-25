"""Tests for enable/disable Deep Research Ultra voice tools."""

from __future__ import annotations

import pytest

from jarvis.tools.local.deep_research_ultra_tools import (
    DisableDeepResearchUltraTool,
    EnableDeepResearchUltraTool,
)
from jarvis.tools.registry import EmptyArgs


@pytest.mark.asyncio
async def test_enable_ultra_tool():
    calls: list[bool] = []

    def set_ultra(enabled: bool) -> str:
        calls.append(enabled)
        return "on" if enabled else "off"

    tool = EnableDeepResearchUltraTool(set_ultra=set_ultra)
    result = await tool.execute(EmptyArgs())
    assert result.success
    assert calls == [True]
    assert "on" in (result.output or "")


@pytest.mark.asyncio
async def test_disable_ultra_tool():
    calls: list[bool] = []

    def set_ultra(enabled: bool) -> str:
        calls.append(enabled)
        return "off"

    tool = DisableDeepResearchUltraTool(set_ultra=set_ultra)
    result = await tool.execute(EmptyArgs())
    assert result.success
    assert calls == [False]
