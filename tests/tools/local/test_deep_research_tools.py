"""Tests for deep research voice tools."""

from __future__ import annotations

import asyncio
import queue
from unittest.mock import AsyncMock, MagicMock

from jarvis.tools.local.deep_research_tools import (
    DeepResearchArgs,
    DeepResearchTool,
    DeleteAllDeepResearchTool,
    DeleteDeepResearchArgs,
    DeleteDeepResearchTool,
    PauseDeepResearchTool,
    ResumeDeepResearchTool,
)
from jarvis.tools.registry import EmptyArgs


def _run(coro):
    return asyncio.run(coro)


def test_deep_research_empty_query():
    tool = DeepResearchTool(on_start=MagicMock())
    result = _run(tool.execute(DeepResearchArgs(query="  ")))
    assert result.success is False


def test_deep_research_waits_for_queue_result():
    q: queue.Queue = queue.Queue()

    def on_start(query, result_q):
        result_q.put(("Done, sir.", "sess-1"))

    tool = DeepResearchTool(on_start=on_start, on_speak=AsyncMock())
    result = _run(tool.execute(DeepResearchArgs(query="mars colonization")))
    assert result.success is True
    assert "Done" in (result.output or "")


def test_pause_calls_callback():
    pause = MagicMock()
    tool = PauseDeepResearchTool(on_pause=pause)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    pause.assert_called_once()


def test_resume_no_session():
    tool = ResumeDeepResearchTool(on_resume_latest=lambda _q: None)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is False


def test_delete_by_query_calls_callback():
    by_query = MagicMock(return_value="quantum computing")
    active = MagicMock(return_value=None)
    tool = DeleteDeepResearchTool(delete_by_query=by_query, delete_active=active)
    result = _run(tool.execute(DeleteDeepResearchArgs(query="quantum")))
    assert result.success is True
    by_query.assert_called_once_with("quantum")
    active.assert_not_called()
    assert "quantum" in (result.output or "").lower()


def test_delete_empty_query_uses_active():
    active = MagicMock(return_value="solar power")
    by_query = MagicMock()
    tool = DeleteDeepResearchTool(delete_by_query=by_query, delete_active=active)
    result = _run(tool.execute(DeleteDeepResearchArgs(query=" ")))
    assert result.success is True
    active.assert_called_once()
    by_query.assert_not_called()


def test_delete_no_match():
    tool = DeleteDeepResearchTool(
        delete_by_query=lambda _q: None,
        delete_active=lambda: None,
    )
    result = _run(tool.execute(DeleteDeepResearchArgs(query="nothing")))
    assert result.success is False


def test_delete_all_reports_count():
    tool = DeleteAllDeepResearchTool(delete_all=lambda: 3)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    assert "3" in (result.output or "")


def test_delete_all_empty():
    tool = DeleteAllDeepResearchTool(delete_all=lambda: 0)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    assert "no" in (result.output or "").lower()
