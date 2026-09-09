"""Tests for ResearchTool and associated tools (research.py)."""

from __future__ import annotations

import asyncio
import queue
from unittest.mock import AsyncMock, MagicMock

from jarvis.tools.local.research import (
    CloseResearchTool,
    CopyResearchTool,
    ReadMoreTool,
    ResearchArgs,
    ResearchTool,
    _first_n_sentences,
)
from jarvis.tools.registry import EmptyArgs


def _run(coro):
    return asyncio.run(coro)


def _make_tool(on_start=None):
    cb = on_start or MagicMock()
    return ResearchTool(on_start=cb), cb


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_research_tool_metadata():
    tool, _ = _make_tool()
    assert tool.name == "research"
    assert tool.requires_confirmation is False
    assert tool.args_schema is ResearchArgs
    assert "ollama" in tool.description.lower()


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_empty_query_returns_error():
    tool, on_start = _make_tool()
    result = _run(tool.execute(ResearchArgs(query="   ")))
    assert result.success is False
    assert "empty" in (result.error or "").lower()
    on_start.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_on_start_called_with_query_and_queue():
    """on_start must receive (query, result_queue) exactly once."""
    captured: list = []

    def fake_on_start(query, result_q):
        captured.append((query, result_q))
        result_q.put((f"Summary for {query}", []))

    tool = ResearchTool(on_start=fake_on_start)
    result = _run(tool.execute(ResearchArgs(query="black holes")))

    assert len(captured) == 1
    query, rq = captured[0]
    assert query == "black holes"
    assert isinstance(rq, queue.Queue)
    assert result.success is True


def test_output_is_first_two_sentences():
    summary = "First sentence. Second sentence. Third sentence. Fourth sentence."

    def fake_on_start(query, result_q):
        result_q.put((summary, []))

    tool = ResearchTool(on_start=fake_on_start)
    result = _run(tool.execute(ResearchArgs(query="test")))

    assert result.success is True
    assert result.output == "First sentence. Second sentence."
    assert "Third" not in (result.output or "")


def test_bullet_points_stripped_from_tts():
    summary = "• Point one\n• Point two\n• Point three"

    def fake_on_start(query, result_q):
        result_q.put((summary, []))

    tool = ResearchTool(on_start=fake_on_start)
    result = _run(tool.execute(ResearchArgs(query="test")))

    assert "•" not in (result.output or "")


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_worker_error_propagates_as_failure():
    def fake_on_start(query, result_q):
        result_q.put(RuntimeError("SSL error"))

    tool = ResearchTool(on_start=fake_on_start)
    result = _run(tool.execute(ResearchArgs(query="test")))

    assert result.success is False
    assert "SSL error" in (result.error or "")


def test_timeout_returns_failure(monkeypatch):
    def fake_on_start(query, result_q):
        pass

    async def _fast_to_thread(fn):
        raise queue.Empty

    import jarvis.tools.local.research as rmod
    monkeypatch.setattr(rmod.asyncio, "to_thread", _fast_to_thread)

    tool = ResearchTool(on_start=fake_on_start)
    result = _run(tool.execute(ResearchArgs(query="test")))

    assert result.success is False
    assert "timed out" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# CloseResearchTool
# ---------------------------------------------------------------------------


def test_close_tool_metadata():
    cb = MagicMock()
    tool = CloseResearchTool(close_callback=cb)
    assert tool.name == "close_research"
    assert tool.requires_confirmation is False
    assert tool.args_schema is EmptyArgs


def test_close_tool_invokes_callback():
    cb = MagicMock()
    tool = CloseResearchTool(close_callback=cb)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    cb.assert_called_once()


# ---------------------------------------------------------------------------
# _first_n_sentences helper
# ---------------------------------------------------------------------------


def test_first_n_sentences_basic():
    text = "First sentence. Second sentence. Third sentence."
    assert _first_n_sentences(text, 2) == "First sentence. Second sentence."


def test_first_n_sentences_strips_bullets():
    text = "• Point one. • Point two. • Point three."
    result = _first_n_sentences(text, 2)
    assert "•" not in result


def test_first_n_sentences_short_text():
    text = "Only one sentence."
    assert _first_n_sentences(text, 2) == "Only one sentence."


# ---------------------------------------------------------------------------
# on_speak pre-search announcement
# ---------------------------------------------------------------------------


def test_on_speak_called_before_research():
    call_order: list[str] = []

    async def fake_speak(msg: str) -> None:
        call_order.append(f"speak:{msg}")

    def fake_on_start(query, result_q):
        call_order.append("on_start")
        result_q.put(("Summary.", []))

    tool = ResearchTool(on_start=fake_on_start, on_speak=fake_speak)
    _run(tool.execute(ResearchArgs(query="black holes")))

    assert call_order[0].startswith("speak:")
    assert call_order[1] == "on_start"


def test_on_speak_failure_does_not_break_tool():
    async def bad_speak(msg: str) -> None:
        raise RuntimeError("TTS broken")

    def fake_on_start(query, result_q):
        result_q.put(("Summary.", []))

    tool = ResearchTool(on_start=fake_on_start, on_speak=bad_speak)
    result = _run(tool.execute(ResearchArgs(query="test")))

    assert result.success is True


def test_on_speak_not_wired_by_default():
    def fake_on_start(query, result_q):
        result_q.put(("Summary.", []))

    tool = ResearchTool(on_start=fake_on_start)
    result = _run(tool.execute(ResearchArgs(query="test")))

    assert result.success is True


# ---------------------------------------------------------------------------
# ReadMoreTool
# ---------------------------------------------------------------------------


def test_read_more_metadata():
    tool = ReadMoreTool(get_next=MagicMock())
    assert tool.name == "read_more"
    assert tool.requires_confirmation is False
    assert tool.args_schema is EmptyArgs


def test_read_more_returns_next_sentences():
    get_next = MagicMock(return_value="Sentence three. Sentence four.")
    tool = ReadMoreTool(get_next=get_next)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    assert result.output == "Sentence three. Sentence four."
    get_next.assert_called_once()


def test_read_more_end_of_summary():
    get_next = MagicMock(return_value=None)
    tool = ReadMoreTool(get_next=get_next)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    assert "end" in (result.output or "").lower()


# ---------------------------------------------------------------------------
# CopyResearchTool
# ---------------------------------------------------------------------------


def test_copy_research_metadata():
    tool = CopyResearchTool(copy_callback=MagicMock())
    assert tool.name == "copy_research"
    assert tool.requires_confirmation is False
    assert tool.args_schema is EmptyArgs


def test_copy_research_invokes_callback():
    cb = MagicMock()
    tool = CopyResearchTool(copy_callback=cb)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    assert "clipboard" in (result.output or "").lower()
    cb.assert_called_once()
