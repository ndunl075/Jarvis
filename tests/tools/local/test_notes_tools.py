"""Tests for notes voice tools."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from jarvis.tools.local.notes_tools import (
    AppendToNoteArgs,
    AppendToNoteTool,
    DeleteNoteArgs,
    DeleteNoteTool,
    OpenNotesTool,
    ReadNoteArgs,
    ReadNoteTool,
    TakeNoteArgs,
    TakeNoteTool,
)
from jarvis.tools.registry import EmptyArgs


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_take_note_calls_callback_with_derived_title():
    captured = {}

    def on_create(title, content):
        captured["title"] = title
        captured["content"] = content
        return title

    tool = TakeNoteTool(on_create=on_create)
    result = _run(tool.execute(TakeNoteArgs(content="buy milk and eggs")))
    assert result.success is True
    assert "buy milk and eggs" in captured["content"]
    assert captured["title"]


def test_take_note_empty_content_fails():
    tool = TakeNoteTool(on_create=MagicMock())
    result = _run(tool.execute(TakeNoteArgs(content="  ")))
    assert result.success is False


def test_take_note_explicit_title_used():
    captured = {}

    def on_create(title, content):
        captured["title"] = title
        return title

    tool = TakeNoteTool(on_create=on_create)
    _run(tool.execute(TakeNoteArgs(content="anything", title="Custom Title")))
    assert captured["title"] == "Custom Title"


def test_append_active_used_when_title_empty():
    active = MagicMock(return_value="Meeting note")
    by_title = MagicMock()
    tool = AppendToNoteTool(on_append_active=active, on_append_by_title=by_title)
    result = _run(tool.execute(AppendToNoteArgs(content="extra")))
    assert result.success is True
    active.assert_called_once_with("extra")
    by_title.assert_not_called()


def test_append_by_title_used_when_provided():
    active = MagicMock()
    by_title = MagicMock(return_value="Groceries")
    tool = AppendToNoteTool(on_append_active=active, on_append_by_title=by_title)
    _run(tool.execute(AppendToNoteArgs(content="paper towels", title="groceries")))
    by_title.assert_called_once_with("groceries", "paper towels")
    active.assert_not_called()


def test_read_active_returns_body():
    tool = ReadNoteTool(
        on_read_active=lambda: ("Meeting", "Discussed launch."),
        on_read_by_title=lambda _t: None,
    )
    result = _run(tool.execute(ReadNoteArgs()))
    assert result.success is True
    assert "Discussed launch." in (result.output or "")


def test_read_active_no_match():
    tool = ReadNoteTool(
        on_read_active=lambda: None,
        on_read_by_title=lambda _t: None,
    )
    result = _run(tool.execute(ReadNoteArgs()))
    assert result.success is False


def test_open_notes():
    cb = MagicMock()
    tool = OpenNotesTool(on_open=cb)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    cb.assert_called_once()


def test_delete_active_used_when_empty():
    active = MagicMock(return_value="my note")
    by_title = MagicMock()
    tool = DeleteNoteTool(on_delete_active=active, on_delete_by_title=by_title)
    result = _run(tool.execute(DeleteNoteArgs()))
    assert result.success is True
    active.assert_called_once()


def test_delete_no_match():
    tool = DeleteNoteTool(
        on_delete_active=lambda: None,
        on_delete_by_title=lambda _t: None,
    )
    result = _run(tool.execute(DeleteNoteArgs(query="" if False else "missing")
                  if False else DeleteNoteArgs(title="missing")))
    assert result.success is False
