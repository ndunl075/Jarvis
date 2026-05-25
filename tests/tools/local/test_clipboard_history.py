"""Tests for the clipboard history store and voice tools."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from jarvis.tools.local.clipboard_history import (
    ClipboardHistory,
    load_history,
    save_history,
)
from jarvis.tools.local.clipboard_history_tools import (
    ClearClipboardHistoryTool,
    CloseClipboardHistoryTool,
    PasteClipboardItemArgs,
    PasteClipboardItemTool,
    ShowClipboardHistoryTool,
)
from jarvis.tools.registry import EmptyArgs


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# --- store -----------------------------------------------------------


def test_add_appends_unique_items_at_front():
    h = ClipboardHistory()
    assert h.add("alpha") is True
    assert h.add("beta") is True
    assert [i.text for i in h.items] == ["beta", "alpha"]


def test_add_dedupes_consecutive_duplicates():
    h = ClipboardHistory()
    h.add("alpha")
    assert h.add("alpha") is False
    assert len(h.items) == 1


def test_add_moves_existing_to_front():
    h = ClipboardHistory()
    h.add("alpha")
    h.add("beta")
    h.add("gamma")
    h.add("alpha")
    assert h.items[0].text == "alpha"
    assert h.items[1].text == "gamma"
    assert len(h.items) == 3


def test_add_empty_and_whitespace_rejected():
    h = ClipboardHistory()
    assert h.add("") is False
    assert h.add("   \n\t") is False
    assert h.add(None) is False  # type: ignore[arg-type]
    assert h.items == []


def test_add_rejects_credential_looking_payloads():
    h = ClipboardHistory()
    assert h.add("user=foo password=hunter2") is False
    assert h.add("api_key=sk-abcd1234") is False
    assert h.items == []


def test_pin_keeps_item_through_eviction():
    h = ClipboardHistory()
    for i in range(60):
        h.add(f"x{i}")
    assert len(h.items) <= 50
    pinned_idx = 49
    h.pin(pinned_idx, True)
    pinned_text = h.items[pinned_idx].text
    for i in range(60, 130):
        h.add(f"y{i}")
    assert any(it.text == pinned_text for it in h.items)


def test_remove_and_clear():
    h = ClipboardHistory()
    h.add("a")
    h.add("b")
    h.add("c")
    assert h.remove(1) is True
    assert [i.text for i in h.items] == ["c", "a"]
    h.pin(0, True)
    cleared = h.clear(keep_pinned=True)
    assert cleared == 1
    assert [i.text for i in h.items] == ["c"]
    cleared = h.clear(keep_pinned=False)
    assert cleared == 1


def test_round_trip_save_load(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.clipboard_history.history_path",
        lambda: tmp_path / "ch.json",
    )
    h = ClipboardHistory()
    h.add("hello world")
    h.add("multi\nline\nentry")
    h.pin(0, True)
    save_history(h)
    loaded = load_history()
    assert [i.text for i in loaded.items] == [
        "multi\nline\nentry",
        "hello world",
    ]
    assert loaded.items[0].pinned is True


# --- voice tools -----------------------------------------------------


def test_show_clipboard_history_calls_open():
    cb = MagicMock()
    tool = ShowClipboardHistoryTool(on_open=cb)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    cb.assert_called_once()


def test_close_clipboard_history_calls_close():
    cb = MagicMock()
    tool = CloseClipboardHistoryTool(on_close=cb)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    cb.assert_called_once()


def test_paste_item_routes_index():
    captured = {}

    def on_paste(idx):
        captured["idx"] = idx
        return "preview text"

    tool = PasteClipboardItemTool(on_paste=on_paste)
    result = _run(tool.execute(PasteClipboardItemArgs(index=3)))
    assert result.success is True
    assert captured["idx"] == 3
    assert "preview text" in (result.output or "")


def test_paste_item_no_match_returns_failure():
    tool = PasteClipboardItemTool(on_paste=lambda _i: None)
    result = _run(tool.execute(PasteClipboardItemArgs(index=99)))
    assert result.success is False


def test_clear_returns_count():
    tool = ClearClipboardHistoryTool(on_clear=lambda: 7)
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is True
    assert "7" in (result.output or "")
