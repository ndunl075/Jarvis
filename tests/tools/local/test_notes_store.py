"""Tests for notes persistence."""

from __future__ import annotations

from jarvis.tools.local.notes_store import (
    append_to_note,
    create_note,
    delete_all_notes,
    delete_note,
    find_note_by_title,
    list_notes,
    load_note,
    save_note,
)


def _patch_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.tools.local.notes_store.notes_root",
        lambda: tmp_path,
    )


def test_create_note_writes_file_with_front_matter(tmp_path, monkeypatch):
    _patch_root(tmp_path, monkeypatch)
    note = create_note("Meeting with Alex", "Discussed launch plan.")
    assert note.title == "Meeting with Alex"
    assert note.content == "Discussed launch plan."
    raw = (tmp_path / f"{note.id}.md").read_text(encoding="utf-8")
    assert raw.startswith("---")
    assert "title: Meeting with Alex" in raw
    assert "Discussed launch plan." in raw


def test_load_note_round_trip(tmp_path, monkeypatch):
    _patch_root(tmp_path, monkeypatch)
    note = create_note("Groceries", "milk\neggs")
    loaded = load_note(note.id)
    assert loaded is not None
    assert loaded.title == "Groceries"
    assert "milk" in loaded.content
    assert "eggs" in loaded.content


def test_list_notes_orders_newest_first(tmp_path, monkeypatch):
    _patch_root(tmp_path, monkeypatch)
    import time as _t

    a = create_note("Older", "1")
    _t.sleep(0.02)
    b = create_note("Newer", "2")
    items = list_notes()
    assert items[0].id == b.id
    assert items[1].id == a.id


def test_append_to_note(tmp_path, monkeypatch):
    _patch_root(tmp_path, monkeypatch)
    n = create_note("Project", "Kickoff Monday.")
    append_to_note(n, "Send proposal Friday.")
    reloaded = load_note(n.id)
    assert reloaded is not None
    assert "Kickoff Monday." in reloaded.content
    assert "Send proposal Friday." in reloaded.content


def test_find_note_by_title_substring(tmp_path, monkeypatch):
    _patch_root(tmp_path, monkeypatch)
    create_note("Quantum computing thoughts", "")
    create_note("Wind energy", "")
    found = find_note_by_title("quantum")
    assert found is not None
    assert "Quantum" in found.title
    assert find_note_by_title("nothing") is None
    assert find_note_by_title("") is None


def test_delete_note(tmp_path, monkeypatch):
    _patch_root(tmp_path, monkeypatch)
    n = create_note("Disposable", "x")
    assert delete_note(n.id) is True
    assert load_note(n.id) is None


def test_delete_note_rejects_traversal(tmp_path, monkeypatch):
    _patch_root(tmp_path, monkeypatch)
    assert delete_note("../evil") is False
    assert delete_note("a/b") is False
    assert delete_note("") is False


def test_delete_all_notes(tmp_path, monkeypatch):
    _patch_root(tmp_path, monkeypatch)
    create_note("a", "1")
    create_note("b", "2")
    create_note("c", "3")
    n = delete_all_notes()
    assert n == 3
    assert list_notes() == []


def test_save_note_updates_updated_at(tmp_path, monkeypatch):
    _patch_root(tmp_path, monkeypatch)
    n = create_note("Edit me", "original")
    original_updated = n.updated_at
    import time as _t

    _t.sleep(0.02)
    n.content = "modified"
    save_note(n)
    reloaded = load_note(n.id)
    assert reloaded is not None
    assert reloaded.content == "modified"
    assert reloaded.updated_at >= original_updated
