"""Tests for jarvis.tools.local.files.ListDirectoryTool.

Read-only by design — there are no write/delete/move tests because there
are no such operations. If a future PR adds them, this test file should
guard their destructive nature with explicit tmp_path containment plus
requires_confirmation=True checks (see registry.py header on deferred
UX)."""

from __future__ import annotations

from jarvis.tools.local.files import ListDirectoryArgs, ListDirectoryTool


async def test_lists_directory_entries_sorted(tmp_path):
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "sub").mkdir()
    result = await ListDirectoryTool().execute(
        ListDirectoryArgs(path=str(tmp_path))
    )
    assert result.success
    out = result.output or ""
    # Names appear sorted (a, b, sub).
    assert out.index("a.txt") < out.index("b.txt") < out.index("sub")


async def test_empty_directory_says_so(tmp_path):
    result = await ListDirectoryTool().execute(
        ListDirectoryArgs(path=str(tmp_path))
    )
    assert result.success
    assert "empty" in (result.output or "").lower()


async def test_nonexistent_path_returns_error(tmp_path):
    result = await ListDirectoryTool().execute(
        ListDirectoryArgs(path=str(tmp_path / "no_such_dir"))
    )
    assert not result.success
    assert "no such" in (result.error or "").lower()


async def test_file_path_rejected_as_not_a_directory(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hi")
    result = await ListDirectoryTool().execute(ListDirectoryArgs(path=str(f)))
    assert not result.success
    assert "not a directory" in (result.error or "").lower()


async def test_large_directory_summarised(tmp_path):
    for i in range(250):
        (tmp_path / f"f_{i:04d}.txt").write_text("")
    result = await ListDirectoryTool().execute(
        ListDirectoryArgs(path=str(tmp_path))
    )
    assert result.success
    out = result.output or ""
    assert "250" in out
    # The summary shouldn't include every entry verbatim.
    assert "f_0249" not in out


def test_requires_confirmation_false():
    assert ListDirectoryTool().requires_confirmation is False
