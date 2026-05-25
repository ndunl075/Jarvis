"""Tests for jarvis.tools.local.close_app.CloseAppTool.

The process lister and killer are injected so tests never touch real
processes."""

from __future__ import annotations

from jarvis.tools.local.close_app import CloseAppTool, RunningProcess


def _lister(*procs: tuple[int, str]):
    return lambda: [RunningProcess(pid=pid, name=name) for pid, name in procs]


def _recording_killer():
    killed: list[int] = []

    def kill(pids):
        killed.extend(pids)

    return killed, kill


async def test_closes_matching_process_group():
    """A multi-process app (Chrome) terminates every PID under that name."""
    killed, kill = _recording_killer()
    tool = CloseAppTool(
        process_lister=_lister((10, "chrome.exe"), (11, "chrome.exe"), (12, "notepad.exe")),
        process_killer=kill,
    )
    result = await tool.execute(CloseAppTool.args_schema(name="chrome"))
    assert result.success
    assert sorted(killed) == [10, 11]
    assert "Chrome" in (result.output or "")


async def test_fuzzy_matches_single_word_to_process():
    killed, kill = _recording_killer()
    tool = CloseAppTool(
        process_lister=_lister((20, "Spotify.exe")),
        process_killer=kill,
    )
    result = await tool.execute(CloseAppTool.args_schema(name="spotify"))
    assert result.success
    assert killed == [20]


async def test_alias_canonical_matches_process():
    """'vscode' -> alias 'code' -> Code.exe process."""
    killed, kill = _recording_killer()
    tool = CloseAppTool(
        process_lister=_lister((30, "Code.exe")),
        process_killer=kill,
    )
    result = await tool.execute(CloseAppTool.args_schema(name="vscode"))
    assert result.success
    assert killed == [30]


async def test_not_running_returns_spoken_error():
    killed, kill = _recording_killer()
    tool = CloseAppTool(
        process_lister=_lister((40, "notepad.exe")),
        process_killer=kill,
    )
    result = await tool.execute(CloseAppTool.args_schema(name="spotify"))
    assert not result.success
    assert "spotify" in (result.error or "").lower()
    assert killed == []  # nothing killed on a miss


async def test_protected_process_never_matched():
    """A query that fuzzes toward a protected name must not kill it."""
    killed, kill = _recording_killer()
    tool = CloseAppTool(
        process_lister=_lister((1, "explorer.exe"), (2, "csrss.exe"), (3, "svchost.exe")),
        process_killer=kill,
    )
    result = await tool.execute(CloseAppTool.args_schema(name="explorer"))
    assert not result.success
    assert killed == []


async def test_empty_name_rejected():
    killed, kill = _recording_killer()
    tool = CloseAppTool(process_lister=_lister((1, "chrome.exe")), process_killer=kill)
    result = await tool.execute(CloseAppTool.args_schema(name="   "))
    assert not result.success
    assert killed == []


async def test_lister_failure_is_handled():
    def boom():
        raise RuntimeError("psutil exploded")

    tool = CloseAppTool(process_lister=boom, process_killer=lambda pids: None)
    result = await tool.execute(CloseAppTool.args_schema(name="chrome"))
    assert not result.success
    assert "could not list processes" in (result.error or "")


def test_requires_confirmation_false():
    assert CloseAppTool().requires_confirmation is False
