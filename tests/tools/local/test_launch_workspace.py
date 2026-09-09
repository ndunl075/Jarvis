"""Tests for LaunchWorkspaceTool."""

from __future__ import annotations

import asyncio
import pathlib
from unittest.mock import patch

import pytest

from jarvis.core.config import WorkspaceAppEntry
from jarvis.tools.local.launch_workspace import LaunchWorkspaceTool, _launch_entry
from jarvis.tools.registry import EmptyArgs


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def sample_apps() -> list[WorkspaceAppEntry]:
    return [
        WorkspaceAppEntry(label="Cursor", kind="executable", target=r"C:\cursor\Cursor.exe"),
        WorkspaceAppEntry(label="Music", kind="shell", target=r"shell:AppsFolder\Music"),
    ]


@pytest.fixture()
def tool(sample_apps):
    return LaunchWorkspaceTool(workspace_apps=sample_apps)


def test_tool_metadata(tool, sample_apps):
    assert tool.name == "launch_workspace"
    assert tool.requires_confirmation is False
    assert "Cursor" in tool.description
    assert "Music" in tool.description
    assert tool.args_schema is EmptyArgs


def test_execute_launches_all_apps(tool, sample_apps):
    with patch(
        "jarvis.tools.local.launch_workspace._launch_entry",
    ) as mock_launch:
        result = _run(tool.execute(EmptyArgs()))

    assert result.success is True
    assert "workspace" in result.output.lower()
    assert mock_launch.call_count == len(sample_apps)


def test_execute_partial_failure_still_succeeds(tool):
    def side_effect(entry):
        if entry.label == "Cursor":
            raise FileNotFoundError("missing")

    with patch(
        "jarvis.tools.local.launch_workspace._launch_entry",
        side_effect=side_effect,
    ):
        result = _run(tool.execute(EmptyArgs()))

    assert result.success is True


def test_execute_returns_failure_when_all_apps_fail(tool):
    with patch(
        "jarvis.tools.local.launch_workspace._launch_entry",
        side_effect=OSError("fail"),
    ):
        result = _run(tool.execute(EmptyArgs()))

    assert result.success is False
    assert result.error is not None


def test_execute_empty_workspace_list():
    tool = LaunchWorkspaceTool(workspace_apps=[])
    result = _run(tool.execute(EmptyArgs()))
    assert result.success is False
    assert "configured" in (result.error or "").lower()


def test_launch_entry_executable():
    entry = WorkspaceAppEntry(
        label="App", kind="executable", target=r"C:\test\app.exe",
    )
    with patch("jarvis.tools.local.launch_workspace.winplat.launch_path") as mock:
        with patch("pathlib.Path.is_file", return_value=True):
            with patch(
                "pathlib.Path.resolve",
                return_value=pathlib.Path(r"C:\test\app.exe"),
            ):
                _launch_entry(entry)
    mock.assert_called_once()


def test_launch_entry_shell():
    entry = WorkspaceAppEntry(
        label="Store", kind="shell", target=r"shell:AppsFolder\Foo",
    )
    with patch("jarvis.tools.local.launch_workspace.winplat.launch_shell") as mock:
        _launch_entry(entry)
    mock.assert_called_once_with(entry.target)


def test_launch_entry_installed_app():
    entry = WorkspaceAppEntry(label="Notepad", kind="installed_app", target="notepad")
    with patch(
        "jarvis.tools.local.launch_workspace._launch_installed_app",
    ) as mock:
        _launch_entry(entry)
    mock.assert_called_once_with("notepad")
