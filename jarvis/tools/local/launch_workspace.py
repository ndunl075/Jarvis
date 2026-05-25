"""Launch the user's configured workspace apps in parallel."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from jarvis.core.config import WorkspaceAppEntry
from jarvis.platform import windows as winplat
from jarvis.tools.local.open_app import _APP_ALIASES, _needs_path_launch
from jarvis.tools.registry import EmptyArgs, ToolResult

log = logging.getLogger(__name__)


def _launch_installed_app(name: str) -> None:
    from jarvis.platform.windows_apps import resolve_installed_app

    token = name.strip()
    installed = resolve_installed_app(token)
    if installed is not None:
        cmd = installed.launch_command
        if _needs_path_launch(cmd):
            winplat.launch_path(cmd)
        else:
            winplat.launch_app(cmd)
        return

    canonical = _APP_ALIASES.get(token.lower())
    if canonical is not None:
        inst2 = resolve_installed_app(canonical)
        if inst2 is not None:
            cmd = inst2.launch_command
            if _needs_path_launch(cmd):
                winplat.launch_path(cmd)
            else:
                winplat.launch_app(cmd)
            return
        winplat.launch_app(canonical)
        return

    path = Path(token).expanduser()
    if path.is_file():
        winplat.launch_path(str(path.resolve()))
        return

    winplat.launch_app(token)


def _launch_entry(entry: WorkspaceAppEntry) -> None:
    if entry.kind == "executable":
        path = Path(entry.target).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{entry.label} not found at {path}")
        winplat.launch_path(str(path.resolve()))
    elif entry.kind == "shell":
        winplat.launch_shell(entry.target)
    elif entry.kind == "installed_app":
        _launch_installed_app(entry.target)
    else:
        raise ValueError(f"unknown workspace app kind: {entry.kind!r}")


def _workspace_description(apps: list[WorkspaceAppEntry]) -> str:
    if not apps:
        return (
            "Opens the user's workspace (apps configured in Settings → General). "
            "Call when the user says 'open my workspace', 'launch workspace', "
            "or 'start my setup'."
        )
    names = ", ".join(a.label for a in apps)
    return (
        f"Opens the user's workspace by launching: {names}. "
        "Call when the user says 'open my workspace', 'launch workspace', "
        "'start my workspace', or 'start my setup'."
    )


class LaunchWorkspaceTool:
    name: str = "launch_workspace"
    requires_confirmation: bool = False
    args_schema = EmptyArgs

    def __init__(self, *, workspace_apps: list[WorkspaceAppEntry] | None = None) -> None:
        self._apps: list[WorkspaceAppEntry] = list(workspace_apps or [])
        self.description = _workspace_description(self._apps)

    async def execute(self, args: EmptyArgs) -> ToolResult:  # noqa: ARG002
        if not self._apps:
            return ToolResult(
                success=False,
                error=(
                    "No workspace apps configured. "
                    "Add apps under Settings → General → Workspace."
                ),
            )

        results = await asyncio.gather(
            *(
                asyncio.to_thread(_launch_entry, entry)
                for entry in self._apps
            ),
            return_exceptions=True,
        )

        errors = [
            f"{self._apps[i].label}: {r}"
            for i, r in enumerate(results)
            if isinstance(r, BaseException)
        ]
        if errors:
            log.warning("launch_workspace: failures: %s", errors)
            if len(errors) == len(results):
                return ToolResult(success=False, error="; ".join(errors))

        labels = ", ".join(a.label for a in self._apps)
        return ToolResult(
            success=True,
            output=f"Opening your workspace, sir. {labels}.",
        )
