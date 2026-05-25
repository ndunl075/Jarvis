"""Launch a desktop application by name with fuzzy matching.

Resolution order (most reliable first):
  1. Fuzzy match against installed apps (Start Menu + App Paths) on the
     spoken token — yields a real .lnk / .exe path, which launches
     reliably. This comes FIRST because `start "<name>"` (steps 2/4)
     returns success even when Windows can't actually resolve the name
     (it pops an error dialog but the process exits 0), so a per-user
     install like Spotify or Discord would silently "succeed" while
     nothing opened.
  2. Alias table (chrome -> chrome, vscode -> code, …): the alias is a
     normalization hint. We re-resolve the canonical token against the
     installed index, then fall back to the bare command for
     App-Paths / PATH resolution.
  3. Direct path launch if the token is an existing file.
  4. Pass-through to Windows `start` for PATH / file associations.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarvis.core.request_context import current_user_transcription
from jarvis.platform import windows as winplat
from jarvis.platform.app_discovery import normalize_open_query
from jarvis.tools.registry import ToolResult

if TYPE_CHECKING:
    from jarvis.platform.windows_apps import InstalledApp

log = logging.getLogger(__name__)

_APP_ALIASES: dict[str, str] = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "firefox": "firefox",
    "spotify": "spotify",
    "vscode": "code",
    "visual studio code": "code",
    "code": "code",
    "notepad": "notepad",
    "calc": "calc",
    "calculator": "calc",
    "terminal": "wt",
    "windows terminal": "wt",
    "cmd": "cmd",
    "command prompt": "cmd",
    "explorer": "explorer",
    "file explorer": "explorer",
    "discord": "discord",
    "steam": "steam",
}


def _needs_path_launch(command: str) -> bool:
    low = command.lower()
    return (
        low.endswith((".lnk", ".exe", ".bat", ".cmd"))
        or "\\" in command
        or "/" in command
    )


def _resolve_name(raw: str) -> str:
    """Prefer a clean tool arg; fall back to the full voice utterance."""
    cleaned = normalize_open_query(raw)
    if cleaned and len(cleaned) >= 2:
        return cleaned
    utterance = current_user_transcription.get()
    if utterance:
        from_voice = normalize_open_query(utterance)
        if from_voice:
            log.info("open_app: derived name %r from utterance", from_voice)
            return from_voice
    return cleaned or raw.strip()


class OpenAppArgs(BaseModel):
    name: str = Field(
        description="Application name from the user's request (chrome, notepad, …)."
    )


class OpenAppTool:
    name: str = "open_app"
    description: str = (
        "Launches a desktop application. Always call this when the user "
        "asks to open, launch, or start an app — do not only describe the "
        "action in text. Resolves installed apps via fuzzy matching."
    )
    args_schema = OpenAppArgs
    requires_confirmation: bool = False

    def __init__(
        self,
        *,
        app_resolver: Callable[[str], InstalledApp | None] | None = None,
    ) -> None:
        self._app_resolver = app_resolver

    def _resolve_installed(self, query: str) -> InstalledApp | None:
        if self._app_resolver is not None:
            return self._app_resolver(query)
        from jarvis.platform.windows_apps import resolve_installed_app

        return resolve_installed_app(query)

    async def _safe_resolve(self, query: str) -> InstalledApp | None:
        """Resolve against the installed-app index, swallowing failures.

        The index build touches the registry + filesystem and raises
        NotImplementedError off Windows; any failure here just means "no
        installed match", never a tool crash."""
        try:
            return await asyncio.to_thread(self._resolve_installed, query)
        except Exception:
            log.debug("open_app: installed resolve failed for %r", query, exc_info=True)
            return None

    async def _try_launch(self, command: str) -> None:
        if _needs_path_launch(command):
            await asyncio.to_thread(winplat.launch_path, command)
        else:
            await asyncio.to_thread(winplat.launch_app, command)

    async def execute(self, args: OpenAppArgs) -> ToolResult:
        token = _resolve_name(args.name)
        if not token:
            return ToolResult(success=False, error="app name is empty")

        # (command, display) pairs, tried in order; first launch that
        # doesn't raise wins. Reliable real-path candidates come first.
        candidates: list[tuple[str, str]] = []

        # 1. Installed-app index on the spoken token (real .lnk / .exe).
        installed = await self._safe_resolve(token)
        if installed is not None:
            candidates.append((installed.launch_command, installed.display_name))

        # 2. Alias canonical token. Re-resolve the canonical in the index
        #    (e.g. "vscode" -> "code" -> Visual Studio Code.lnk), then
        #    fall back to the bare command for App-Paths / PATH.
        canonical = _APP_ALIASES.get(token.lower())
        if canonical is not None:
            if canonical.lower() != token.lower():
                inst2 = await self._safe_resolve(canonical)
                if inst2 is not None:
                    candidates.append((inst2.launch_command, inst2.display_name))
            candidates.append((canonical, token))

        # 3. Direct path launch if the token is a real file.
        path = Path(token).expanduser()
        if path.is_file():
            candidates.append((str(path.resolve()), token))

        # 4. Last resort: hand the raw token to Windows `start`.
        candidates.append((token, token))

        seen: set[str] = set()
        last_error: str | None = None
        for command, display in candidates:
            if not command or command in seen:
                continue
            seen.add(command)
            try:
                await self._try_launch(command)
                return ToolResult(success=True, output=f"Opening {display}, sir.")
            except NotImplementedError as e:
                return ToolResult(success=False, error=str(e))
            except OSError as e:
                last_error = str(e)
                log.warning("open_app: launch failed for %r: %s", command, e)

        return ToolResult(
            success=False,
            error=f"could not launch {token!r}"
            + (f": {last_error}" if last_error else ""),
        )
