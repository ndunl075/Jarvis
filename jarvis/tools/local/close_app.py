"""Close / quit a running desktop application by name with fuzzy matching.

Matches the spoken name against running process names (psutil), then
terminates every process in the matched group (apps like Chrome spawn
many processes under one name). Falls back to a hard kill for any that
ignore the graceful terminate.

Protected processes (the OS shell, core services, and Jarvis's own
Python process) are never matched, so a stray "close system" or "close
python" can't take down the desktop or the assistant itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from jarvis.core.request_context import current_user_transcription
from jarvis.platform.app_discovery import fuzzy_resolve, normalize_open_query, normalize_query
from jarvis.tools.local.open_app import _APP_ALIASES
from jarvis.tools.registry import ToolResult

log = logging.getLogger(__name__)

# Normalized process-name stems that must never be matched/killed. Killing
# any of these either crashes the desktop shell or kills Jarvis itself.
_PROTECTED_STEMS: frozenset[str] = frozenset({
    "system", "idle", "registry", "smss", "csrss", "wininit", "winlogon",
    "services", "lsass", "lsm", "svchost", "explorer", "dwm", "ctfmon",
    "fontdrvhost", "python", "pythonw", "conhost", "runtimebroker",
    "shellexperiencehost", "startmenuexperiencehost", "searchhost",
})


@dataclass(frozen=True, slots=True)
class RunningProcess:
    pid: int
    name: str


def _default_list_processes() -> list[RunningProcess]:
    import psutil

    out: list[RunningProcess] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = proc.info.get("name")
            pid = proc.info.get("pid")
        except Exception:
            continue
        if name and isinstance(pid, int):
            out.append(RunningProcess(pid=pid, name=name))
    return out


def _default_kill(pids: list[int]) -> None:
    import psutil

    procs: list[psutil.Process] = []
    for pid in pids:
        try:
            procs.append(psutil.Process(pid))
        except psutil.NoSuchProcess:
            continue
    for p in procs:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # Hard-kill anything still alive after a short grace period.
    _gone, alive = psutil.wait_procs(procs, timeout=2.0)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _resolve_name(raw: str) -> str:
    """Prefer a clean tool arg; fall back to the full voice utterance."""
    cleaned = normalize_open_query(raw)
    if cleaned and len(cleaned) >= 2:
        return cleaned
    utterance = current_user_transcription.get()
    if utterance:
        from_voice = normalize_open_query(utterance)
        if from_voice:
            return from_voice
    return cleaned or raw.strip()


class CloseAppArgs(BaseModel):
    name: str = Field(
        description="Application/program name to close (chrome, spotify, notepad, …)."
    )


class CloseAppTool:
    name: str = "close_app"
    description: str = (
        "Closes, quits, or kills a running desktop application. Call this "
        "when the user asks to close, quit, exit, stop, or kill an app or "
        "program (e.g. 'close Chrome', 'quit Spotify'). Matches the "
        "running process by name; do not use it for closing files or tabs."
    )
    args_schema = CloseAppArgs
    requires_confirmation: bool = False

    def __init__(
        self,
        *,
        process_lister: Callable[[], list[RunningProcess]] | None = None,
        process_killer: Callable[[list[int]], None] | None = None,
    ) -> None:
        self._list_processes = process_lister or _default_list_processes
        self._kill = process_killer or _default_kill

    async def execute(self, args: CloseAppArgs) -> ToolResult:
        token = _resolve_name(args.name)
        if not token:
            return ToolResult(success=False, error="app name is empty")

        try:
            processes = await asyncio.to_thread(self._list_processes)
        except Exception as e:
            return ToolResult(success=False, error=f"could not list processes: {e}")

        own_pid = os.getpid()
        # Group live PIDs by normalized process-name stem, skipping
        # protected processes and Jarvis's own process.
        groups: dict[str, list[int]] = {}
        display: dict[str, str] = {}
        for proc in processes:
            if proc.pid == own_pid:
                continue
            stem = Path(proc.name).stem
            key = normalize_query(stem)
            if not key or key in _PROTECTED_STEMS:
                continue
            groups.setdefault(key, []).append(proc.pid)
            display.setdefault(key, stem)

        # Try the spoken token, then the alias canonical (vscode -> code,
        # so a "close vscode" matches the "Code" process).
        queries = [token]
        canonical = _APP_ALIASES.get(token.lower())
        if canonical is not None and canonical.lower() != token.lower():
            queries.append(canonical)

        hit: tuple[str, list[int]] | None = None
        for query in queries:
            hit = fuzzy_resolve(query, groups)
            if hit is not None:
                break

        if hit is None:
            return ToolResult(
                success=False,
                error=f"I don't see {token} running, sir.",
            )

        key, pids = hit
        name = display.get(key, key)
        try:
            await asyncio.to_thread(self._kill, list(pids))
        except Exception as e:
            log.warning("close_app: kill failed for %r: %s", name, e)
            return ToolResult(success=False, error=f"could not close {name}: {e}")

        return ToolResult(success=True, output=f"Closing {name.title()}, sir.")
