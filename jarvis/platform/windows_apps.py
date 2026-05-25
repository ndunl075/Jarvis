"""Windows application and Steam game discovery.

Builds searchable indexes for fuzzy launch. All filesystem and registry
access lives here; tools call these helpers through thin wrappers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from jarvis.platform.app_discovery import fuzzy_resolve, normalize_query

log = logging.getLogger(__name__)

_APP_PATHS_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
_STEAM_KEY = r"Software\Valve\Steam"
_INDEX_CACHE_TTL_SECONDS = 300.0
# Apps launched from the Windows "apps folder" namespace (Start menu's
# unified list). `start "" shell:appsFolder\<AppUserModelID>` opens any
# entry Get-StartApps reports, including UWP/Store apps that have no
# .lnk or App Paths registry entry.
_APPS_FOLDER_PREFIX = "shell:appsFolder\\"
_GET_START_APPS_TIMEOUT = 15.0
_installed_index_cache: dict[str, InstalledApp] | None = None
_installed_index_cached_at: float = 0.0


@dataclass(frozen=True, slots=True)
class InstalledApp:
    """A launchable desktop application."""

    display_name: str
    launch_command: str  # passed to launch_app or launch_path


@dataclass(frozen=True, slots=True)
class SteamGame:
    name: str
    app_id: int


def _require_windows() -> None:
    if sys.platform != "win32":
        raise NotImplementedError(
            f"Windows app discovery is Windows-only; current: {sys.platform}"
        )


def _read_steam_install_dir() -> Path | None:
    _require_windows()
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _STEAM_KEY) as key:
            raw, _ = winreg.QueryValueEx(key, "SteamPath")
    except OSError:
        return None
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_dir() else None


def _parse_vdf_paths(text: str) -> list[str]:
    """Extract quoted path values from a Steam VDF file."""
    return [
        m.group(1).replace("\\\\", "\\")
        for m in re.finditer(r'"path"\s+"([^"]+)"', text)
    ]


def _steam_library_dirs(steam_dir: Path) -> list[Path]:
    """Return every steamapps directory (main + library folders)."""
    dirs: list[Path] = []
    main = steam_dir / "steamapps"
    if main.is_dir():
        dirs.append(main)
    vdf = main / "libraryfolders.vdf"
    if not vdf.is_file():
        return dirs
    try:
        text = vdf.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return dirs
    for p in _parse_vdf_paths(text):
        lib = Path(p) / "steamapps"
        if lib.is_dir() and lib not in dirs:
            dirs.append(lib)
    return dirs


def _parse_appmanifest(path: Path) -> SteamGame | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    appid_m = re.search(r'"appid"\s+"(\d+)"', text)
    name_m = re.search(r'"name"\s+"([^"]+)"', text)
    if not appid_m or not name_m:
        return None
    return SteamGame(name=name_m.group(1), app_id=int(appid_m.group(1)))


def scan_steam_games(steam_dir: Path | None = None) -> dict[str, SteamGame]:
    """Map normalized game title -> SteamGame for installed library titles."""
    if steam_dir is None:
        _require_windows()
        steam_dir = _read_steam_install_dir()
    if steam_dir is None:
        return {}
    games: dict[str, SteamGame] = {}
    for lib in _steam_library_dirs(steam_dir):
        for manifest in lib.glob("appmanifest_*.acf"):
            game = _parse_appmanifest(manifest)
            if game is None:
                continue
            key = normalize_query(game.name)
            if key and key not in games:
                games[key] = game
    return games


def _start_menu_roots() -> list[Path]:
    roots: list[Path] = []
    program_data = os.environ.get("ProgramData", "")
    appdata = os.environ.get("APPDATA", "")
    if program_data:
        roots.append(
            Path(program_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        )
    if appdata:
        roots.append(
            Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        )
    return [r for r in roots if r.is_dir()]


def _scan_shortcuts(roots: list[Path]) -> dict[str, InstalledApp]:
    """Index .lnk files by normalized shortcut stem."""
    apps: dict[str, InstalledApp] = {}
    for root in roots:
        for lnk in root.rglob("*.lnk"):
            stem = lnk.stem
            key = normalize_query(stem)
            if not key or key in apps:
                continue
            apps[key] = InstalledApp(
                display_name=stem,
                launch_command=str(lnk.resolve()),
            )
    return apps


def _scan_app_paths_registry() -> dict[str, InstalledApp]:
    _require_windows()
    import winreg

    apps: dict[str, InstalledApp] = {}
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _APP_PATHS_KEY) as root:
            i = 0
            while True:
                try:
                    sub_name = winreg.EnumKey(root, i)
                    i += 1
                except OSError:
                    break
                try:
                    with winreg.OpenKey(root, sub_name) as sub:
                        exe, _ = winreg.QueryValueEx(sub, None)
                except OSError:
                    continue
                if not exe:
                    continue
                token = Path(sub_name).stem
                key = normalize_query(token)
                if not key or key in apps:
                    continue
                apps[key] = InstalledApp(
                    display_name=token,
                    launch_command=str(exe),
                )
    except OSError as e:
        log.debug("App Paths registry scan failed: %s", e)
    return apps


def _scan_start_apps() -> dict[str, InstalledApp]:
    """Enumerate every app in the Start menu's unified list via PowerShell
    `Get-StartApps`. This is the only source that includes UWP / Microsoft
    Store apps (Spotify, Calculator, Photos, …) and modern installs that
    register no .lnk or App Paths entry. Each entry launches through the
    apps-folder shell namespace, exactly as the Start menu does.

    Best-effort: a PowerShell failure / timeout / parse error yields {}."""
    apps: dict[str, InstalledApp] = {}
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-StartApps | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=_GET_START_APPS_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("Get-StartApps invocation failed: %s", e)
        return apps
    out = completed.stdout.strip()
    if not out:
        return apps
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        log.debug("Get-StartApps JSON parse failed: %s", e)
        return apps
    # ConvertTo-Json emits a bare object for a single result, a list otherwise.
    if isinstance(data, dict):
        data = [data]
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("Name", "")).strip()
        app_id = str(entry.get("AppID", "")).strip()
        if not name or not app_id:
            continue
        key = normalize_query(name)
        if not key or key in apps:
            continue
        apps[key] = InstalledApp(
            display_name=name,
            launch_command=_APPS_FOLDER_PREFIX + app_id,
        )
    return apps


def build_installed_app_index(*, force_refresh: bool = False) -> dict[str, InstalledApp]:
    """Merge App Paths registry, Start Menu shortcuts, and the Start-apps
    list (UWP/Store + everything else).

    Precedence on duplicate normalized names is by source order below:
    direct-path sources (registry, .lnk) win over the apps-folder launch,
    since a real path is marginally faster to start; the Start-apps scan
    fills in everything the first two miss."""
    global _installed_index_cache, _installed_index_cached_at
    _require_windows()
    now = time.monotonic()
    if (
        not force_refresh
        and _installed_index_cache is not None
        and now - _installed_index_cached_at < _INDEX_CACHE_TTL_SECONDS
    ):
        return _installed_index_cache
    merged: dict[str, InstalledApp] = {}
    for batch in (
        _scan_app_paths_registry(),
        _scan_shortcuts(_start_menu_roots()),
        _scan_start_apps(),
    ):
        for key, app in batch.items():
            if key not in merged:
                merged[key] = app
    _installed_index_cache = merged
    _installed_index_cached_at = now
    return merged


def resolve_installed_app(
    query: str,
    index: dict[str, InstalledApp] | None = None,
) -> InstalledApp | None:
    if index is None:
        index = build_installed_app_index()
    hit = fuzzy_resolve(query, index)
    return hit[1] if hit else None


def resolve_steam_game(
    query: str,
    games: dict[str, SteamGame] | None = None,
) -> SteamGame | None:
    if games is None:
        games = scan_steam_games()
    hit = fuzzy_resolve(query, games)
    return hit[1] if hit else None
