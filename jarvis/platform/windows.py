"""Windows-specific OS primitives used by jarvis/tools/local/*.

Every Windows-specific call lives here so tool implementations stay pure
Python and testable by patching this module. Tools never import ctypes,
windll, subprocess, or webbrowser directly; they go through these
functions and tests patch them.

Functions raise OSError on Windows-API failures so callers can wrap the
message into a ToolResult.error. Non-Windows hosts get NotImplementedError
from the screen / volume / lock primitives — those tools simply return a
ToolResult error on Linux/macOS dev machines."""

from __future__ import annotations

import ctypes
import ntpath
import os
import subprocess
import sys
import webbrowser
from ctypes import wintypes
from pathlib import Path

# -- platform check -----------------------------------------------------


def _require_windows(feature: str) -> None:
    if sys.platform != "win32":
        raise NotImplementedError(
            f"{feature} is Windows-only; current platform: {sys.platform}"
        )


# -- browser / shell ----------------------------------------------------


def open_url(url: str) -> None:
    """Open `url` in the default browser. Wraps webbrowser.open so tests
    can patch this single seam instead of monkey-patching stdlib."""
    webbrowser.open(url)


# Windows has no argv: subprocess joins the list back into one command
# line (subprocess.list2cmdline) and the launched program re-parses it.
# list2cmdline quotes only arguments containing spaces or tabs, so when the
# launched program was cmd.exe its metacharacters passed through raw —
# `cmd /c start "" notepad&calc` is two commands, `shell=False` or not.
# These launchers therefore never invoke cmd.exe: os.startfile hands the
# target to ShellExecuteW as one opaque string, which is the same API
# `start` ultimately calls, so name resolution is unchanged. The validators
# below are the second line of defence.

# cmd.exe metacharacters. Strict set, for bare command tokens only.
_COMMAND_METACHARACTERS = frozenset('&|<>^"%!')

# Characters Windows already forbids in a path, and that no shell: URI
# contains. The strict set above would be wrong for paths — see
# validate_launch_target_path.
_PATH_ILLEGAL_CHARACTERS = frozenset('<>"|')


def _reject_unsafe(value: str, illegal: frozenset[str], *, kind: str) -> None:
    """Raise ValueError if `value` holds characters a launcher must not
    forward. Control characters (CR/LF and NUL among them) are rejected for
    every kind: they are illegal in Windows paths, and a newline is the
    plainest way to smuggle a second line into anything that later parses a
    command line."""
    if not value.strip():
        raise ValueError(f"{kind} is empty")
    bad = sorted({c for c in value if c in illegal or ord(c) < 0x20 or ord(c) == 0x7F})
    if bad:
        raise ValueError(
            f"{kind} contains disallowed characters: " + " ".join(repr(c) for c in bad)
        )


def validate_launch_command(command: str) -> None:
    """Validate a bare app name / command token ('chrome', 'msedge').

    Strict, because this is the launcher input that carries untrusted text:
    open_app's last-resort candidate is the LLM's tool argument or the raw
    voice transcription, normalized only for filler words. Nothing
    legitimate arriving here — an executable name resolved through App
    Paths or PATH — needs a shell metacharacter."""
    _reject_unsafe(command, _COMMAND_METACHARACTERS, kind="launch command")


def validate_launch_target_path(path: str) -> None:
    """Validate a filesystem path or shell: namespace URI.

    Deliberately narrower than validate_launch_command: it rejects only
    what Windows already forbids in a path. Legitimate targets do contain
    command metacharacters — Store-app entries end in `!App`
    (shell:appsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App), App
    Paths values can be %ProgramFiles%\\..., and installed apps have names
    like "Dungeons & Dragons" — so the strict set would stop real apps from
    launching. They are safe here because the target no longer passes
    through a command line."""
    _reject_unsafe(path, _PATH_ILLEGAL_CHARACTERS, kind="launch path")


def _shell_execute(target: str) -> None:
    """Open `target` via ShellExecuteW, with no shell in the loop.

    os.startfile passes the string straight to ShellExecuteW, which resolves
    it exactly as `start` did — file associations (.lnk, documents),
    protocol handlers (steam://), the shell: namespace, and the App Paths
    registry / PATH lookup for bare names — but never parses it as a
    command line. Its own function so tests can patch this one seam;
    os.startfile exists only on Windows."""
    os.startfile(target)  # type: ignore[attr-defined]  # Windows-only


def launch_app(command: str) -> None:
    """Launch an app by bare name, so Windows resolves names like 'chrome'
    or 'spotify' without us knowing their install path.

    Was `cmd /c start "" <command>` (the leading "" being start's
    window-title slot). ShellExecuteW keeps that resolution — `start` is a
    wrapper around it — and drops the cmd.exe re-parse that let a token
    like `notepad&calc` run two commands. Raises OSError when Windows
    cannot resolve the name, which `start` did not; callers already treat
    OSError as "try the next candidate". Tests patch _shell_execute."""
    _require_windows("launch_app")
    validate_launch_command(command)
    _shell_execute(command)


def launch_path(path: str) -> None:
    """Launch a file, shortcut or shell: URI by target (.lnk, .exe,
    shell:appsFolder\\<AppUserModelID>, ...)."""
    _require_windows("launch_path")
    validate_launch_target_path(path)
    # `cmd /c start` expanded %VAR% before ShellExecuteW ever saw the
    # target, and App Paths registry values are sometimes REG_EXPAND_SZ
    # (%ProgramFiles%\...), so keep that resolution behaviour. ntpath
    # rather than os.path so the %VAR% form expands identically on the
    # Linux hosts that run the tests. Pure string substitution — the
    # expansion inserts a value, it cannot introduce a command.
    _shell_execute(ntpath.expandvars(path))


def launch_shell(target: str) -> None:
    """Open a shell namespace URI (e.g. shell:AppsFolder\\... for Store apps).

    explorer.exe, not cmd.exe: it takes the URI as a single argument and
    never treats `&` or `|` as a separator, so this function was not part of
    the command-injection issue and keeps its subprocess call. The
    validation is only defence in depth, and uses the path rule because
    Store-app URIs legitimately contain `!`."""
    _require_windows("launch_shell")
    validate_launch_target_path(target)
    subprocess.Popen(
        ["explorer.exe", target],
        close_fds=True,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
    )


def launch_steam_game(app_id: int) -> None:
    """Start a Steam library title via the steam:// protocol handler.

    app_id is an int, so this never carried injectable text; it goes through
    ShellExecuteW for consistency with the other launchers."""
    _require_windows("launch_steam_game")
    _shell_execute(f"steam://run/{int(app_id)}")


# -- screenshot ---------------------------------------------------------


def screenshots_dir() -> Path:
    """User-facing screenshot directory; created if absent."""
    base = Path.home() / "Pictures" / "Jarvis_Screenshots"
    base.mkdir(parents=True, exist_ok=True)
    return base


# -- lock screen --------------------------------------------------------


def lock_screen() -> None:
    """Lock the workstation (Win+L equivalent). Raises OSError on failure."""
    _require_windows("lock_screen")
    if not ctypes.windll.user32.LockWorkStation():
        raise OSError("LockWorkStation returned 0")


# -- volume (media keys) -----------------------------------------------

# Using keybd_event with the media-key virtual key codes avoids depending
# on pycaw / Core Audio APIs. Granularity is one Windows volume "tick"
# (~2%); we step `amount` ticks to approximate larger jumps.

_VK_VOLUME_MUTE = 0xAD
_VK_VOLUME_DOWN = 0xAE
_VK_VOLUME_UP = 0xAF
_KEYEVENTF_KEYUP = 0x0002


def _tap_media_key(vk: int) -> None:
    _require_windows("volume control")
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)


def volume_up(ticks: int = 1) -> None:
    for _ in range(max(1, ticks)):
        _tap_media_key(_VK_VOLUME_UP)


def volume_down(ticks: int = 1) -> None:
    for _ in range(max(1, ticks)):
        _tap_media_key(_VK_VOLUME_DOWN)


def volume_mute_toggle() -> None:
    """The mute key is a toggle; "mute" and "unmute" both tap it. Callers
    are responsible for knowing the current state if they need it."""
    _tap_media_key(_VK_VOLUME_MUTE)


# -- clipboard ----------------------------------------------------------

_CF_UNICODETEXT = 13


def read_clipboard_text() -> str:
    """Read the current clipboard as text. Returns empty string if the
    clipboard is empty or holds a non-text format. Raises OSError on
    Windows API failures (OpenClipboard contention etc.)."""
    _require_windows("clipboard")
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalUnlock.restype = wintypes.BOOL

    if not user32.OpenClipboard(None):
        raise OSError("OpenClipboard failed")
    try:
        h = user32.GetClipboardData(_CF_UNICODETEXT)
        if not h:
            return ""
        ptr = kernel32.GlobalLock(h)
        if not ptr:
            return ""
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()


# Win32 GlobalAlloc / EmptyClipboard / SetClipboardData constants
_GMEM_MOVEABLE = 0x0002


def write_clipboard_text(text: str) -> None:
    """Write ``text`` to the system clipboard as CF_UNICODETEXT. Raises
    OSError on Windows API failures (clipboard contention etc.). Used by
    the clipboard history panel to paste a previous item back."""
    _require_windows("clipboard")
    if text is None:
        text = ""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalUnlock.restype = wintypes.BOOL

    data = text.encode("utf-16-le") + b"\x00\x00"
    h_mem = kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(data))
    if not h_mem:
        raise OSError("GlobalAlloc failed")
    ptr = kernel32.GlobalLock(h_mem)
    if not ptr:
        raise OSError("GlobalLock failed")
    try:
        ctypes.memmove(ptr, data, len(data))
    finally:
        kernel32.GlobalUnlock(h_mem)

    if not user32.OpenClipboard(None):
        raise OSError("OpenClipboard failed")
    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(_CF_UNICODETEXT, h_mem):
            raise OSError("SetClipboardData failed")
    finally:
        user32.CloseClipboard()


__all__ = [
    "launch_app",
    "launch_path",
    "launch_shell",
    "launch_steam_game",
    "lock_screen",
    "open_url",
    "read_clipboard_text",
    "screenshots_dir",
    "validate_launch_command",
    "validate_launch_target_path",
    "volume_down",
    "volume_mute_toggle",
    "volume_up",
    "write_clipboard_text",
]
