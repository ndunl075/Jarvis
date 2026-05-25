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


def launch_app(command: str) -> None:
    """Launch an app by command line. Uses `cmd /c start` so Windows file
    associations resolve names like 'chrome' or 'spotify' without us
    knowing their install path.

    `command` is passed to start as a single argument (the leading "" is
    start's required window-title slot). Tests patch subprocess.Popen."""
    _require_windows("launch_app")
    subprocess.Popen(
        ["cmd", "/c", "start", "", command],
        shell=False,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
    )


def launch_path(path: str) -> None:
    """Launch a file or shortcut by full path (.lnk, .exe, etc.)."""
    _require_windows("launch_path")
    subprocess.Popen(
        ["cmd", "/c", "start", "", path],
        shell=False,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
    )


def launch_shell(target: str) -> None:
    """Open a shell namespace URI (e.g. shell:AppsFolder\\... for Store apps)."""
    _require_windows("launch_shell")
    subprocess.Popen(
        ["explorer.exe", target],
        close_fds=True,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
    )


def launch_steam_game(app_id: int) -> None:
    """Start a Steam library title via the steam:// protocol handler."""
    _require_windows("launch_steam_game")
    subprocess.Popen(
        ["cmd", "/c", "start", "", f"steam://run/{app_id}"],
        shell=False,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
    )


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
    "launch_steam_game",
    "lock_screen",
    "open_url",
    "read_clipboard_text",
    "screenshots_dir",
    "volume_down",
    "volume_mute_toggle",
    "volume_up",
    "write_clipboard_text",
]
