"""PyInstaller runtime hook — runs before jarvis imports.

Sets DLL search paths, Qt plugin path, and optional boot logging.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _frozen_base() -> Path | None:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return None


def _apply() -> None:
    base = _frozen_base()
    if base is None:
        return
    # Help Windows find bundled DLLs (onnxruntime, ctranslate2, PortAudio, Qt).
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(base))
        except OSError:
            pass
    for sub in ("onnxruntime/capi", "ctranslate2", "PySide6"):
        dll_dir = base / sub
        if dll_dir.is_dir() and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(dll_dir))
            except OSError:
                pass
    plugins = base / "PySide6" / "plugins"
    if plugins.is_dir():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))
    qml = base / "PySide6" / "qml"
    if qml.is_dir():
        os.environ.setdefault("QML2_IMPORT_PATH", str(qml))


_apply()
