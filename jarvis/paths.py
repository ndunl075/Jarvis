"""Install and bundle paths for dev runs vs PyInstaller builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_root() -> Path:
    """Directory containing Jarvis.exe (one-folder build) or the repo root."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundle_root() -> Path:
    """PyInstaller _MEIPASS when frozen; else repo root."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", install_root()))
    return install_root()


def _asset_roots() -> tuple[Path, ...]:
    """Search order for bundled ML assets when frozen."""
    if is_frozen():
        return (install_root(), bundle_root())
    return (install_root(),)


def _first_existing_dir(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return path
    return None


def _first_existing_file(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def default_voices_dir() -> Path:
    env = os.environ.get("JARVIS_PIPER_VOICES_DIR")
    if env:
        return Path(env)
    candidates = [root / "voices" for root in _asset_roots()]
    hit = _first_existing_dir(*candidates)
    if hit is not None:
        return hit
    return Path.home() / ".jarvis" / "voices"


def default_whisper_download_root() -> Path | None:
    candidates = [root / "models" / "whisper" for root in _asset_roots()]
    return _first_existing_dir(*candidates)


def default_silero_onnx_path() -> Path | None:
    candidates = [root / "models" / "silero_vad.onnx" for root in _asset_roots()]
    return _first_existing_file(*candidates)


def bundled_asset_report() -> dict[str, str]:
    """Human-readable paths for startup diagnostics."""
    return {
        "frozen": str(is_frozen()),
        "install_root": str(install_root()),
        "bundle_root": str(bundle_root()),
        "voices_dir": str(default_voices_dir()),
        "whisper_root": str(default_whisper_download_root() or "<missing>"),
        "silero_onnx": str(default_silero_onnx_path() or "<missing>"),
    }
