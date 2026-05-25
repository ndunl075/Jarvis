"""Verify dist/Jarvis contains all bundled assets. Exit 1 on failure."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "Jarvis"
INTERNAL = DIST / "_internal"


def _check(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"MISSING {label}: {path}")


def main() -> int:
    if not DIST.is_dir():
        raise SystemExit(f"dist folder not found: {DIST}")
    _check(DIST / "Jarvis.exe", "Jarvis.exe")
    for rel in (
        "_internal/openwakeword/resources/models/hey_jarvis_v0.1.onnx",
        "voices/en_GB-alan-medium.onnx",
        "voices/en_GB-alan-medium.onnx.json",
        "models/silero_vad.onnx",
        "models/whisper",
    ):
        _check(DIST / rel.replace("/", "\\") if "\\" in rel else DIST / rel, rel)
    print("Bundle verification OK:", DIST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
