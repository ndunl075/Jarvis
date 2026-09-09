from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path


def _frozen_log_path() -> Path:
    appdata = os.environ.get("APPDATA") or str(Path.home())
    return Path(appdata) / "Jarvis" / "logs" / "jarvis.log"


def _write_crash_log(exc: BaseException) -> Path:
    path = _frozen_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n--- crash ---\n")
        traceback.print_exception(exc, file=f)
    return path


def _show_error_box(message: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
            None, message, "Jarvis", 0x10,
        )
    except Exception:
        pass


def main() -> None:
    from jarvis.paths import is_frozen

    if is_frozen():
        log_path = _frozen_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    try:
        from jarvis.app import run

        raise SystemExit(run())
    except Exception as exc:
        log_file = _write_crash_log(exc)
        if is_frozen():
            _show_error_box(
                f"Jarvis failed to start:\n\n{exc}\n\nDetails: {log_file}"
            )
        raise


if __name__ == "__main__":
    main()
