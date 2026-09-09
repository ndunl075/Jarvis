"""Clipboard history — capped ring buffer of recent clipboard items,
with pinned entries that survive eviction.

Persistence layout::

    %APPDATA%/Jarvis/clipboard_history.json

    {
      "items": [
        {"text": "…", "ts": "2026-…", "pinned": false},
        …
      ]
    }

The polling thread lives in the UI panel (QTimer) — this module
provides pure data + filesystem layer.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


_MAX_ITEMS = 50
_PER_ITEM_CAP = 8000  # truncate individual entries; we're a quick-paste buffer
_PASSWORD_SENTINELS = ("password=", "secret=", "api_key=")


@dataclass
class ClipboardItem:
    text: str
    ts: str
    pinned: bool = False

    def preview(self, n: int = 120) -> str:
        s = self.text.replace("\n", " ").strip()
        return (s[:n] + "…") if len(s) > n else s


@dataclass
class ClipboardHistory:
    items: list[ClipboardItem] = field(default_factory=list)

    def add(self, text: str) -> bool:
        """Push a new entry to the front. Returns True if added."""
        if text is None:
            return False
        text = text.strip("\x00")
        if not text or not text.strip():
            return False
        if len(text) > _PER_ITEM_CAP:
            text = text[:_PER_ITEM_CAP]
        lower = text.lower()
        if any(s in lower for s in _PASSWORD_SENTINELS):
            log.debug("clipboard history: skipping likely-credential payload")
            return False
        if self.items and self.items[0].text == text:
            return False
        for i, existing in enumerate(self.items):
            if existing.text == text:
                self.items.pop(i)
                break
        self.items.insert(
            0, ClipboardItem(text=text, ts=_now_iso(), pinned=False)
        )
        self._evict()
        return True

    def pin(self, idx: int, pinned: bool = True) -> bool:
        if 0 <= idx < len(self.items):
            self.items[idx].pinned = pinned
            return True
        return False

    def remove(self, idx: int) -> bool:
        if 0 <= idx < len(self.items):
            self.items.pop(idx)
            return True
        return False

    def clear(self, *, keep_pinned: bool = True) -> int:
        before = len(self.items)
        if keep_pinned:
            self.items = [i for i in self.items if i.pinned]
        else:
            self.items = []
        return before - len(self.items)

    def _evict(self) -> None:
        if len(self.items) <= _MAX_ITEMS:
            return
        keep: list[ClipboardItem] = []
        unpinned: list[ClipboardItem] = []
        for it in self.items:
            (keep if it.pinned else unpinned).append(it)
        room = max(0, _MAX_ITEMS - len(keep))
        self.items = keep + unpinned[:room]

    def get(self, idx: int) -> ClipboardItem | None:
        if 0 <= idx < len(self.items):
            return self.items[idx]
        return None

    def to_dict(self) -> dict:
        return {
            "items": [
                {"text": i.text, "ts": i.ts, "pinned": i.pinned}
                for i in self.items
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClipboardHistory:
        out = cls()
        for entry in data.get("items", []) or []:
            text = entry.get("text", "")
            if not isinstance(text, str) or not text:
                continue
            out.items.append(
                ClipboardItem(
                    text=text,
                    ts=entry.get("ts") or _now_iso(),
                    pinned=bool(entry.get("pinned", False)),
                )
            )
        return out


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def history_path() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Jarvis" / "clipboard_history.json"
    return Path.home() / ".jarvis" / "clipboard_history.json"


def load_history() -> ClipboardHistory:
    p = history_path()
    if not p.is_file():
        return ClipboardHistory()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.warning("clipboard history file unreadable; starting fresh")
        return ClipboardHistory()
    return ClipboardHistory.from_dict(raw)


def save_history(history: ClipboardHistory) -> None:
    p = history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(history.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(p)
