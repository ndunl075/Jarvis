"""Notes persistence — one markdown file per note under the data folder.

Layout::

    %APPDATA%/Jarvis/notes/
        <id>.md            <-- note body, prepended with YAML-ish front matter

Each file looks like::

    ---
    title: Meeting with Alex
    created: 2026-05-24T22:10:11+00:00
    updated: 2026-05-24T22:15:02+00:00
    ---

    Discussed launch plan and timeline.
    Follow-up: send proposal Friday.

This format is human-readable, hand-editable in any text editor, and avoids
a second JSON state file per note (the markdown IS the state).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Note(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    content: str = ""
    created_at: str
    updated_at: str

    @property
    def preview(self) -> str:
        text = self.content.strip().splitlines()
        for line in text:
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:80]
        return ""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def notes_root() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Jarvis" / "notes"
    return Path.home() / ".jarvis" / "notes"


def _note_path(note_id: str) -> Path:
    return notes_root() / f"{note_id}.md"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[-\s]+", "-", s)
    return s[:48] or "note"


def new_note_id(title: str) -> str:
    return f"{_slugify(title)}-{uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


_FRONT_MATTER_RE = re.compile(
    r"\A---\s*\n(?P<meta>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)


def _serialize(note: Note) -> str:
    safe_title = note.title.replace("\n", " ").strip()
    return (
        f"---\n"
        f"title: {safe_title}\n"
        f"created: {note.created_at}\n"
        f"updated: {note.updated_at}\n"
        f"---\n\n"
        f"{note.content.rstrip()}\n"
    )


def _parse(note_id: str, text: str) -> Note | None:
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        # Tolerate plain markdown without front matter — first non-empty line
        # becomes the title, rest is body.
        lines = text.splitlines()
        title = "Untitled"
        for ln in lines:
            stripped = ln.strip().lstrip("#").strip()
            if stripped:
                title = stripped[:80]
                break
        now = _now_iso()
        return Note(
            id=note_id,
            title=title,
            content=text.strip(),
            created_at=now,
            updated_at=now,
        )
    meta_block = m.group("meta") or ""
    body = m.group("body") or ""
    meta: dict[str, str] = {}
    for line in meta_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip()
    title = meta.get("title") or "Untitled"
    created = meta.get("created") or _now_iso()
    updated = meta.get("updated") or created
    return Note(
        id=note_id,
        title=title,
        content=body.strip(),
        created_at=created,
        updated_at=updated,
    )


# ---------------------------------------------------------------------------
# Public CRUD
# ---------------------------------------------------------------------------


def create_note(title: str, content: str = "") -> Note:
    title = (title or "").strip() or "Untitled note"
    note_id = new_note_id(title)
    now = _now_iso()
    note = Note(
        id=note_id,
        title=title,
        content=content.strip(),
        created_at=now,
        updated_at=now,
    )
    save_note(note)
    return note


def save_note(note: Note) -> None:
    note.updated_at = _now_iso()
    root = notes_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _note_path(note.id)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(_serialize(note), encoding="utf-8")
    tmp.replace(path)


def load_note(note_id: str) -> Note | None:
    path = _note_path(note_id)
    if not path.is_file():
        return None
    return _parse(note_id, path.read_text(encoding="utf-8"))


def list_notes() -> list[Note]:
    root = notes_root()
    if not root.is_dir():
        return []
    out: list[Note] = []
    for child in sorted(
        root.iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if not child.is_file() or child.suffix.lower() != ".md":
            continue
        note = load_note(child.stem)
        if note is not None:
            out.append(note)
    return out


def delete_note(note_id: str) -> bool:
    if not note_id or "/" in note_id or "\\" in note_id or ".." in note_id:
        return False
    path = _note_path(note_id)
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def delete_all_notes() -> int:
    root = notes_root()
    if not root.is_dir():
        return 0
    n = 0
    for child in list(root.iterdir()):
        if child.is_file() and child.suffix.lower() == ".md":
            try:
                child.unlink()
                n += 1
            except OSError:
                pass
    return n


def find_note_by_title(query: str) -> Note | None:
    """Most recent note whose title contains ``query`` (case-insensitive)."""
    q = (query or "").strip().lower()
    if not q:
        return None
    for note in list_notes():
        if q in note.title.lower():
            return note
    return None


def append_to_note(note: Note, text: str) -> Note:
    addition = text.strip()
    if not addition:
        return note
    sep = "\n\n" if note.content else ""
    note.content = f"{note.content}{sep}{addition}"
    save_note(note)
    return note


def ensure_root() -> Path:
    root = notes_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def export_all(target_dir: Path) -> int:
    """Copy every note .md into ``target_dir``. Returns the count copied."""
    root = notes_root()
    if not root.is_dir():
        return 0
    target_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for child in root.iterdir():
        if child.is_file() and child.suffix.lower() == ".md":
            shutil.copy2(child, target_dir / child.name)
            n += 1
    return n
