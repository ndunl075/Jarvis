"""Persistence for deep research sessions (markdown report + JSON state)."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

SessionStatus = Literal["running", "paused", "completed", "failed"]


class CitedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[int] = Field(default_factory=list)


class DeepResearchSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sub_question: str
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    cited_claims: list[CitedClaim] = Field(default_factory=list)
    sources: list[dict[str, str]] = Field(default_factory=list)
    queries_used: list[str] = Field(default_factory=list)


class DeepResearchState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    status: SessionStatus = "running"
    created_at: str
    updated_at: str
    planner_model: str = ""
    worker_model: str = ""
    ultra_mode: bool = False
    sub_questions: list[str] = Field(default_factory=list)
    completed_indices: list[int] = Field(default_factory=list)
    sections: list[DeepResearchSection] = Field(default_factory=list)
    citations: list[dict[str, str]] = Field(default_factory=list)
    progress_message: str = ""
    error: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_research_root() -> Path:
    """``%APPDATA%/Jarvis/deep_research`` on Windows."""
    if sys.platform == "win32":
        appdata = __import__("os").environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Jarvis" / "deep_research"
    return Path.home() / ".jarvis" / "deep_research"


def _session_dir(session_id: str) -> Path:
    return deep_research_root() / session_id


def report_path(session_id: str) -> Path:
    return _session_dir(session_id) / "report.md"


def state_path(session_id: str) -> Path:
    return _session_dir(session_id) / "state.json"


def _slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[-\s]+", "-", s)
    return s[:48] or "topic"


def new_session_id(query: str) -> str:
    return f"{_slugify(query)}-{uuid4().hex[:8]}"


def create_session(query: str) -> DeepResearchState:
    """Create on-disk session folder and initial report header."""
    q = query.strip()
    if not q:
        raise ValueError("empty query")
    session_id = new_session_id(q)
    now = _now_iso()
    state = DeepResearchState(
        id=session_id,
        query=q,
        status="running",
        created_at=now,
        updated_at=now,
        progress_message="Starting deep research…",
    )
    root = deep_research_root()
    root.mkdir(parents=True, exist_ok=True)
    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Deep Research: {q}\n\n"
        f"- **Session:** `{session_id}`\n"
        f"- **Started:** {now}\n"
        f"- **Status:** running\n\n"
        "---\n\n"
        "## Report\n\n"
        "_Research in progress. Key points will appear below as each topic "
        "is completed._\n\n"
    )
    report_path(session_id).write_text(header, encoding="utf-8")
    save_state(state)
    return state


def load_state(session_id: str) -> DeepResearchState | None:
    path = state_path(session_id)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return DeepResearchState.model_validate(data)


def save_state(state: DeepResearchState) -> None:
    state.updated_at = _now_iso()
    path = state_path(state.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    _sync_status_in_report(state)


def read_report(session_id: str) -> str:
    path = report_path(session_id)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def append_report(session_id: str, text: str) -> None:
    path = report_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def write_report(session_id: str, content: str) -> None:
    report_path(session_id).write_text(content, encoding="utf-8")


def _sync_status_in_report(state: DeepResearchState) -> None:
    """Update the status line in the report header without wiping body."""
    path = report_path(state.id)
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if "**Status:**" in text:
        text = re.sub(
            r"\*\*Status:\*\* \w+",
            f"**Status:** {state.status}",
            text,
            count=1,
        )
    else:
        text = text.replace(
            "## Report\n\n",
            f"## Report\n\n**Status:** {state.status}\n\n",
            1,
        )
    path.write_text(text, encoding="utf-8")


def list_sessions() -> list[DeepResearchState]:
    root = deep_research_root()
    if not root.is_dir():
        return []
    out: list[DeepResearchState] = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        state = load_state(child.name)
        if state is not None:
            out.append(state)
    return out


def latest_paused_session() -> DeepResearchState | None:
    for state in list_sessions():
        if state.status == "paused":
            return state
    return None


def delete_session(session_id: str) -> bool:
    """Delete one session's folder. Returns True if a folder was removed."""
    import shutil

    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
        return False
    path = _session_dir(session_id)
    if not path.is_dir():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def delete_all_sessions() -> int:
    """Delete every session folder. Returns the count removed."""
    import shutil

    root = deep_research_root()
    if not root.is_dir():
        return 0
    n = 0
    for child in list(root.iterdir()):
        if not child.is_dir():
            continue
        shutil.rmtree(child, ignore_errors=True)
        if not child.exists():
            n += 1
    return n


def find_session_by_query(query: str) -> DeepResearchState | None:
    """Most recent session whose query contains ``query`` (case-insensitive)."""
    q = (query or "").strip().lower()
    if not q:
        return None
    for state in list_sessions():
        if q in state.query.lower():
            return state
    return None


def format_section_markdown(section: DeepResearchSection, index: int) -> str:
    lines = [f"### {index}. {section.sub_question}\n"]
    if section.summary:
        lines.append(section.summary.strip())
        lines.append("")
    if section.cited_claims:
        for claim in section.cited_claims:
            cite_text = ""
            if claim.citations:
                cite_text = " " + " ".join(f"[{c}]" for c in claim.citations)
            lines.append(f"- {claim.text.strip()}{cite_text}")
    else:
        for point in section.key_points:
            lines.append(f"- {point}")
    if section.queries_used:
        lines.append("")
        joined = ", ".join(f"`{q}`" for q in section.queries_used)
        lines.append(f"_Searched: {joined}_")
    lines.append("")
    return "\n".join(lines) + "\n"


def format_references_markdown(citations: list[dict[str, str]]) -> str:
    if not citations:
        return ""
    lines = ["## References\n"]
    for i, src in enumerate(citations, 1):
        title = src.get("title") or src.get("url", "")
        url = src.get("url", "")
        if url:
            lines.append(f"{i}. [{title}]({url})")
        else:
            lines.append(f"{i}. {title}")
    lines.append("")
    return "\n".join(lines) + "\n"
