"""Research results panel — slides in from the right edge of the primary screen.

Threading model
---------------
ResearchWorker (QThread) fetches web snippets (DuckDuckGo) and summarizes via
local Ollama. It emits:
  chunk_received(str)      — one token at a time (→ panel text area)
  finished(str, object)    — (full_text, sources: list[dict]) on success
  failed(str)              — error message on failure

FollowUpWorker (QThread) generates 3 follow-up questions after research
finishes. It emits questions_ready(list) on success; failures are silently
logged (follow-ups are decorative).

ResearchTool (audio asyncio loop) calls panel.show_for_query(query,
result_queue).  This queues _on_start to run on the Qt main thread, which
creates and starts a ResearchWorker.  When the worker finishes it puts
(summary, sources) into result_queue so the awaiting tool coroutine can
return a ToolResult for TTS without blocking the audio loop.

History: last _MAX_HISTORY queries are stored in-memory. The ≡ button in the
header toggles a history list view; clicking any entry re-runs that query.

Source cards fetch favicons off-thread via _FaviconThread (one per card,
uses urllib — no extra deps).  If the fetch fails the card simply shows
without a favicon.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import webbrowser
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

_DEFAULT_PANEL_WIDTH = 420
_MIN_PANEL_WIDTH = 280
_MAX_PANEL_WIDTH = 700
_RESIZE_HIT_PX = 8
_SIDE_MARGIN = 16
_ANIM_MS = 300
_MAX_HISTORY = 10

_BG          = "#0d0d0d"
_BG_CARD     = "#141414"
_BORDER      = "#1f1f1f"
_CHIP_BORDER = "#2a2a2a"
_TEXT        = "#ffffff"
_TEXT_DIM    = "#808080"
_TEXT_MID    = "#b0b0b0"
_CYAN        = "#38f4ff"

_URL_RE = re.compile(r"https?://[^\s\)\"\']+")

# Header button base style (shared by COPY and ≡)
_HDR_BTN = (
    f"QPushButton {{ color:{_TEXT_DIM}; background:transparent; border:none; "
    "font-size:8pt; letter-spacing:2px; font-weight:300; padding:0 6px; }}"
    f"QPushButton:hover {{ color:{_TEXT}; }}"
)
_HDR_BTN_ACTIVE = (
    f"QPushButton {{ color:{_CYAN}; background:transparent; border:none; "
    "font-size:8pt; letter-spacing:2px; font-weight:300; padding:0 6px; }}"
    f"QPushButton:hover {{ color:{_TEXT}; }}"
)


# ---------------------------------------------------------------------------
# Source extraction helper (used by ResearchWorker)
# ---------------------------------------------------------------------------


def _extract_sources(message: Any) -> list[dict]:
    """Pull source dicts from an Anthropic response message.

    Tries structured web_search_tool_result blocks first; falls back to
    regex URL extraction from text blocks.
    """
    sources: list[dict] = []
    seen: set[str] = set()
    text_blocks: list[str] = []

    try:
        for block in getattr(message, "content", []):
            btype = getattr(block, "type", "")

            if btype in ("web_search_tool_result", "server_tool_result"):
                for result in getattr(block, "content", []):
                    url = getattr(result, "url", None)
                    if url and url not in seen:
                        seen.add(url)
                        sources.append({
                            "title": getattr(result, "title", url),
                            "url": url,
                        })

            elif btype == "tool_use":
                inp = getattr(block, "input", {}) or {}
                for v in inp.values():
                    if isinstance(v, str):
                        text_blocks.append(v)

            elif btype == "text":
                text_blocks.append(getattr(block, "text", ""))
    except Exception:
        log.debug("source extraction failed", exc_info=True)

    if not sources:
        for text in text_blocks:
            for url in _URL_RE.findall(text):
                url = url.rstrip(".,;:")
                if url not in seen:
                    seen.add(url)
                    sources.append({"title": url, "url": url})

    return sources[:6]


# ---------------------------------------------------------------------------
# QThread workers
# ---------------------------------------------------------------------------


class ResearchWorker(QThread):
    """DuckDuckGo snippets + local Ollama summarization in a dedicated thread."""

    chunk_received = Signal(str)
    finished       = Signal(str, object)  # (full_text, sources: list[dict])
    failed         = Signal(str)

    _MAX_TOKENS = 1024
    _SYSTEM = (
        "You are a research assistant. Using ONLY the web search excerpts below, "
        "write a concise summary with 3-5 bullet points. Be direct and factual. "
        "Format bullet points with • prefix. Mention source numbers [1], [2], etc. "
        "when relevant. If the excerpts are insufficient, say what is missing."
    )

    def __init__(
        self,
        query: str,
        *,
        ollama_endpoint: str,
        ollama_model: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._query = query
        self._ollama_endpoint = ollama_endpoint
        self._ollama_model = ollama_model

    def run(self) -> None:
        try:
            from jarvis.tools.local.ollama_sync import stream_chat
            from jarvis.tools.local.web_search_fetch import (
                fetch_search_snippets,
                format_snippets_for_prompt,
            )

            snippets = fetch_search_snippets(self._query, max_results=5)
            if not snippets:
                self.failed.emit(
                    "No web results found. Check your internet connection and try again."
                )
                return

            sources = [
                {"title": s.get("title", ""), "url": s.get("url", "")}
                for s in snippets
            ]
            context = format_snippets_for_prompt(snippets)
            user_msg = (
                f"Research query: {self._query}\n\n"
                f"Web search excerpts:\n{context}\n\n"
                "Write the summary now."
            )
            messages = [
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": user_msg},
            ]

            summary = stream_chat(
                endpoint=self._ollama_endpoint,
                model=self._ollama_model,
                messages=messages,
                on_chunk=self.chunk_received.emit,
                max_tokens=self._MAX_TOKENS,
            )
            if not summary.strip():
                self.failed.emit("Ollama returned an empty summary.")
                return
            self.finished.emit(summary, sources)

        except Exception as exc:
            log.exception("ResearchWorker failed")
            err = str(exc)
            if "connect" in err.lower() or "connection" in err.lower():
                err = (
                    "Could not reach Ollama. Is it running? "
                    "Start with: ollama serve"
                )
            self.failed.emit(err)


class FollowUpWorker(QThread):
    """Asks the local Ollama model for 3 follow-up questions.

    Emits questions_ready(list[str]) on success. Failures are silently logged
    since follow-ups are non-critical.
    """

    questions_ready = Signal(list)

    _MAX_TOKENS = 256
    _PROMPT = (
        "Given this research summary, suggest 3 short follow-up questions the user "
        "might want to ask next. Return ONLY a JSON array of 3 strings, no other text."
    )

    def __init__(
        self,
        summary: str,
        *,
        ollama_endpoint: str,
        ollama_model: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._summary = summary
        self._ollama_endpoint = ollama_endpoint
        self._ollama_model = ollama_model

    def run(self) -> None:
        try:
            from jarvis.tools.local.ollama_sync import chat_once

            text = chat_once(
                endpoint=self._ollama_endpoint,
                model=self._ollama_model,
                user_content=f"{self._PROMPT}\n\n{self._summary}",
                max_tokens=self._MAX_TOKENS,
            ).strip()
            if "```" in text:
                text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
            questions = json.loads(text)
            if isinstance(questions, list) and questions:
                self.questions_ready.emit([str(q) for q in questions[:3]])
        except Exception:
            log.debug("FollowUpWorker failed", exc_info=True)


class _FaviconThread(QThread):
    """Fetches one favicon URL and emits raw bytes on success."""

    loaded = Signal(bytes)

    def __init__(self, url: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._url = url

    def run(self) -> None:
        try:
            import urllib.request

            req = urllib.request.Request(
                self._url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.loaded.emit(resp.read())
        except Exception:
            pass  # favicon is decorative; silence all failures


# ---------------------------------------------------------------------------
# Source card widget
# ---------------------------------------------------------------------------


class _SourceCard(QFrame):
    """Card: favicon (16×16) + domain label + optional page title."""

    def __init__(
        self,
        title: str,
        url: str,
        *,
        elide_width: int = _DEFAULT_PANEL_WIDTH - 64,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._url = url
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"QFrame {{ background:{_BG_CARD}; border:1px solid {_BORDER}; }}"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        self._favicon_lbl = QLabel()
        self._favicon_lbl.setFixedSize(16, 16)
        self._favicon_lbl.setStyleSheet("background:transparent; border:none;")
        row.addWidget(self._favicon_lbl, 0, Qt.AlignmentFlag.AlignVCenter)

        parsed = urlparse(url)
        domain = parsed.netloc or url

        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        text_col.setContentsMargins(0, 0, 0, 0)

        domain_lbl = QLabel(domain)
        domain_lbl.setStyleSheet(
            f"color:{_TEXT}; font-size:9pt; font-weight:500; "
            "background:transparent; border:none;"
        )
        text_col.addWidget(domain_lbl)

        if title and title != url and title != domain:
            fm = domain_lbl.fontMetrics()
            elided = fm.elidedText(title, Qt.TextElideMode.ElideRight, elide_width)
            page_lbl = QLabel(elided)
            page_lbl.setStyleSheet(
                f"color:{_TEXT_DIM}; font-size:8pt; background:transparent; border:none;"
            )
            text_col.addWidget(page_lbl)

        row.addLayout(text_col)
        row.addStretch(1)

        favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
        self._fav_thread = _FaviconThread(favicon_url, self)
        self._fav_thread.loaded.connect(self._on_favicon_loaded)
        self._fav_thread.start()

    @Slot(bytes)
    def _on_favicon_loaded(self, data: bytes) -> None:
        pixmap = QPixmap()
        if pixmap.loadFromData(data):
            self._favicon_lbl.setPixmap(
                pixmap.scaled(
                    16, 16,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        webbrowser.open(self._url)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Panel widget
# ---------------------------------------------------------------------------


class ResearchPanel(QWidget):
    """Frameless panel that slides in from the right edge of the screen.

    Thread-safe public API (callable from any thread):
      show_for_query(query, result_queue) — start worker, show panel
      close_panel()                       — stop worker, slide out
    """

    # Cross-thread signals
    _sig_start = Signal(str, object)  # query, result_queue (may be None)
    _sig_close = Signal()
    _sig_copy  = Signal()

    def __init__(
        self,
        *,
        panel_width: int = _DEFAULT_PANEL_WIDTH,
        on_width_changed: Callable[[int], None] | None = None,
        ollama_endpoint: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b-instruct",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)

        self._panel_width = max(
            _MIN_PANEL_WIDTH, min(_MAX_PANEL_WIDTH, panel_width)
        )
        self._on_width_changed: Callable[[int], None] | None = on_width_changed
        self._ollama_endpoint = ollama_endpoint.rstrip("/")
        self._ollama_model = ollama_model
        self._resizing = False
        self._resize_start_global_x = 0.0
        self._resize_start_width = self._panel_width

        self._worker: ResearchWorker | None = None
        self._followup_worker: FollowUpWorker | None = None
        self._result_queue: queue.Queue | None = None
        self._summary_text = ""
        self._first_chunk = True
        self._anim: QPropertyAnimation | None = None
        self._history: list[str] = []
        self._history_visible: bool = False
        self._followup_was_visible: bool = False
        self._sources_was_visible: bool = False
        self._read_cursor: int = 0

        self._build_ui()
        self._position_offscreen()

        self._sig_start.connect(self._on_start)
        self._sig_close.connect(self._on_close)
        self._sig_copy.connect(self._on_copy)

    # ------------------------------------------------------------------ #
    # Public thread-safe API                                               #
    # ------------------------------------------------------------------ #

    def show_for_query(
        self,
        query: str,
        result_queue: object,
    ) -> None:
        """Trigger a new research query. Safe to call from any thread."""
        self._sig_start.emit(query, result_queue)

    def close_panel(self) -> None:
        """Stop any running worker and hide the panel. Safe from any thread."""
        self._sig_close.emit()

    def copy_summary(self) -> None:
        """Copy the current summary to clipboard. Safe to call from any thread."""
        self._sig_copy.emit()

    def get_next_sentences(self, n: int = 2) -> str | None:
        """Return the next n sentences from the summary starting at _read_cursor.

        Advances the cursor by n.  Returns None when the summary is exhausted.
        Thread-safe under CPython's GIL (simple string/int reads are atomic).
        """
        import re as _re
        text = self._summary_text.replace("•", "").strip()
        parts = _re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in parts if s.strip()]
        chunk = sentences[self._read_cursor : self._read_cursor + n]
        if not chunk:
            return None
        self._read_cursor += n
        return " ".join(chunk)

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def set_panel_width(self, width: int) -> None:
        """Apply width from settings or hot-reload. Repositions when visible."""
        clamped = max(_MIN_PANEL_WIDTH, min(_MAX_PANEL_WIDTH, width))
        if clamped == self._panel_width:
            return
        self._apply_panel_width(clamped)
        if self.isVisible():
            self.move(self._panel_target_pos())

    def _apply_panel_width(self, width: int) -> None:
        self._panel_width = width
        self.setFixedWidth(width)

    def _build_ui(self) -> None:
        self._apply_panel_width(self._panel_width)
        self.setObjectName("ResearchPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        inner = QWidget()
        inner.setStyleSheet(
            f"QWidget {{ background:{_BG}; border-left:1px solid {_BORDER}; }}"
        )
        il = QVBoxLayout(inner)
        il.setContentsMargins(20, 20, 20, 20)
        il.setSpacing(0)

        # --- header: RESEARCH | <stretch> | ≡ | COPY | × ---
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(6)

        title_lbl = QLabel("RESEARCH")
        title_lbl.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent; border:none;"
        )
        hdr.addWidget(title_lbl)
        hdr.addStretch(1)

        self._history_btn = QPushButton("≡")
        self._history_btn.setFixedHeight(24)
        self._history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._history_btn.setToolTip("Query history")
        self._history_btn.setStyleSheet(_HDR_BTN)
        self._history_btn.clicked.connect(self._on_history_toggle)
        hdr.addWidget(self._history_btn)

        self._copy_btn = QPushButton("COPY")
        self._copy_btn.setFixedHeight(24)
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.setStyleSheet(_HDR_BTN)
        self._copy_btn.clicked.connect(self._on_copy)
        hdr.addWidget(self._copy_btn)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 24)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:16pt; background:transparent; "
            "border:none; padding:0;"
        )
        close_btn.clicked.connect(self._on_close)
        hdr.addWidget(close_btn)
        il.addLayout(hdr)
        il.addSpacing(12)

        # --- query label ---
        self._query_label = QLabel()
        self._query_label.setWordWrap(True)
        self._query_label.setStyleSheet(
            f"color:{_CYAN}; font-size:13pt; font-weight:300; "
            "letter-spacing:1px; background:transparent; border:none;"
        )
        il.addWidget(self._query_label)
        il.addSpacing(14)
        il.addWidget(self._divider())
        il.addSpacing(14)

        # --- history section (hidden; shown when ≡ is active) ---
        self._history_section = self._build_history_section()
        il.addWidget(self._history_section, stretch=1)

        # --- streaming content area ---
        self._content = QPlainTextEdit()
        self._content.setReadOnly(True)
        self._content.setFrameShape(QFrame.Shape.NoFrame)
        self._content.setStyleSheet(
            f"background:{_BG}; color:{_TEXT}; border:none; "
            "font-size:10pt; font-weight:300;"
        )
        self._content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        il.addWidget(self._content, stretch=1)

        # --- follow-up chips section (hidden until chips arrive) ---
        self._followup_section = self._build_followup_section()
        il.addWidget(self._followup_section)

        # --- sources section (hidden until sources arrive) ---
        self._sources_section = self._build_sources_section()
        il.addWidget(self._sources_section)

        root.addWidget(inner)

    def _build_history_section(self) -> QWidget:
        w = QWidget()
        w.hide()

        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr_lbl = QLabel("RECENT SEARCHES")
        hdr_lbl.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent; border:none;"
        )
        layout.addWidget(hdr_lbl)
        layout.addSpacing(10)

        self._history_list_container = QWidget()
        self._history_list_container.setStyleSheet(f"background:{_BG};")
        self._history_list_layout = QVBoxLayout(self._history_list_container)
        self._history_list_layout.setContentsMargins(0, 0, 0, 0)
        self._history_list_layout.setSpacing(6)
        self._history_list_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background:{_BG}; border:none;")
        scroll.setWidget(self._history_list_container)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=1)

        return w

    def _build_followup_section(self) -> QWidget:
        w = QWidget()
        w.hide()

        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._divider())
        layout.addSpacing(10)

        hdr_lbl = QLabel("FOLLOW-UP")
        hdr_lbl.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent; border:none;"
        )
        layout.addWidget(hdr_lbl)
        layout.addSpacing(6)

        self._chips_container = QWidget()
        self._chips_layout = QVBoxLayout(self._chips_container)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(6)
        layout.addWidget(self._chips_container)

        return w

    def _build_sources_section(self) -> QWidget:
        w = QWidget()

        src_l = QVBoxLayout(w)
        src_l.setContentsMargins(0, 14, 0, 0)
        src_l.setSpacing(8)
        src_l.addWidget(self._divider())
        src_l.addSpacing(10)

        src_hdr = QLabel("SOURCES")
        src_hdr.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent; border:none;"
        )
        src_l.addWidget(src_hdr)
        src_l.addSpacing(6)

        self._sources_scroll = QScrollArea()
        self._sources_scroll.setFixedHeight(160)
        self._sources_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._sources_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._sources_scroll.setStyleSheet(f"background:{_BG}; border:none;")
        self._sources_container = QWidget()
        self._sources_container.setStyleSheet(f"background:{_BG};")
        self._sources_inner = QVBoxLayout(self._sources_container)
        self._sources_inner.setContentsMargins(0, 0, 0, 0)
        self._sources_inner.setSpacing(6)
        self._sources_inner.addStretch(1)
        self._sources_scroll.setWidget(self._sources_container)
        self._sources_scroll.setWidgetResizable(True)
        src_l.addWidget(self._sources_scroll)

        w.hide()
        return w

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background:{_BORDER}; border:none;")
        return line

    # ------------------------------------------------------------------ #
    # Slots (always Qt main thread)                                        #
    # ------------------------------------------------------------------ #

    @Slot(str, object)
    def _on_start(self, query: str, result_queue: object) -> None:
        self._stop_worker()
        self._result_queue = result_queue  # type: ignore[assignment]
        self._summary_text = ""
        self._first_chunk = True

        # Add to history (deduplicate, newest first, cap at _MAX_HISTORY)
        if query in self._history:
            self._history.remove(query)
        self._history.insert(0, query)
        self._history = self._history[:_MAX_HISTORY]

        # Reset to main content view
        self._history_visible = False
        self._history_section.hide()
        self._content.show()
        self._history_btn.setStyleSheet(_HDR_BTN)

        # Reset content
        self._query_label.setText(query)
        self._content.clear()
        self._content.setPlainText("Searching the web…")
        self._content.setStyleSheet(
            f"background:{_BG}; color:{_TEXT_DIM}; border:none; "
            "font-size:10pt; font-weight:300;"
        )
        self._clear_source_cards()
        self._sources_section.hide()
        self._clear_chips()
        self._followup_section.hide()
        self._read_cursor = 0

        # Start QThread worker
        self._worker = ResearchWorker(
            query,
            ollama_endpoint=self._ollama_endpoint,
            ollama_model=self._ollama_model,
        )
        self._worker.chunk_received.connect(self._on_chunk)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

        # Show panel
        self._resize_to_screen()
        self.show()
        self._slide_in()

    @Slot(str)
    def _on_chunk(self, chunk: str) -> None:
        if self._first_chunk:
            self._content.clear()
            self._content.setStyleSheet(
                f"background:{_BG}; color:{_TEXT}; border:none; "
                "font-size:10pt; font-weight:300;"
            )
            self._first_chunk = False
        self._summary_text += chunk
        self._content.moveCursor(self._content.textCursor().MoveOperation.End)
        self._content.insertPlainText(chunk)

    @Slot(str, object)
    def _on_worker_finished(self, summary: str, sources: list) -> None:
        self._summary_text = summary
        if sources:
            self._populate_source_cards(sources)
        if self._result_queue is not None:
            try:
                self._result_queue.put_nowait((summary, sources))  # type: ignore[union-attr]
            except Exception:
                pass
            self._result_queue = None
            # ResearchTool will speak the first 2 sentences; advance cursor past them
            self._read_cursor = 2
        # Generate follow-up questions asynchronously
        if summary:
            self._start_followup_worker(summary)

    @Slot(str)
    def _on_worker_failed(self, error: str) -> None:
        self._content.appendPlainText(f"\n\n[Error: {error}]")
        if self._result_queue is not None:
            try:
                self._result_queue.put_nowait(RuntimeError(error))  # type: ignore[union-attr]
            except Exception:
                pass
            self._result_queue = None

    @Slot()
    def _on_close(self) -> None:
        self._stop_worker()
        if self.isVisible():
            self._slide_out()

    @Slot()
    def _on_copy(self) -> None:
        if self._summary_text:
            QApplication.clipboard().setText(self._summary_text)
        self._copy_btn.setText("Copied!")
        QTimer.singleShot(1500, lambda: self._copy_btn.setText("COPY"))

    @Slot()
    def _on_history_toggle(self) -> None:
        self._history_visible = not self._history_visible
        if self._history_visible:
            # Snapshot transient section visibility so we can restore it
            self._followup_was_visible = self._followup_section.isVisible()
            self._sources_was_visible = self._sources_section.isVisible()
            self._update_history_list()
            self._history_section.show()
            self._content.hide()
            self._followup_section.hide()
            self._sources_section.hide()
            self._history_btn.setStyleSheet(_HDR_BTN_ACTIVE)
        else:
            self._history_section.hide()
            self._content.show()
            if self._followup_was_visible:
                self._followup_section.show()
            if self._sources_was_visible:
                self._sources_section.show()
            self._history_btn.setStyleSheet(_HDR_BTN)

    def _on_history_query_clicked(self, query: str) -> None:
        """Re-run a query from the history list (main thread only)."""
        self._history_visible = False
        self._sig_start.emit(query, None)

    @Slot(list)
    def _on_followup_questions(self, questions: list) -> None:
        self._clear_chips()
        for question in questions:
            chip = QPushButton(question)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setStyleSheet(
                f"QPushButton {{ background:transparent; color:{_TEXT_MID}; "
                f"border:1px solid {_CHIP_BORDER}; border-radius:14px; "
                "padding:5px 14px; font-size:9pt; font-weight:300; "
                "text-align:left; }}"
                f"QPushButton:hover {{ border-color:{_CYAN}; color:{_TEXT}; "
                "background:#0a1a1a; }}"
                "QPushButton:pressed { background:#0d1a1a; }"
            )
            chip.clicked.connect(
                lambda _checked=False, q=question: self._on_chip_click(q)
            )
            self._chips_layout.addWidget(chip)
        # Only show if the user isn't looking at the history list
        if not self._history_visible and questions:
            self._followup_section.show()

    def _on_chip_click(self, query: str) -> None:
        """Re-run research from a follow-up chip (main thread only)."""
        self._sig_start.emit(query, None)

    # ------------------------------------------------------------------ #
    # Keyboard                                                             #
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._on_close()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _stop_worker(self) -> None:
        """Gracefully stop any running workers and drain the result queue slot."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        self._worker = None
        if self._result_queue is not None:
            try:
                self._result_queue.put_nowait(RuntimeError("interrupted"))  # type: ignore[union-attr]
            except Exception:
                pass
            self._result_queue = None
        self._stop_followup_worker()

    def _stop_followup_worker(self) -> None:
        if self._followup_worker is not None and self._followup_worker.isRunning():
            self._followup_worker.quit()
            self._followup_worker.wait(1000)
        self._followup_worker = None

    def _start_followup_worker(self, summary: str) -> None:
        self._stop_followup_worker()
        self._followup_worker = FollowUpWorker(
            summary,
            ollama_endpoint=self._ollama_endpoint,
            ollama_model=self._ollama_model,
        )
        self._followup_worker.questions_ready.connect(self._on_followup_questions)
        self._followup_worker.start()

    def _update_history_list(self) -> None:
        """Rebuild the history list buttons from self._history."""
        # Remove all but the trailing stretch
        while self._history_list_layout.count() > 1:
            item = self._history_list_layout.takeAt(0)
            w = item.widget() if item else None
            if w:
                w.deleteLater()

        for query in self._history:
            btn = QPushButton(query)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            btn.setStyleSheet(
                f"QPushButton {{ background:{_BG_CARD}; color:{_TEXT_MID}; "
                f"border:1px solid {_BORDER}; border-radius:6px; "
                "padding:8px 12px; font-size:9pt; font-weight:300; "
                "text-align:left; }}"
                f"QPushButton:hover {{ border-color:{_CYAN}; color:{_TEXT}; }}"
                "QPushButton:pressed { background:#0a1515; }"
            )
            btn.clicked.connect(
                lambda _checked=False, q=query: self._on_history_query_clicked(q)
            )
            # Insert before the trailing stretch
            self._history_list_layout.insertWidget(
                self._history_list_layout.count() - 1, btn
            )

    def _clear_source_cards(self) -> None:
        stretch = self._sources_inner.count() - 1
        for i in range(stretch - 1, -1, -1):
            item = self._sources_inner.takeAt(i)
            w = item.widget() if item else None
            if w:
                w.deleteLater()

    def _populate_source_cards(self, sources: list) -> None:
        self._clear_source_cards()
        for src in sources:
            card = _SourceCard(
                src.get("title", ""),
                src.get("url", ""),
                elide_width=self._panel_width - 64,
                parent=self._sources_container,
            )
            self._sources_inner.insertWidget(self._sources_inner.count() - 1, card)
        self._sources_section.show()

    def _clear_chips(self) -> None:
        while self._chips_layout.count():
            item = self._chips_layout.takeAt(0)
            w = item.widget() if item else None
            if w:
                w.deleteLater()

    # ------------------------------------------------------------------ #
    # Geometry + animation                                                 #
    # ------------------------------------------------------------------ #

    def _screen_geometry(self):
        screen = QApplication.primaryScreen()
        return screen.availableGeometry() if screen else None

    def _resize_to_screen(self) -> None:
        geo = self._screen_geometry()
        if geo is not None:
            self.setFixedHeight(int(geo.height() * 0.80))

    def _panel_target_pos(self) -> QPoint:
        geo = self._screen_geometry()
        if geo is None:
            return QPoint(0, 0)
        x = geo.right() - self._panel_width - _SIDE_MARGIN + 1
        y = geo.top() + (geo.height() - self.height()) // 2
        return QPoint(x, y)

    def _position_offscreen(self) -> None:
        geo = self._screen_geometry()
        if geo is not None:
            self.move(geo.right() + 10, 0)

    def _slide_in(self) -> None:
        target = self._panel_target_pos()
        start = QPoint(target.x() + self._panel_width + _SIDE_MARGIN, target.y())
        self.move(start)
        self._animate(start, target, QEasingCurve.Type.OutQuad)

    def _slide_out(self) -> None:
        current = self.pos()
        end = QPoint(current.x() + self._panel_width + _SIDE_MARGIN, current.y())

        def _hide() -> None:
            self.hide()
            self._position_offscreen()

        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(_ANIM_MS)
        anim.setEasingCurve(QEasingCurve.Type.InQuad)
        anim.setStartValue(current)
        anim.setEndValue(end)
        anim.finished.connect(_hide)
        self._anim = anim
        anim.start()

    def _animate(self, start: QPoint, end: QPoint, curve: QEasingCurve.Type) -> None:
        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(_ANIM_MS)
        anim.setEasingCurve(curve)
        anim.setStartValue(start)
        anim.setEndValue(end)
        self._anim = anim
        anim.start()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.position().x() <= _RESIZE_HIT_PX
        ):
            self._resizing = True
            self._resize_start_global_x = event.globalPosition().x()
            self._resize_start_width = self._panel_width
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._resizing:
            delta = int(self._resize_start_global_x - event.globalPosition().x())
            new_w = max(
                _MIN_PANEL_WIDTH,
                min(_MAX_PANEL_WIDTH, self._resize_start_width + delta),
            )
            if new_w != self._panel_width:
                self._apply_panel_width(new_w)
                if self.isVisible():
                    self.move(self._panel_target_pos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._resizing and event.button() == Qt.MouseButton.LeftButton:
            self._resizing = False
            if self._on_width_changed is not None:
                self._on_width_changed(self._panel_width)
            event.accept()
            return
        super().mouseReleaseEvent(event)
