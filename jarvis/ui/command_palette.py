"""Command palette — keyboard-driven fuzzy launcher.

Press the configured hotkey (default ``ctrl+shift+p``). Start typing —
the list filters live against every Jarvis capability's example phrases
and category names. Up/Down to navigate, Enter to submit the highlighted
phrase as if it were a spoken command, Esc to dismiss.

Submission is routed through the audio loop via the injected
``submit_text`` callable (see ``app.py`` for the wiring), so all the
same intent-router patterns and tools fire as for STT input.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from jarvis.ui.capabilities import (
    CAPABILITY_CATEGORIES,
    Capability,
    CapabilityCategory,
)

log = logging.getLogger(__name__)

_PANEL_WIDTH = 620
_PANEL_HEIGHT = 480
_BG = "#0d0d0d"
_BG_INPUT = "#0a0a0a"
_BORDER = "#1f1f1f"
_BORDER_BRIGHT = "#38f4ff"
_TEXT = "#ffffff"
_TEXT_DIM = "#808080"
_TEXT_MID = "#b0b0b0"
_CYAN = "#38f4ff"


@dataclass
class PaletteEntry:
    phrase: str
    capability_name: str
    category_name: str

    @property
    def display(self) -> str:
        return self.phrase

    @property
    def hint(self) -> str:
        return f"{self.category_name} · {self.capability_name}"


def _build_entries() -> list[PaletteEntry]:
    out: list[PaletteEntry] = []
    for cat in CAPABILITY_CATEGORIES:
        for cap in cat.capabilities:
            for phrase in cap.examples:
                out.append(
                    PaletteEntry(
                        phrase=phrase,
                        capability_name=cap.name,
                        category_name=cat.name,
                    )
                )
    return out


def _fuzzy_score(query: str, text: str) -> int:
    """Tiny fuzzy ranker: 0 = no match, higher = better.

    - Exact substring is strongest.
    - Subsequence (letters appear in order) is weaker but still matches.
    - Empty query matches everything with neutral score.
    """
    if not query:
        return 1
    q = query.lower()
    t = text.lower()
    if q in t:
        # Earlier hits rank higher.
        return 1000 - t.index(q)
    qi = 0
    matched = 0
    for ch in t:
        if qi < len(q) and ch == q[qi]:
            qi += 1
            matched += 1
    if qi == len(q):
        return 100 + matched
    return 0


class CommandPalette(QWidget):
    """Frameless, centered, modal-looking command launcher."""

    _sig_open = Signal()

    def __init__(
        self,
        *,
        submit_text: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._submit_text = submit_text
        self._entries = _build_entries()

        self._build_ui()
        self._sig_open.connect(self._on_open)
        self.hide()

    # ------------------------------------------------------------ public

    def open_palette(self) -> None:
        self._sig_open.emit()

    def close_palette(self) -> None:
        self.hide()

    # ----------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self.setFixedSize(_PANEL_WIDTH, _PANEL_HEIGHT)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background:{_BG}; border:1px solid {_BORDER_BRIGHT}; "
            "border-radius:8px; }}"
        )
        flay = QVBoxLayout(frame)
        flay.setContentsMargins(14, 14, 14, 14)
        flay.setSpacing(8)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("COMMAND PALETTE")
        title.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent;"
        )
        hdr.addWidget(title)
        hdr.addStretch(1)
        hint = QLabel("Enter: run    Esc: close")
        hint.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; background:transparent;"
        )
        hdr.addWidget(hint)
        flay.addLayout(hdr)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Type a command or phrase…")
        self._input.setStyleSheet(
            f"QLineEdit {{ color:{_TEXT}; background:{_BG_INPUT}; "
            f"border:1px solid {_BORDER}; padding:8px 12px; font-size:12pt; }}"
            f"QLineEdit:focus {{ border:1px solid {_CYAN}; }}"
        )
        self._input.textChanged.connect(self._on_text_changed)
        self._input.returnPressed.connect(self._submit_current)
        flay.addWidget(self._input)

        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget {{ background:{_BG_INPUT}; color:{_TEXT}; "
            f"border:1px solid {_BORDER}; font-size:10pt; padding:2px; }}"
            f"QListWidget::item {{ padding:6px 8px; }}"
            f"QListWidget::item:selected {{ background:#1a2a2a; color:{_CYAN}; }}"
        )
        self._list.itemActivated.connect(lambda _it: self._submit_current())
        flay.addWidget(self._list, 1)

        footer = QLabel(
            "Tip: anything you can say to Jarvis works here too. "
            "Examples — 'show dashboard', 'open my notes', "
            "'research black holes', 'play some jazz'."
        )
        footer.setWordWrap(True)
        footer.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:9pt; background:transparent;"
        )
        flay.addWidget(footer)

        root.addWidget(frame)
        self._populate("")

    def _populate(self, query: str) -> None:
        scored: list[tuple[int, PaletteEntry]] = []
        for entry in self._entries:
            haystack = f"{entry.phrase}  {entry.capability_name}  {entry.category_name}"
            score = _fuzzy_score(query, haystack)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda t: -t[0])
        self._list.clear()
        if not scored and query:
            li = QListWidgetItem(f"↵  Run anyway: “{query}”")
            li.setData(Qt.ItemDataRole.UserRole, query)
            self._list.addItem(li)
            self._list.setCurrentRow(0)
            return
        for _score, entry in scored[:80]:
            li = QListWidgetItem(f"{entry.display}\n   {entry.hint}")
            li.setData(Qt.ItemDataRole.UserRole, entry.phrase)
            self._list.addItem(li)
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    @Slot(str)
    def _on_text_changed(self, text: str) -> None:
        self._populate(text)

    def _submit_current(self) -> None:
        item = self._list.currentItem()
        text = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not text:
            text = self._input.text().strip()
        if not text:
            return
        try:
            self._submit_text(text)
        except Exception:  # noqa: BLE001
            log.exception("command palette submit failed")
        self.hide()

    # ----------------------------------------------------------- events

    @Slot()
    def _on_open(self) -> None:
        self._input.clear()
        self._populate("")
        self._center_on_screen()
        self.show()
        self.raise_()
        self.activateWindow()
        self._input.setFocus(Qt.FocusReason.OtherFocusReason)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.hide()
            return
        if key == Qt.Key.Key_Down:
            row = min(self._list.count() - 1, self._list.currentRow() + 1)
            if row >= 0:
                self._list.setCurrentRow(row)
            event.accept()
            return
        if key == Qt.Key.Key_Up:
            row = max(0, self._list.currentRow() - 1)
            self._list.setCurrentRow(row)
            event.accept()
            return
        super().keyPressEvent(event)

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.left() + (geo.width() - self.width()) // 2
        y = geo.top() + int(geo.height() * 0.25)
        self.move(x, y)
