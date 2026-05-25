"""Settings → Help tab: the same capability list, always discoverable."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from jarvis.core.config import JarvisConfig
from jarvis.ui.capabilities import (
    CAPABILITY_CATEGORIES,
    Capability,
    CapabilityCategory,
    search_capabilities,
)


def _capability_row(cap: Capability) -> QFrame:
    card = QFrame()
    card.setStyleSheet(
        "QFrame { background:#181818; border:1px solid #2a2a2a; "
        "border-radius:4px; padding:8px; }"
    )
    lay = QVBoxLayout(card)
    lay.setContentsMargins(10, 6, 10, 8)
    lay.setSpacing(4)

    name = QLabel(cap.name)
    name.setStyleSheet(
        "color:#ffffff; font-size:10pt; font-weight:500; background:transparent;"
    )
    lay.addWidget(name)

    desc = QLabel(cap.description)
    desc.setWordWrap(True)
    desc.setStyleSheet(
        "color:#b0b0b0; font-size:9pt; background:transparent;"
    )
    lay.addWidget(desc)

    for example in cap.examples:
        line = QLabel(f"  Say: “{example}”")
        line.setWordWrap(True)
        line.setStyleSheet(
            "color:#38f4ff; font-size:9pt; font-style:italic; background:transparent;"
        )
        lay.addWidget(line)
    return card


class HelpTab(QWidget):
    """Read-only capability reference. No config writes."""

    def __init__(
        self,
        *,
        config: JarvisConfig,
        on_change: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Stored for interface parity with other tabs (auto-saved by main_window).
        self._config = config
        self._on_change = on_change

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        intro = QLabel(
            "Every command Jarvis understands. Say “Hey Jarvis” first, "
            "then any example phrase. Use the search box to narrow down."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#b0b0b0; font-size:9pt;")
        root.addWidget(intro)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search (e.g. note, weather, music)…")
        self._search.textChanged.connect(self._on_search_changed)
        root.addWidget(self._search)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)
        self._populate_full()
        self._scroll.setWidget(self._content)
        root.addWidget(self._scroll, 1)

    def _clear(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget() if item else None
            if w:
                w.deleteLater()

    def _add_header(self, cat: CapabilityCategory) -> None:
        h = QLabel(f"{cat.icon_glyph}  {cat.name.upper()}")
        h.setStyleSheet(
            "color:#38f4ff; font-size:9pt; font-weight:500; "
            "letter-spacing:3px; padding-top:8px;"
        )
        self._content_layout.addWidget(h)

    def _populate_full(self) -> None:
        self._clear()
        for cat in CAPABILITY_CATEGORIES:
            self._add_header(cat)
            for cap in cat.capabilities:
                self._content_layout.addWidget(_capability_row(cap))
        self._content_layout.addStretch(1)

    def _populate_filtered(self, results) -> None:
        self._clear()
        if not results:
            empty = QLabel("No matches.")
            empty.setStyleSheet("color:#808080; padding:20px;")
            self._content_layout.addWidget(empty)
            self._content_layout.addStretch(1)
            return
        last_cat: CapabilityCategory | None = None
        for cat, cap in results:
            if cat is not last_cat:
                self._add_header(cat)
                last_cat = cat
            self._content_layout.addWidget(_capability_row(cap))
        self._content_layout.addStretch(1)

    def _on_search_changed(self, text: str) -> None:
        if not text.strip():
            self._populate_full()
        else:
            self._populate_filtered(search_capabilities(text))
