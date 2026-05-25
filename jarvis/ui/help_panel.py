"""Help panel — slide-in 'what can I say?' reference for non-technical users."""

from __future__ import annotations

import logging

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from jarvis.ui.capabilities import (
    CAPABILITY_CATEGORIES,
    Capability,
    CapabilityCategory,
    search_capabilities,
)

log = logging.getLogger(__name__)

_DEFAULT_PANEL_WIDTH = 520
_SIDE_MARGIN = 16
_ANIM_MS = 280

_BG = "#0d0d0d"
_BG_CARD = "#141414"
_BORDER = "#1f1f1f"
_TEXT = "#ffffff"
_TEXT_DIM = "#808080"
_TEXT_MID = "#b0b0b0"
_CYAN = "#38f4ff"

_HDR_BTN = (
    f"QPushButton {{ color:{_TEXT_DIM}; background:transparent; border:none; "
    "font-size:8pt; letter-spacing:2px; font-weight:300; padding:0 6px; }}"
    f"QPushButton:hover {{ color:{_TEXT}; }}"
)


def _build_capability_card(
    cap: Capability,
    *,
    on_phrase_click=None,
) -> QFrame:
    card = QFrame()
    card.setStyleSheet(
        f"QFrame {{ background:{_BG_CARD}; border:1px solid {_BORDER}; "
        "border-radius:6px; padding:10px; }}"
    )
    lay = QVBoxLayout(card)
    lay.setContentsMargins(12, 8, 12, 10)
    lay.setSpacing(6)

    name = QLabel(cap.name)
    name.setStyleSheet(
        f"color:{_TEXT}; font-size:11pt; font-weight:500; background:transparent;"
    )
    lay.addWidget(name)

    desc = QLabel(cap.description)
    desc.setWordWrap(True)
    desc.setStyleSheet(
        f"color:{_TEXT_MID}; font-size:9pt; background:transparent;"
    )
    lay.addWidget(desc)

    say_row = QHBoxLayout()
    say_row.setContentsMargins(0, 4, 0, 0)
    say_row.setSpacing(6)
    say_label = QLabel("Say:")
    say_label.setStyleSheet(
        f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
        "letter-spacing:2px; background:transparent;"
    )
    say_row.addWidget(say_label)
    say_row.addStretch(0)
    lay.addLayout(say_row)

    for example in cap.examples:
        chip = QLabel(f"  “{example}”")
        chip.setWordWrap(True)
        chip.setStyleSheet(
            f"color:{_CYAN}; font-size:9pt; font-style:italic; background:transparent;"
        )
        lay.addWidget(chip)
    return card


class HelpPanel(QWidget):
    """Slide-in capabilities reference, with live search."""

    _sig_open = Signal()
    _sig_close = Signal()

    def __init__(
        self,
        *,
        panel_width: int = _DEFAULT_PANEL_WIDTH,
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

        self._panel_width = panel_width
        self._anim: QPropertyAnimation | None = None

        self._build_ui()
        self._position_offscreen()

        self._sig_open.connect(self._on_open)
        self._sig_close.connect(self._on_close)

    # -------------------------------------------------------- public api

    def open_panel(self) -> None:
        self._sig_open.emit()

    def close_panel(self) -> None:
        self._sig_close.emit()

    # -------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self.setFixedWidth(self._panel_width)
        self.setObjectName("HelpPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        inner = QWidget()
        inner.setStyleSheet(
            f"QWidget {{ background:{_BG}; border-left:1px solid {_BORDER}; }}"
        )
        il = QVBoxLayout(inner)
        il.setContentsMargins(16, 16, 16, 16)
        il.setSpacing(10)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("WHAT CAN I SAY?")
        title.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent;"
        )
        hdr.addWidget(title)
        hdr.addStretch(1)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 24)
        close_btn.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:16pt; background:transparent; border:none;"
        )
        close_btn.clicked.connect(self._on_close)
        hdr.addWidget(close_btn)
        il.addLayout(hdr)

        tagline = QLabel(
            "Every command Jarvis understands, in plain English. "
            "Say “Hey Jarvis” first, then any of the example phrases below."
        )
        tagline.setWordWrap(True)
        tagline.setStyleSheet(
            f"color:{_TEXT_MID}; font-size:9pt; background:transparent;"
        )
        il.addWidget(tagline)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search (e.g. note, weather, music)…")
        self._search.setStyleSheet(
            f"QLineEdit {{ background:#0a0a0a; color:{_TEXT}; "
            f"border:1px solid {_BORDER}; padding:6px 10px; font-size:10pt; }}"
            f"QLineEdit:focus {{ border:1px solid {_CYAN}; }}"
        )
        self._search.textChanged.connect(self._on_search_changed)
        il.addWidget(self._search)

        # Scroll area with cards
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background:{_BG}; border:none; }}"
        )

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(8)
        self._populate_full()
        self._scroll.setWidget(self._content)
        il.addWidget(self._scroll, 1)

        root.addWidget(inner)

    def _clear_content(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget() if item else None
            if w:
                w.deleteLater()

    def _add_category_header(self, cat: CapabilityCategory) -> None:
        header = QLabel(f"{cat.icon_glyph}  {cat.name.upper()}")
        header.setStyleSheet(
            f"color:{_CYAN}; font-size:9pt; font-weight:500; "
            "letter-spacing:3px; background:transparent; "
            "padding-top:8px; padding-bottom:2px;"
        )
        self._content_layout.addWidget(header)

    def _populate_full(self) -> None:
        self._clear_content()
        for cat in CAPABILITY_CATEGORIES:
            self._add_category_header(cat)
            for cap in cat.capabilities:
                self._content_layout.addWidget(_build_capability_card(cap))
        self._content_layout.addStretch(1)

    def _populate_filtered(self, results) -> None:
        self._clear_content()
        if not results:
            empty = QLabel("No matches. Try a different word.")
            empty.setStyleSheet(
                f"color:{_TEXT_DIM}; font-size:9pt; padding:20px; background:transparent;"
            )
            self._content_layout.addWidget(empty)
            self._content_layout.addStretch(1)
            return
        last_cat: CapabilityCategory | None = None
        for cat, cap in results:
            if cat is not last_cat:
                self._add_category_header(cat)
                last_cat = cat
            self._content_layout.addWidget(_build_capability_card(cap))
        self._content_layout.addStretch(1)

    @Slot(str)
    def _on_search_changed(self, text: str) -> None:
        if not text.strip():
            self._populate_full()
            return
        self._populate_filtered(search_capabilities(text))

    # ---------------------------------------------------- slots / geom

    @Slot()
    def _on_open(self) -> None:
        self._show_panel()
        self._search.setFocus()

    @Slot()
    def _on_close(self) -> None:
        if self.isVisible():
            self._slide_out()

    def _show_panel(self) -> None:
        self._resize_to_screen()
        self.show()
        self._slide_in()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._on_close()
        else:
            super().keyPressEvent(event)

    def _screen_geometry(self):
        screen = QApplication.primaryScreen()
        return screen.availableGeometry() if screen else None

    def _resize_to_screen(self) -> None:
        geo = self._screen_geometry()
        if geo is not None:
            self.setFixedHeight(int(geo.height() * 0.85))

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
        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(_ANIM_MS)
        anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        anim.setStartValue(start)
        anim.setEndValue(target)
        self._anim = anim
        anim.start()

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
