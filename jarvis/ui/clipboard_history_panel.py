"""Clipboard history panel — slide-in list of recent clipboard items.

A QTimer polls the system clipboard every ~700 ms (only while the panel
is alive; we don't run a global background poller — clipboard polling
is mildly chatty so we keep it scoped to active use). Each new text is
appended via ClipboardHistory.add().
"""

from __future__ import annotations

import logging

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jarvis.platform import windows as winplat
from jarvis.tools.local.clipboard_history import (
    ClipboardHistory,
    load_history,
    save_history,
)

log = logging.getLogger(__name__)

_DEFAULT_PANEL_WIDTH = 460
_SIDE_MARGIN = 16
_ANIM_MS = 280
_POLL_MS = 700

_BG = "#0d0d0d"
_BORDER = "#1f1f1f"
_TEXT = "#ffffff"
_TEXT_DIM = "#808080"
_CYAN = "#38f4ff"
_PIN = "#e6b94a"

_HDR_BTN = (
    f"QPushButton {{ color:{_TEXT_DIM}; background:transparent; border:none; "
    "font-size:8pt; letter-spacing:2px; font-weight:300; padding:0 6px; }}"
    f"QPushButton:hover {{ color:{_TEXT}; }}"
)


class ClipboardHistoryPanel(QWidget):
    """Slide-in clipboard history with copy / pin / delete."""

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
        self._history: ClipboardHistory = load_history()
        # Seed against current clipboard so we don't double-add on first open.
        self._last_seen: str | None = self._safe_read_clipboard()

        self._build_ui()
        self._refresh_list()
        self._position_offscreen()

        self._poll = QTimer(self)
        self._poll.setInterval(_POLL_MS)
        self._poll.timeout.connect(self._poll_clipboard)

        self._sig_open.connect(self._on_open)
        self._sig_close.connect(self._on_close)

    # ------------------------------------------------------------ public

    def open_panel(self) -> None:
        self._sig_open.emit()

    def close_panel(self) -> None:
        self._sig_close.emit()

    def paste_index(self, one_based_index: int) -> str | None:
        """Voice/tool entry: copy the Nth-most-recent item to the live
        clipboard. Returns the preview of what was pasted, or None if
        the index is out of range."""
        idx = one_based_index - 1
        item = self._history.get(idx)
        if item is None:
            return None
        try:
            winplat.write_clipboard_text(item.text)
        except Exception:  # noqa: BLE001
            log.exception("clipboard write failed")
            return None
        # Bump it to the front and update last_seen so the poller doesn't
        # re-record it as a fresh entry.
        self._history.remove(idx)
        self._history.add(item.text)
        if item.pinned:
            self._history.pin(0, True)
        self._last_seen = item.text
        save_history(self._history)
        self._refresh_list()
        return item.preview(80)

    def clear_all(self, *, keep_pinned: bool = True) -> int:
        n = self._history.clear(keep_pinned=keep_pinned)
        save_history(self._history)
        self._refresh_list()
        return n

    def count(self) -> int:
        return len(self._history.items)

    # ----------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self.setFixedWidth(self._panel_width)
        self.setObjectName("ClipboardHistoryPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        inner = QWidget()
        inner.setStyleSheet(
            f"QWidget {{ background:{_BG}; border-left:1px solid {_BORDER}; }}"
        )
        il = QVBoxLayout(inner)
        il.setContentsMargins(16, 16, 16, 16)
        il.setSpacing(8)

        hdr = QHBoxLayout()
        title = QLabel("CLIPBOARD HISTORY")
        title.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent;"
        )
        hdr.addWidget(title)
        hdr.addStretch(1)

        self._clear_btn = QPushButton("CLEAR")
        self._clear_btn.setStyleSheet(_HDR_BTN)
        self._clear_btn.setToolTip("Clear unpinned items")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        hdr.addWidget(self._clear_btn)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 24)
        close_btn.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:16pt; background:transparent; border:none;"
        )
        close_btn.clicked.connect(self._on_close)
        hdr.addWidget(close_btn)
        il.addLayout(hdr)

        sub = QLabel(
            "Items captured while this panel is open. Double-click to copy. "
            "Right-click for pin / delete."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:9pt; background:transparent;"
        )
        il.addWidget(sub)

        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget {{ background:#0a0a0a; color:{_TEXT}; "
            f"border:1px solid {_BORDER}; font-size:9pt; }}"
            f"QListWidget::item:selected {{ background:#1a2a2a; color:{_CYAN}; }}"
        )
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        il.addWidget(self._list, 1)

        root.addWidget(inner)

    # ------------------------------------------------------------ slots

    @Slot()
    def _on_open(self) -> None:
        # Re-load from disk so changes made via voice tools while panel
        # was closed are reflected.
        self._history = load_history()
        self._last_seen = self._safe_read_clipboard()
        self._refresh_list()
        if not self._poll.isActive():
            self._poll.start()
        self._show_panel()

    @Slot()
    def _on_close(self) -> None:
        if self._poll.isActive():
            self._poll.stop()
        if self.isVisible():
            self._slide_out()

    def _safe_read_clipboard(self) -> str | None:
        try:
            return winplat.read_clipboard_text()
        except Exception:  # noqa: BLE001
            return None

    def _poll_clipboard(self) -> None:
        text = self._safe_read_clipboard()
        if text is None:
            return
        if text == self._last_seen:
            return
        self._last_seen = text
        if self._history.add(text):
            save_history(self._history)
            self._refresh_list()

    def _refresh_list(self) -> None:
        self._list.clear()
        if not self._history.items:
            empty = QListWidgetItem("(empty — copy something while this is open)")
            empty.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._list.addItem(empty)
            return
        for idx, item in enumerate(self._history.items):
            pin = "★ " if item.pinned else ""
            li = QListWidgetItem(f"{idx + 1}.  {pin}{item.preview(120)}")
            li.setData(Qt.ItemDataRole.UserRole, idx)
            if item.pinned:
                li.setForeground(Qt.GlobalColor.yellow)
            self._list.addItem(li)

    def _on_double_click(self, li: QListWidgetItem) -> None:
        idx = li.data(Qt.ItemDataRole.UserRole)
        if not isinstance(idx, int):
            return
        self.paste_index(idx + 1)

    def _on_context_menu(self, pos) -> None:
        li = self._list.itemAt(pos)
        if li is None:
            return
        idx = li.data(Qt.ItemDataRole.UserRole)
        if not isinstance(idx, int):
            return
        item = self._history.get(idx)
        if item is None:
            return
        menu = QMenu(self)
        copy_act = menu.addAction("Copy to clipboard")
        pin_label = "Unpin" if item.pinned else "Pin"
        pin_act = menu.addAction(pin_label)
        menu.addSeparator()
        del_act = menu.addAction("Delete")
        chosen = menu.exec(self._list.mapToGlobal(pos))
        if chosen is copy_act:
            self.paste_index(idx + 1)
        elif chosen is pin_act:
            self._history.pin(idx, not item.pinned)
            save_history(self._history)
            self._refresh_list()
        elif chosen is del_act:
            self._history.remove(idx)
            save_history(self._history)
            self._refresh_list()

    def _on_clear_clicked(self) -> None:
        unpinned = sum(1 for i in self._history.items if not i.pinned)
        if unpinned == 0:
            QMessageBox.information(self, "Clear history", "Nothing to clear.")
            return
        menu = QMenu(self)
        clear_unpinned = menu.addAction(f"Clear {unpinned} unpinned item(s)")
        menu.addSeparator()
        clear_all = menu.addAction("Clear EVERYTHING (including pinned)")
        chosen = menu.exec(self._clear_btn.mapToGlobal(self._clear_btn.rect().bottomLeft()))
        if chosen is clear_unpinned:
            self.clear_all(keep_pinned=True)
        elif chosen is clear_all:
            reply = QMessageBox.question(
                self,
                "Clear everything?",
                "Delete all clipboard items, including pinned?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.clear_all(keep_pinned=False)

    # ---------------------------------------------------- geometry / anim

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
