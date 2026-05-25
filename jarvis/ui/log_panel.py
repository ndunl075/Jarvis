"""Live log viewer panel — tails %APPDATA%/Jarvis/logs/jarvis.log."""

from __future__ import annotations

import logging
import os
import sys
import webbrowser
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

_DEFAULT_PANEL_WIDTH = 720
_SIDE_MARGIN = 16
_ANIM_MS = 280
_POLL_MS = 1000
_MAX_TAIL_BYTES = 200_000  # initial read cap so a giant log file isn't yanked in whole

_BG = "#0d0d0d"
_BORDER = "#1f1f1f"
_TEXT = "#e8e8e8"
_TEXT_DIM = "#808080"
_CYAN = "#38f4ff"

_LEVELS = ("ALL", "DEBUG", "INFO", "WARNING", "ERROR")
_LEVEL_RANK = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

_HDR_BTN = (
    f"QPushButton {{ color:{_TEXT_DIM}; background:transparent; border:none; "
    "font-size:8pt; letter-spacing:2px; font-weight:300; padding:0 6px; }}"
    f"QPushButton:hover {{ color:{_TEXT}; }}"
)


def default_log_path() -> Path:
    """%APPDATA%/Jarvis/logs/jarvis.log — same path the frozen launcher uses."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Jarvis" / "logs" / "jarvis.log"
    return Path.home() / ".jarvis" / "logs" / "jarvis.log"


def _line_level(line: str) -> str | None:
    """Best-effort: pull the level out of a logging line formatted as
    ``HH:MM:SS [LEVEL] name: msg``. Returns None if no level found."""
    if not line:
        return None
    # Find first '[' and the next ']' near the start.
    lb = line.find("[")
    if lb == -1 or lb > 12:
        return None
    rb = line.find("]", lb)
    if rb == -1:
        return None
    tok = line[lb + 1 : rb].strip().upper()
    if tok in _LEVEL_RANK:
        return tok
    return None


class LogPanel(QWidget):
    """Slide-in tail viewer for jarvis.log with level filter + search."""

    _sig_open = Signal()
    _sig_close = Signal()

    def __init__(
        self,
        *,
        log_path: Path | None = None,
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
        self._log_path = log_path or default_log_path()
        self._read_pos: int = 0
        self._all_lines: list[str] = []
        self._level_filter: str = "ALL"
        self._search: str = ""

        self._build_ui()
        self._position_offscreen()

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._tail_new)

        self._sig_open.connect(self._on_open)
        self._sig_close.connect(self._on_close)

    # ------------------------------------------------------------ public

    def open_panel(self) -> None:
        self._sig_open.emit()

    def close_panel(self) -> None:
        self._sig_close.emit()

    # ----------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self.setFixedWidth(self._panel_width)
        self.setObjectName("LogPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        inner = QWidget()
        inner.setStyleSheet(
            f"QWidget {{ background:{_BG}; border-left:1px solid {_BORDER}; }}"
        )
        il = QVBoxLayout(inner)
        il.setContentsMargins(16, 16, 16, 16)
        il.setSpacing(8)

        # --- header --------------------------------------------------------
        hdr = QHBoxLayout()
        title = QLabel("LOGS")
        title.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent;"
        )
        hdr.addWidget(title)
        hdr.addStretch(1)

        self._open_file_btn = QPushButton("OPEN FILE")
        self._open_file_btn.setStyleSheet(_HDR_BTN)
        self._open_file_btn.setToolTip("Open jarvis.log in Explorer")
        self._open_file_btn.clicked.connect(self._on_open_file)
        hdr.addWidget(self._open_file_btn)

        self._clear_btn = QPushButton("CLEAR VIEW")
        self._clear_btn.setStyleSheet(_HDR_BTN)
        self._clear_btn.setToolTip("Clear what's shown (does NOT delete the log file)")
        self._clear_btn.clicked.connect(self._on_clear_view)
        hdr.addWidget(self._clear_btn)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 24)
        close_btn.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:16pt; background:transparent; border:none;"
        )
        close_btn.clicked.connect(self._on_close)
        hdr.addWidget(close_btn)
        il.addLayout(hdr)

        # --- path + filter row --------------------------------------------
        path_row = QHBoxLayout()
        self._path_label = QLabel(str(self._log_path))
        self._path_label.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:9pt; background:transparent;"
        )
        self._path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._path_label.setWordWrap(True)
        path_row.addWidget(self._path_label, 1)
        il.addLayout(path_row)

        filt = QHBoxLayout()
        level_lbl = QLabel("Level:")
        level_lbl.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:9pt; background:transparent;"
        )
        filt.addWidget(level_lbl)
        self._level = QComboBox()
        for lv in _LEVELS:
            self._level.addItem(lv, lv)
        self._level.currentIndexChanged.connect(self._on_level_changed)
        self._level.setStyleSheet(
            f"QComboBox {{ color:{_TEXT}; background:#161616; "
            f"border:1px solid {_BORDER}; padding:2px 6px; }}"
        )
        filt.addWidget(self._level)
        filt.addSpacing(12)

        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search (substring)…")
        self._search_box.textChanged.connect(self._on_search_changed)
        self._search_box.setStyleSheet(
            f"QLineEdit {{ color:{_TEXT}; background:#0a0a0a; "
            f"border:1px solid {_BORDER}; padding:4px 8px; }}"
            f"QLineEdit:focus {{ border:1px solid {_CYAN}; }}"
        )
        filt.addWidget(self._search_box, 1)
        il.addLayout(filt)

        # --- text view -----------------------------------------------------
        self._view = QTextEdit()
        self._view.setReadOnly(True)
        self._view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._view.setStyleSheet(
            "QTextEdit { background:#0a0a0a; color:#d8d8d8; "
            "font-family: Consolas, 'Courier New', monospace; "
            f"font-size:9pt; border:1px solid {_BORDER}; padding:6px; }}"
        )
        il.addWidget(self._view, 1)

        root.addWidget(inner)

    # ----------------------------------------------------------- log io

    def _read_initial(self) -> None:
        self._all_lines.clear()
        self._read_pos = 0
        try:
            if not self._log_path.is_file():
                self._view.setPlainText(
                    f"(log file not yet created at {self._log_path})"
                )
                return
            size = self._log_path.stat().st_size
            start = max(0, size - _MAX_TAIL_BYTES)
            with self._log_path.open("rb") as f:
                f.seek(start)
                # If we cut mid-line, discard the first partial line.
                if start > 0:
                    f.readline()
                data = f.read()
                self._read_pos = f.tell()
            text = data.decode("utf-8", errors="replace")
            for line in text.splitlines():
                self._all_lines.append(line)
            self._render()
        except OSError as e:
            self._view.setPlainText(f"(could not read log: {e})")

    def _tail_new(self) -> None:
        try:
            if not self._log_path.is_file():
                return
            size = self._log_path.stat().st_size
            if size < self._read_pos:
                # Truncated / rotated — restart.
                self._read_pos = 0
                self._all_lines.clear()
            if size == self._read_pos:
                return
            with self._log_path.open("rb") as f:
                f.seek(self._read_pos)
                data = f.read()
                self._read_pos = f.tell()
            text = data.decode("utf-8", errors="replace")
            new_lines = text.splitlines()
            if not new_lines:
                return
            self._all_lines.extend(new_lines)
            # Keep buffer bounded.
            if len(self._all_lines) > 5000:
                self._all_lines = self._all_lines[-3000:]
            self._render(append=True, new_lines=new_lines)
        except OSError:
            pass

    # ------------------------------------------------------- rendering

    def _line_passes_filter(self, line: str) -> bool:
        if self._level_filter != "ALL":
            lvl = _line_level(line)
            min_rank = _LEVEL_RANK.get(self._level_filter, 0)
            if lvl is None:
                return False
            if _LEVEL_RANK.get(lvl, -1) < min_rank:
                return False
        if self._search and self._search.lower() not in line.lower():
            return False
        return True

    def _color_for_line(self, line: str) -> QColor:
        lvl = _line_level(line)
        if lvl == "ERROR" or lvl == "CRITICAL":
            return QColor("#ff7070")
        if lvl == "WARNING":
            return QColor("#ffd060")
        if lvl == "DEBUG":
            return QColor("#888888")
        return QColor("#d8d8d8")

    def _render(
        self,
        *,
        append: bool = False,
        new_lines: list[str] | None = None,
    ) -> None:
        if not append:
            self._view.clear()
            iter_lines = self._all_lines
        else:
            iter_lines = new_lines or []
        cursor = self._view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        for line in iter_lines:
            if not self._line_passes_filter(line):
                continue
            fmt = QTextCharFormat()
            fmt.setForeground(self._color_for_line(line))
            cursor.insertText(line + "\n", fmt)
        # Auto-scroll to bottom only when user hasn't scrolled up.
        sb = self._view.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    @Slot(int)
    def _on_level_changed(self, _idx: int) -> None:
        self._level_filter = self._level.currentData() or "ALL"
        self._render()

    @Slot(str)
    def _on_search_changed(self, text: str) -> None:
        self._search = text.strip()
        self._render()

    def _on_open_file(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_path.is_file():
            self._log_path.touch()
        webbrowser.open(self._log_path.as_uri())

    def _on_clear_view(self) -> None:
        self._view.clear()
        # Don't clear self._all_lines so a re-render restores them; this
        # mirrors what users expect from a "clear view, keep file" action.

    # ------------------------------------------------------------ slots

    @Slot()
    def _on_open(self) -> None:
        self._read_initial()
        if not self._timer.isActive():
            self._timer.start()
        self._show_panel()

    @Slot()
    def _on_close(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        if self.isVisible():
            self._slide_out()

    # ----------------------------------------------------- geometry/anim

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
