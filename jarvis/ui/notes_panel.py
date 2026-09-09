"""Notes panel — slide-in markdown notebook with voice capture."""

from __future__ import annotations

import logging
import webbrowser

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
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from jarvis.tools.local.notes_store import (
    append_to_note,
    create_note,
    delete_all_notes,
    delete_note,
    ensure_root,
    find_note_by_title,
    list_notes,
    load_note,
    notes_root,
    save_note,
)

log = logging.getLogger(__name__)

_DEFAULT_PANEL_WIDTH = 560
_MIN_PANEL_WIDTH = 360
_MAX_PANEL_WIDTH = 900
_SIDE_MARGIN = 16
_ANIM_MS = 300

_BG = "#0d0d0d"
_BORDER = "#1f1f1f"
_TEXT = "#ffffff"
_TEXT_DIM = "#808080"
_CYAN = "#38f4ff"

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


class NotesPanel(QWidget):
    """Slide-in markdown notebook. Frameless, right-edge, always-on-top."""

    _sig_open = Signal()
    _sig_close = Signal()
    _sig_create = Signal(str, str, object)  # title, content, callback list
    _sig_show_note = Signal(str)
    _sig_delete = Signal(str)
    _sig_delete_all = Signal()
    _sig_refresh = Signal()

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

        self._panel_width = max(
            _MIN_PANEL_WIDTH, min(_MAX_PANEL_WIDTH, panel_width)
        )
        self._active_note_id: str | None = None
        self._anim: QPropertyAnimation | None = None
        self._edit_mode = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(800)
        self._save_timer.timeout.connect(self._persist_active_edits)

        self._build_ui()
        self._position_offscreen()
        ensure_root()

        self._sig_open.connect(self._on_open)
        self._sig_close.connect(self._on_close)
        self._sig_create.connect(self._on_create_note)
        self._sig_show_note.connect(self._on_show_note)
        self._sig_delete.connect(self._on_delete_note)
        self._sig_delete_all.connect(self._on_delete_all)
        self._sig_refresh.connect(self._refresh_notes_list)

    # -------------------------------------------------------------- public

    def open_panel(self) -> None:
        self._sig_open.emit()

    def close_panel(self) -> None:
        self._sig_close.emit()

    def create_and_show(self, title: str, content: str) -> str:
        """Voice entry: create a new note, return the final title."""
        result_holder: list[str] = []
        self._sig_create.emit(title, content, result_holder)
        # We need a synchronous title back for the spoken confirmation.
        # The signal/slot is queued; do the work directly on the calling
        # thread by storing the note synchronously (file I/O only) and let
        # the panel refresh through the signal.
        note = create_note(title, content)
        self._active_note_id = note.id
        return note.title

    def append_to_active(self, content: str) -> str | None:
        if self._active_note_id is None:
            return None
        note = load_note(self._active_note_id)
        if note is None:
            return None
        append_to_note(note, content)
        self._sig_show_note.emit(note.id)
        self._sig_refresh.emit()
        return note.title

    def append_by_title(self, title: str, content: str) -> str | None:
        match = find_note_by_title(title)
        if match is None:
            return None
        append_to_note(match, content)
        self._active_note_id = match.id
        self._sig_show_note.emit(match.id)
        self._sig_refresh.emit()
        return match.title

    def read_active(self) -> tuple[str, str] | None:
        if self._active_note_id is None:
            return None
        note = load_note(self._active_note_id)
        if note is None:
            return None
        return (note.title, note.content)

    def read_by_title(self, title: str) -> tuple[str, str] | None:
        match = find_note_by_title(title)
        if match is None:
            return None
        self._active_note_id = match.id
        self._sig_show_note.emit(match.id)
        return (match.title, match.content)

    def delete_active(self) -> str | None:
        if self._active_note_id is None:
            return None
        note = load_note(self._active_note_id)
        if note is None:
            return None
        deleted_title = note.title
        self._sig_delete.emit(note.id)
        return deleted_title

    def delete_by_title(self, title: str) -> str | None:
        match = find_note_by_title(title)
        if match is None:
            return None
        deleted_title = match.title
        self._sig_delete.emit(match.id)
        return deleted_title

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        self.setFixedWidth(self._panel_width)
        self.setObjectName("NotesPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        inner = QWidget()
        inner.setStyleSheet(
            f"QWidget {{ background:{_BG}; border-left:1px solid {_BORDER}; }}"
        )
        il = QVBoxLayout(inner)
        il.setContentsMargins(16, 16, 16, 16)
        il.setSpacing(8)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("NOTES")
        title.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent;"
        )
        hdr.addWidget(title)
        hdr.addStretch(1)

        self._new_btn = QPushButton("NEW")
        self._new_btn.setStyleSheet(_HDR_BTN)
        self._new_btn.clicked.connect(self._on_new_clicked)
        hdr.addWidget(self._new_btn)

        self._edit_btn = QPushButton("EDIT")
        self._edit_btn.setStyleSheet(_HDR_BTN)
        self._edit_btn.clicked.connect(self._on_edit_clicked)
        hdr.addWidget(self._edit_btn)

        self._folder_btn = QPushButton("FOLDER")
        self._folder_btn.setStyleSheet(_HDR_BTN)
        self._folder_btn.setToolTip("Open notes folder in Explorer")
        self._folder_btn.clicked.connect(self._on_open_folder)
        hdr.addWidget(self._folder_btn)

        self._delete_btn = QPushButton("DELETE")
        self._delete_btn.setStyleSheet(_HDR_BTN)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        hdr.addWidget(self._delete_btn)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 24)
        close_btn.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:16pt; background:transparent; border:none;"
        )
        close_btn.clicked.connect(self._on_close)
        hdr.addWidget(close_btn)
        il.addLayout(hdr)

        self._title_label = QLabel("Pick a note or say 'take a note'")
        self._title_label.setWordWrap(True)
        self._title_label.setStyleSheet(
            f"color:{_CYAN}; font-size:13pt; font-weight:300; background:transparent;"
        )
        il.addWidget(self._title_label)

        self._meta_label = QLabel("")
        self._meta_label.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:9pt; background:transparent;"
        )
        il.addWidget(self._meta_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._notes_list = QListWidget()
        self._notes_list.setMinimumWidth(180)
        self._notes_list.setMaximumWidth(240)
        self._notes_list.setStyleSheet(
            f"QListWidget {{ background:{_BG}; color:{_TEXT}; border:1px solid {_BORDER}; "
            "font-size:9pt; }}"
            f"QListWidget::item:selected {{ background:#1a2a2a; color:{_CYAN}; }}"
        )
        self._notes_list.currentItemChanged.connect(self._on_note_selected)
        self._notes_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._notes_list.customContextMenuRequested.connect(self._on_list_context)
        splitter.addWidget(self._notes_list)

        # Stacked view: 0=read (QTextBrowser), 1=edit (QTextEdit)
        self._stack = QStackedWidget()

        self._note_view = QTextBrowser()
        self._note_view.setOpenExternalLinks(True)
        self._note_view.setStyleSheet(
            f"QTextBrowser {{ background:#0a0a0a; color:{_TEXT}; border:1px solid {_BORDER}; "
            "font-size:10pt; padding:8px; }}"
        )
        self._note_view.setPlaceholderText("Note will appear here…")
        self._stack.addWidget(self._note_view)

        self._note_edit = QTextEdit()
        self._note_edit.setStyleSheet(
            f"QTextEdit {{ background:#0a0a0a; color:{_TEXT}; border:1px solid {_CYAN}; "
            "font-size:10pt; padding:8px; }}"
        )
        self._note_edit.textChanged.connect(self._on_edit_text_changed)
        self._stack.addWidget(self._note_edit)

        splitter.addWidget(self._stack)
        splitter.setStretchFactor(1, 1)
        il.addWidget(splitter, 1)

        root.addWidget(inner)
        self._refresh_notes_list()

    # ------------------------------------------------------------- slots

    @Slot()
    def _on_open(self) -> None:
        self._refresh_notes_list()
        self._show_panel()

    @Slot()
    def _on_close(self) -> None:
        self._persist_active_edits()
        if self.isVisible():
            self._slide_out()

    @Slot(str, str, object)
    def _on_create_note(self, title: str, content: str, result_holder: object) -> None:
        # create_note() is run synchronously by create_and_show on the
        # caller's thread; here we just refresh and show on the Qt thread.
        self._refresh_notes_list()
        if self._active_note_id:
            self._select_note(self._active_note_id)
        if not self.isVisible():
            self._show_panel()

    @Slot(str)
    def _on_show_note(self, note_id: str) -> None:
        self._select_note(note_id)
        if not self.isVisible():
            self._show_panel()

    @Slot()
    def _refresh_notes_list(self) -> None:
        selected = self._active_note_id
        self._notes_list.blockSignals(True)
        self._notes_list.clear()
        notes = list_notes()
        for n in notes:
            label = n.title[:60] + ("…" if len(n.title) > 60 else "")
            preview = n.preview
            text = f"{label}\n  {preview}" if preview else label
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, n.id)
            self._notes_list.addItem(item)
            if n.id == selected:
                self._notes_list.setCurrentItem(item)
        self._notes_list.blockSignals(False)
        if not notes:
            self._title_label.setText("No notes yet — say 'take a note …'")
            self._meta_label.setText("")
            self._note_view.clear()
            self._note_edit.clear()
            self._active_note_id = None

    def _select_note(self, note_id: str) -> None:
        for i in range(self._notes_list.count()):
            item = self._notes_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == note_id:
                self._notes_list.setCurrentItem(item)
                self._on_note_selected(item, None)
                return
        # Note may have been created between refreshes — refresh and retry.
        self._refresh_notes_list()
        for i in range(self._notes_list.count()):
            item = self._notes_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == note_id:
                self._notes_list.setCurrentItem(item)
                self._on_note_selected(item, None)
                return

    def _on_note_selected(self, current: QListWidgetItem | None, _prev) -> None:
        if current is None:
            return
        self._persist_active_edits()
        note_id = current.data(Qt.ItemDataRole.UserRole)
        note = load_note(note_id)
        if note is None:
            return
        self._active_note_id = note.id
        self._title_label.setText(note.title)
        self._meta_label.setText(
            f"Updated {note.updated_at[:19].replace('T', ' ')}"
        )
        self._note_view.setMarkdown(note.content)
        self._note_edit.setPlainText(note.content)
        if self._edit_mode:
            self._stack.setCurrentIndex(1)
        else:
            self._stack.setCurrentIndex(0)

    def _on_edit_clicked(self) -> None:
        if self._active_note_id is None:
            return
        self._edit_mode = not self._edit_mode
        if self._edit_mode:
            self._stack.setCurrentIndex(1)
            self._edit_btn.setStyleSheet(_HDR_BTN_ACTIVE)
            self._edit_btn.setText("DONE")
            self._note_edit.setFocus()
        else:
            self._persist_active_edits()
            self._stack.setCurrentIndex(0)
            self._edit_btn.setStyleSheet(_HDR_BTN)
            self._edit_btn.setText("EDIT")
            note = load_note(self._active_note_id)
            if note:
                self._note_view.setMarkdown(note.content)

    def _on_edit_text_changed(self) -> None:
        if self._edit_mode and self._active_note_id:
            self._save_timer.start()

    def _persist_active_edits(self) -> None:
        if not self._edit_mode or self._active_note_id is None:
            return
        note = load_note(self._active_note_id)
        if note is None:
            return
        new_content = self._note_edit.toPlainText()
        if new_content == note.content:
            return
        note.content = new_content
        save_note(note)
        self._meta_label.setText(
            f"Updated {note.updated_at[:19].replace('T', ' ')}"
        )
        self._refresh_notes_list()

    def _on_new_clicked(self) -> None:
        note = create_note("New note", "")
        self._active_note_id = note.id
        self._refresh_notes_list()
        self._select_note(note.id)
        if not self._edit_mode:
            self._on_edit_clicked()

    def _on_open_folder(self) -> None:
        root = notes_root()
        root.mkdir(parents=True, exist_ok=True)
        webbrowser.open(root.as_uri())

    def _on_delete_clicked(self) -> None:
        item = self._notes_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Delete note", "No note selected.")
            return
        note_id = item.data(Qt.ItemDataRole.UserRole)
        note = load_note(note_id)
        label = note.title if note else note_id
        menu = QMenu(self)
        del_one = menu.addAction(f"Delete '{label[:60]}'")
        menu.addSeparator()
        del_all = menu.addAction("Delete ALL notes…")
        chosen = menu.exec(self._delete_btn.mapToGlobal(self._delete_btn.rect().bottomLeft()))
        if chosen is del_one:
            self._confirm_delete(note_id, label)
        elif chosen is del_all:
            self._confirm_delete_all()

    def _on_list_context(self, pos) -> None:
        item = self._notes_list.itemAt(pos)
        if item is None:
            return
        note_id = item.data(Qt.ItemDataRole.UserRole)
        note = load_note(note_id)
        label = note.title if note else note_id
        menu = QMenu(self)
        open_act = menu.addAction("Open")
        menu.addSeparator()
        del_act = menu.addAction(f"Delete '{label[:50]}'")
        chosen = menu.exec(self._notes_list.mapToGlobal(pos))
        if chosen is open_act:
            self._notes_list.setCurrentItem(item)
            self._on_note_selected(item, None)
        elif chosen is del_act:
            self._confirm_delete(note_id, label)

    def _confirm_delete(self, note_id: str, label: str) -> None:
        reply = QMessageBox.question(
            self,
            "Delete note?",
            f"Delete this note permanently?\n\n{label}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._sig_delete.emit(note_id)

    def _confirm_delete_all(self) -> None:
        n = len(list_notes())
        if n == 0:
            QMessageBox.information(self, "Delete all", "No notes to delete.")
            return
        reply = QMessageBox.question(
            self,
            "Delete ALL notes?",
            f"Permanently delete all {n} note(s)?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._sig_delete_all.emit()

    @Slot(str)
    def _on_delete_note(self, note_id: str) -> None:
        if self._active_note_id == note_id:
            self._active_note_id = None
            self._note_view.clear()
            self._note_edit.clear()
            self._title_label.setText("")
            self._meta_label.setText("")
        delete_note(note_id)
        self._refresh_notes_list()

    @Slot()
    def _on_delete_all(self) -> None:
        delete_all_notes()
        self._active_note_id = None
        self._note_view.clear()
        self._note_edit.clear()
        self._title_label.setText("All notes deleted")
        self._meta_label.setText("")
        self._refresh_notes_list()

    # --------------------------------------------------------- geometry

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
