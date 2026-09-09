"""Deep research panel — in-depth research with pause/resume and markdown reports."""

from __future__ import annotations

import logging
import queue
import threading
import webbrowser
from collections.abc import Callable

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    QThread,
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
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from jarvis.tools.local.deep_research_runner import (
    DeepResearchConfig,
    DeepResearchPaused,
    run_deep_research,
)
from jarvis.tools.local.deep_research_store import (
    create_session,
    delete_all_sessions,
    delete_session,
    find_session_by_query,
    latest_paused_session,
    list_sessions,
    load_state,
    read_report,
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


class DeepResearchWorker(QThread):
    progress = Signal(str)
    report_changed = Signal(str)
    paused = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        session_id: str,
        *,
        config: DeepResearchConfig,
        pause_event: threading.Event,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_id = session_id
        self._cfg = config
        self._pause_event = pause_event

    def run(self) -> None:
        state = load_state(self._session_id)
        if state is None:
            self.failed.emit("session not found")
            return
        try:
            def _on_step_progress(msg: str) -> None:
                self.progress.emit(msg)
                self.report_changed.emit(read_report(self._session_id))

            run_deep_research(
                state,
                cfg=self._cfg,
                should_pause=self._pause_event.is_set,
                on_progress=_on_step_progress,
            )
            self.report_changed.emit(read_report(self._session_id))
            self.finished_ok.emit(self._session_id)
        except DeepResearchPaused:
            self.report_changed.emit(read_report(self._session_id))
            self.paused.emit(self._session_id)
        except Exception as exc:
            log.exception("DeepResearchWorker failed")
            err = str(exc)
            if "connect" in err.lower():
                err = "Could not reach Ollama. Is it running? (ollama serve)"
            self.failed.emit(err)


class DeepResearchPanel(QWidget):
    """Slide-in panel for deep research sessions and markdown reports."""

    _sig_start = Signal(str, object)
    _sig_resume = Signal(str, object)
    _sig_pause = Signal()
    _sig_close = Signal()
    _sig_refresh_list = Signal()
    _sig_delete = Signal(str)
    _sig_delete_all = Signal()

    def __init__(
        self,
        *,
        panel_width: int = _DEFAULT_PANEL_WIDTH,
        config_provider: Callable[[], DeepResearchConfig],
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
        self._config_provider = config_provider
        self._pause_event = threading.Event()
        self._worker: DeepResearchWorker | None = None
        self._active_session_id: str | None = None
        self._result_queue: queue.Queue | None = None
        self._anim: QPropertyAnimation | None = None

        self._build_ui()
        self._position_offscreen()
        self._sig_start.connect(self._on_start)
        self._sig_resume.connect(self._on_resume)
        self._sig_pause.connect(self._on_pause)
        self._sig_close.connect(self._on_close)
        self._sig_refresh_list.connect(self._refresh_session_list)
        self._sig_delete.connect(self._on_delete_session)
        self._sig_delete_all.connect(self._on_delete_all_sessions)

    def show_for_query(self, query: str, result_queue: object) -> None:
        self._sig_start.emit(query, result_queue)

    def resume_session(self, session_id: str, result_queue: object | None = None) -> None:
        self._sig_resume.emit(session_id, result_queue)

    def pause_active(self) -> None:
        self._sig_pause.emit()

    def close_panel(self) -> None:
        self._sig_close.emit()

    def resume_latest_paused(self, result_queue: object | None = None) -> str | None:
        state = latest_paused_session()
        if state is None:
            return None
        self.resume_session(state.id, result_queue)
        return state.id

    def delete_by_query(self, query: str) -> str | None:
        """Find and delete the most recent matching session. Returns its query if deleted."""
        state = find_session_by_query(query)
        if state is None:
            return None
        deleted_query = state.query
        self._sig_delete.emit(state.id)
        return deleted_query

    def delete_active(self) -> str | None:
        if self._active_session_id is None:
            return None
        state = load_state(self._active_session_id)
        if state is None:
            return None
        deleted_query = state.query
        self._sig_delete.emit(state.id)
        return deleted_query

    def delete_all(self) -> int:
        sessions = list_sessions()
        n = len(sessions)
        if n == 0:
            return 0
        self._sig_delete_all.emit()
        return n

    def _build_ui(self) -> None:
        self.setFixedWidth(self._panel_width)
        self.setObjectName("DeepResearchPanel")

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
        title = QLabel("DEEP RESEARCH")
        title.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent;"
        )
        hdr.addWidget(title)
        hdr.addStretch(1)

        self._open_folder_btn = QPushButton("FOLDER")
        self._open_folder_btn.setStyleSheet(_HDR_BTN)
        self._open_folder_btn.setToolTip("Open reports folder")
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        hdr.addWidget(self._open_folder_btn)

        self._pause_btn = QPushButton("PAUSE")
        self._pause_btn.setStyleSheet(_HDR_BTN)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        hdr.addWidget(self._pause_btn)

        self._resume_btn = QPushButton("RESUME")
        self._resume_btn.setStyleSheet(_HDR_BTN)
        self._resume_btn.clicked.connect(self._on_resume_clicked)
        hdr.addWidget(self._resume_btn)

        self._delete_btn = QPushButton("DELETE")
        self._delete_btn.setStyleSheet(_HDR_BTN)
        self._delete_btn.setToolTip("Delete the selected session")
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

        self._query_label = QLabel()
        self._query_label.setWordWrap(True)
        self._query_label.setStyleSheet(
            f"color:{_CYAN}; font-size:12pt; font-weight:300; background:transparent;"
        )
        il.addWidget(self._query_label)

        self._progress_label = QLabel("Idle")
        self._progress_label.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:9pt; background:transparent;"
        )
        il.addWidget(self._progress_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._session_list = QListWidget()
        self._session_list.setMinimumWidth(160)
        self._session_list.setMaximumWidth(220)
        self._session_list.setStyleSheet(
            f"QListWidget {{ background:{_BG}; color:{_TEXT}; border:1px solid {_BORDER}; "
            "font-size:9pt; }}"
            f"QListWidget::item:selected {{ background:#1a2a2a; color:{_CYAN}; }}"
        )
        self._session_list.currentItemChanged.connect(self._on_session_selected)
        self._session_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._session_list.customContextMenuRequested.connect(self._on_session_context_menu)
        splitter.addWidget(self._session_list)

        self._report_view = QTextBrowser()
        self._report_view.setOpenExternalLinks(True)
        self._report_view.setStyleSheet(
            f"QTextBrowser {{ background:#0a0a0a; color:{_TEXT}; border:1px solid {_BORDER}; "
            "font-size:10pt; padding:8px; }}"
        )
        self._report_view.setPlaceholderText("Report will appear here…")
        splitter.addWidget(self._report_view)
        splitter.setStretchFactor(1, 1)
        il.addWidget(splitter, 1)

        root.addWidget(inner)
        self._refresh_session_list()

    @Slot(str, object)
    def _on_start(self, query: str, result_queue: object) -> None:
        self._stop_worker(clear_pause=False)
        self._pause_event.clear()
        self._result_queue = result_queue  # type: ignore[assignment]
        try:
            state = create_session(query)
        except Exception as exc:
            self._fail_queue(exc)
            return
        self._active_session_id = state.id
        self._query_label.setText(query)
        self._progress_label.setText(state.progress_message)
        self._report_view.setMarkdown(read_report(state.id))
        self._refresh_session_list()
        self._select_session(state.id)
        self._start_worker(state.id)
        self._show_panel()

    @Slot(str, object)
    def _on_resume(self, session_id: str, result_queue: object) -> None:
        state = load_state(session_id)
        if state is None:
            return
        if state.status == "completed":
            self._active_session_id = session_id
            self._query_label.setText(state.query)
            self._report_view.setMarkdown(read_report(session_id))
            self._refresh_session_list()
            self._select_session(session_id)
            self._show_panel()
            return
        self._stop_worker(clear_pause=False)
        self._pause_event.clear()
        self._result_queue = result_queue  # type: ignore[assignment]
        self._active_session_id = session_id
        self._query_label.setText(state.query)
        self._progress_label.setText("Resuming…")
        self._report_view.setMarkdown(read_report(session_id))
        state.status = "running"
        from jarvis.tools.local.deep_research_store import save_state

        save_state(state)
        self._refresh_session_list()
        self._select_session(session_id)
        self._start_worker(session_id)
        self._show_panel()

    @Slot()
    def _on_pause_clicked(self) -> None:
        self._on_pause()

    @Slot()
    def _on_pause(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._pause_event.set()
        self._pause_btn.setStyleSheet(_HDR_BTN_ACTIVE)
        self._progress_label.setText("Pausing after current step…")

    @Slot()
    def _on_resume_clicked(self) -> None:
        item = self._session_list.currentItem()
        if item is None:
            sid = self._active_session_id
        else:
            sid = item.data(Qt.ItemDataRole.UserRole)
        if sid:
            self._sig_resume.emit(sid, None)

    @Slot(str)
    def _on_progress(self, msg: str) -> None:
        self._progress_label.setText(msg)

    @Slot(str)
    def _on_report_changed(self, md: str) -> None:
        self._report_view.setMarkdown(md)

    @Slot(str)
    def _on_worker_paused(self, session_id: str) -> None:
        self._pause_btn.setStyleSheet(_HDR_BTN)
        self._progress_label.setText("Paused — resume anytime from RESUME or the list.")
        self._finish_queue(
            "Deep research paused, sir. Progress saved. "
            "Say resume deep research or use the panel."
        )
        self._worker = None
        self._refresh_session_list()
        self._select_session(session_id)

    @Slot(str)
    def _on_worker_finished(self, session_id: str) -> None:
        self._pause_btn.setStyleSheet(_HDR_BTN)
        self._progress_label.setText("Complete.")
        self._finish_queue(
            "Deep research complete, sir. The full report is in the panel."
        )
        self._worker = None
        self._refresh_session_list()
        self._select_session(session_id)

    @Slot(str)
    def _on_worker_failed(self, error: str) -> None:
        self._progress_label.setText(f"Error: {error}")
        self._report_view.append(f"\n\n**Error:** {error}")
        self._fail_queue(RuntimeError(error))
        self._worker = None
        self._refresh_session_list()

    @Slot()
    def _on_close(self) -> None:
        self._pause_event.set()
        self._stop_worker()
        if self.isVisible():
            self._slide_out()

    def _on_open_folder(self) -> None:
        from jarvis.tools.local.deep_research_store import deep_research_root

        path = deep_research_root()
        path.mkdir(parents=True, exist_ok=True)
        webbrowser.open(path.as_uri())

    @Slot()
    def _on_delete_clicked(self) -> None:
        item = self._session_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Delete session", "No session selected.")
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        state = load_state(sid)
        label = state.query if state else sid
        menu = QMenu(self)
        del_one = menu.addAction(f"Delete '{label[:60]}'")
        menu.addSeparator()
        del_all = menu.addAction("Delete ALL sessions…")
        chosen = menu.exec(self._delete_btn.mapToGlobal(self._delete_btn.rect().bottomLeft()))
        if chosen is del_one:
            self._confirm_and_delete(sid, label)
        elif chosen is del_all:
            self._confirm_and_delete_all()

    def _on_session_context_menu(self, pos) -> None:
        item = self._session_list.itemAt(pos)
        if item is None:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        state = load_state(sid)
        label = state.query if state else sid
        menu = QMenu(self)
        open_act = menu.addAction("Open in panel")
        menu.addSeparator()
        del_act = menu.addAction(f"Delete '{label[:50]}'")
        chosen = menu.exec(self._session_list.mapToGlobal(pos))
        if chosen is open_act:
            self._session_list.setCurrentItem(item)
            self._on_session_selected(item, None)
        elif chosen is del_act:
            self._confirm_and_delete(sid, label)

    def _confirm_and_delete(self, session_id: str, label: str) -> None:
        reply = QMessageBox.question(
            self,
            "Delete session?",
            f"Delete this session permanently?\n\n{label}\n\n"
            "The markdown report and saved state will be removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._sig_delete.emit(session_id)

    def _confirm_and_delete_all(self) -> None:
        n = len(list_sessions())
        if n == 0:
            QMessageBox.information(self, "Delete all", "No sessions to delete.")
            return
        reply = QMessageBox.question(
            self,
            "Delete ALL sessions?",
            f"Permanently delete all {n} deep research session(s)?\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._sig_delete_all.emit()

    @Slot(str)
    def _on_delete_session(self, session_id: str) -> None:
        if (
            self._active_session_id == session_id
            and self._worker is not None
            and self._worker.isRunning()
        ):
            self._pause_event.set()
            self._stop_worker()
        delete_session(session_id)
        if self._active_session_id == session_id:
            self._active_session_id = None
            self._query_label.setText("")
            self._progress_label.setText("Session deleted.")
            self._report_view.clear()
        self._refresh_session_list()

    @Slot()
    def _on_delete_all_sessions(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._pause_event.set()
            self._stop_worker()
        delete_all_sessions()
        self._active_session_id = None
        self._query_label.setText("")
        self._progress_label.setText("All sessions deleted.")
        self._report_view.clear()
        self._refresh_session_list()

    def _on_session_selected(self, current: QListWidgetItem | None, _prev) -> None:
        if current is None:
            return
        sid = current.data(Qt.ItemDataRole.UserRole)
        if not sid:
            return
        state = load_state(sid)
        if state:
            self._query_label.setText(state.query)
            self._progress_label.setText(state.progress_message or state.status)
        self._report_view.setMarkdown(read_report(sid))

    def _start_worker(self, session_id: str) -> None:
        cfg = self._config_provider()
        self._worker = DeepResearchWorker(
            session_id,
            config=cfg,
            pause_event=self._pause_event,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.report_changed.connect(self._on_report_changed)
        self._worker.paused.connect(self._on_worker_paused)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _stop_worker(self, *, clear_pause: bool = True) -> None:
        if clear_pause:
            self._pause_event.set()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)
        self._worker = None
        if clear_pause:
            self._pause_event.clear()

    def _finish_queue(self, message: str) -> None:
        if self._result_queue is not None:
            try:
                self._result_queue.put_nowait((message, self._active_session_id))
            except Exception:
                pass
            self._result_queue = None

    def _fail_queue(self, exc: BaseException) -> None:
        if self._result_queue is not None:
            try:
                self._result_queue.put_nowait(exc)
            except Exception:
                pass
            self._result_queue = None

    @Slot()
    def _refresh_session_list(self) -> None:
        selected = self._active_session_id
        self._session_list.blockSignals(True)
        self._session_list.clear()
        for state in list_sessions():
            label = f"{state.query[:40]}{'…' if len(state.query) > 40 else ''}"
            item = QListWidgetItem(f"[{state.status}] {label}")
            item.setData(Qt.ItemDataRole.UserRole, state.id)
            item.setToolTip(state.id)
            self._session_list.addItem(item)
            if state.id == selected:
                self._session_list.setCurrentItem(item)
        self._session_list.blockSignals(False)

    def _select_session(self, session_id: str) -> None:
        for i in range(self._session_list.count()):
            item = self._session_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == session_id:
                self._session_list.setCurrentItem(item)
                break

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
