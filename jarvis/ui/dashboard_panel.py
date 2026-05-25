"""Live system dashboard — slide-in HUD with CPU/RAM/mic/mode/Ollama/etc."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

_DEFAULT_PANEL_WIDTH = 440
_SIDE_MARGIN = 16
_ANIM_MS = 280

_BG = "#0d0d0d"
_BG_CARD = "#141414"
_BORDER = "#1f1f1f"
_TEXT = "#ffffff"
_TEXT_DIM = "#808080"
_TEXT_MID = "#b0b0b0"
_CYAN = "#38f4ff"
_GREEN = "#3dd97a"
_AMBER = "#e6b94a"
_RED = "#ff5c5c"

_HDR_BTN = (
    f"QPushButton {{ color:{_TEXT_DIM}; background:transparent; border:none; "
    "font-size:8pt; letter-spacing:2px; font-weight:300; padding:0 6px; }}"
    f"QPushButton:hover {{ color:{_TEXT}; }}"
)

_SECTION_LABEL = (
    f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
    "letter-spacing:3px; background:transparent;"
)
_VALUE_LABEL = (
    f"color:{_TEXT}; font-size:14pt; font-weight:300; background:transparent;"
)
_SUB_LABEL = (
    f"color:{_TEXT_MID}; font-size:9pt; font-weight:300; background:transparent;"
)
_CARD_STYLE = (
    f"QFrame {{ background:{_BG_CARD}; border:1px solid {_BORDER}; "
    "border-radius:6px; padding:10px; }}"
)


def _color_for_percent(pct: float) -> str:
    if pct >= 85:
        return _RED
    if pct >= 65:
        return _AMBER
    return _GREEN


class _MeterBar(QWidget):
    """Thin horizontal meter (0-100). Color shifts green→amber→red."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value: float = 0.0
        self.setFixedHeight(6)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, v: float) -> None:
        self._value = max(0.0, min(100.0, float(v)))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.fillRect(self.rect(), QColor("#181818"))
            w = int(self.rect().width() * (self._value / 100.0))
            color = QColor(_color_for_percent(self._value))
            p.fillRect(0, 0, w, self.rect().height(), color)
        finally:
            p.end()


class DashboardPanel(QWidget):
    """Slide-in live dashboard. QTimer-driven, ~1.5 s refresh."""

    _sig_open = Signal()
    _sig_close = Signal()

    def __init__(
        self,
        *,
        sm: object | None = None,
        amplitude_latch: object | None = None,
        config_provider: Callable[[], object] | None = None,
        deep_research_count_provider: Callable[[], tuple[int, int]] | None = None,
        notes_count_provider: Callable[[], int] | None = None,
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

        self._sm = sm
        self._amplitude_latch = amplitude_latch
        self._config_provider = config_provider
        self._deep_research_count_provider = deep_research_count_provider
        self._notes_count_provider = notes_count_provider

        self._panel_width = panel_width
        self._anim: QPropertyAnimation | None = None
        self._started = time.time()
        self._last_cpu_pct: float = 0.0

        self._build_ui()
        self._position_offscreen()

        self._sig_open.connect(self._on_open)
        self._sig_close.connect(self._on_close)

        # Refresh timer — only ticks while visible.
        self._timer = QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self._refresh)
        # Prime psutil's per-interval CPU sample.
        try:
            import psutil

            psutil.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            pass

    # -------------------------------------------------------- public api

    def open_panel(self) -> None:
        self._sig_open.emit()

    def close_panel(self) -> None:
        self._sig_close.emit()

    # ----------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self.setFixedWidth(self._panel_width)
        self.setObjectName("DashboardPanel")

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
        title = QLabel("DASHBOARD")
        title.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:4px; background:transparent;"
        )
        hdr.addWidget(title)
        hdr.addStretch(1)

        self._clock_label = QLabel("--:--")
        self._clock_label.setStyleSheet(
            f"color:{_CYAN}; font-size:11pt; font-weight:300; "
            "letter-spacing:2px; background:transparent;"
        )
        hdr.addWidget(self._clock_label)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 24)
        close_btn.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:16pt; background:transparent; border:none;"
        )
        close_btn.clicked.connect(self._on_close)
        hdr.addWidget(close_btn)
        il.addLayout(hdr)

        # --- Mode + uptime card -----------------------------------------
        mode_card = QFrame()
        mode_card.setStyleSheet(_CARD_STYLE)
        mode_lay = QHBoxLayout(mode_card)
        mode_lay.setContentsMargins(12, 8, 12, 8)
        left_col = QVBoxLayout()
        left_label = QLabel("MODE")
        left_label.setStyleSheet(_SECTION_LABEL)
        self._mode_value = QLabel("—")
        self._mode_value.setStyleSheet(_VALUE_LABEL)
        left_col.addWidget(left_label)
        left_col.addWidget(self._mode_value)
        mode_lay.addLayout(left_col, 1)

        right_col = QVBoxLayout()
        up_label = QLabel("UPTIME")
        up_label.setStyleSheet(_SECTION_LABEL)
        self._uptime_value = QLabel("0 s")
        self._uptime_value.setStyleSheet(_VALUE_LABEL)
        right_col.addWidget(up_label)
        right_col.addWidget(self._uptime_value)
        mode_lay.addLayout(right_col, 1)
        il.addWidget(mode_card)

        # --- CPU + RAM cards --------------------------------------------
        sys_card = QFrame()
        sys_card.setStyleSheet(_CARD_STYLE)
        sys_lay = QGridLayout(sys_card)
        sys_lay.setContentsMargins(12, 8, 12, 8)
        sys_lay.setVerticalSpacing(4)

        cpu_label = QLabel("CPU")
        cpu_label.setStyleSheet(_SECTION_LABEL)
        self._cpu_value = QLabel("0%")
        self._cpu_value.setStyleSheet(_VALUE_LABEL)
        self._cpu_meter = _MeterBar()
        sys_lay.addWidget(cpu_label, 0, 0)
        sys_lay.addWidget(self._cpu_value, 1, 0)
        sys_lay.addWidget(self._cpu_meter, 2, 0)

        ram_label = QLabel("RAM")
        ram_label.setStyleSheet(_SECTION_LABEL)
        self._ram_value = QLabel("0%")
        self._ram_value.setStyleSheet(_VALUE_LABEL)
        self._ram_sub = QLabel("")
        self._ram_sub.setStyleSheet(_SUB_LABEL)
        self._ram_meter = _MeterBar()
        sys_lay.addWidget(ram_label, 0, 1)
        sys_lay.addWidget(self._ram_value, 1, 1)
        sys_lay.addWidget(self._ram_meter, 2, 1)
        sys_lay.addWidget(self._ram_sub, 3, 1)
        il.addWidget(sys_card)

        # --- Mic card ----------------------------------------------------
        mic_card = QFrame()
        mic_card.setStyleSheet(_CARD_STYLE)
        mic_lay = QVBoxLayout(mic_card)
        mic_lay.setContentsMargins(12, 8, 12, 8)
        mic_label = QLabel("MIC LEVEL")
        mic_label.setStyleSheet(_SECTION_LABEL)
        self._mic_value = QLabel("idle")
        self._mic_value.setStyleSheet(_VALUE_LABEL)
        self._mic_meter = _MeterBar()
        mic_lay.addWidget(mic_label)
        mic_lay.addWidget(self._mic_value)
        mic_lay.addWidget(self._mic_meter)
        il.addWidget(mic_card)

        # --- Models card -------------------------------------------------
        mdl_card = QFrame()
        mdl_card.setStyleSheet(_CARD_STYLE)
        mdl_lay = QGridLayout(mdl_card)
        mdl_lay.setContentsMargins(12, 8, 12, 8)
        mdl_lay.setVerticalSpacing(4)
        main_lbl = QLabel("MAIN MODEL")
        main_lbl.setStyleSheet(_SECTION_LABEL)
        self._main_model_value = QLabel("—")
        self._main_model_value.setStyleSheet(_VALUE_LABEL)
        self._main_model_value.setWordWrap(True)
        plan_lbl = QLabel("RESEARCH PLANNER / WORKER")
        plan_lbl.setStyleSheet(_SECTION_LABEL)
        self._planner_value = QLabel("—")
        self._planner_value.setStyleSheet(_SUB_LABEL)
        self._planner_value.setWordWrap(True)
        mdl_lay.addWidget(main_lbl, 0, 0)
        mdl_lay.addWidget(self._main_model_value, 1, 0)
        mdl_lay.addWidget(plan_lbl, 2, 0)
        mdl_lay.addWidget(self._planner_value, 3, 0)
        il.addWidget(mdl_card)

        # --- Activity card ----------------------------------------------
        act_card = QFrame()
        act_card.setStyleSheet(_CARD_STYLE)
        act_lay = QGridLayout(act_card)
        act_lay.setContentsMargins(12, 8, 12, 8)
        act_lay.setVerticalSpacing(4)

        act_lbl = QLabel("FOREGROUND APP")
        act_lbl.setStyleSheet(_SECTION_LABEL)
        self._fg_value = QLabel("—")
        self._fg_value.setStyleSheet(_VALUE_LABEL)
        self._fg_value.setWordWrap(True)
        act_lay.addWidget(act_lbl, 0, 0, 1, 2)
        act_lay.addWidget(self._fg_value, 1, 0, 1, 2)

        notes_lbl = QLabel("NOTES")
        notes_lbl.setStyleSheet(_SECTION_LABEL)
        self._notes_value = QLabel("0")
        self._notes_value.setStyleSheet(_VALUE_LABEL)
        act_lay.addWidget(notes_lbl, 2, 0)
        act_lay.addWidget(self._notes_value, 3, 0)

        dr_lbl = QLabel("DEEP RESEARCH")
        dr_lbl.setStyleSheet(_SECTION_LABEL)
        self._dr_value = QLabel("0")
        self._dr_value.setStyleSheet(_VALUE_LABEL)
        self._dr_sub = QLabel("")
        self._dr_sub.setStyleSheet(_SUB_LABEL)
        act_lay.addWidget(dr_lbl, 2, 1)
        act_lay.addWidget(self._dr_value, 3, 1)
        act_lay.addWidget(self._dr_sub, 4, 1)

        il.addWidget(act_card)

        il.addStretch(1)
        root.addWidget(inner)

    # ------------------------------------------------------------ slots

    @Slot()
    def _on_open(self) -> None:
        self._refresh()
        if not self._timer.isActive():
            self._timer.start()
        self._show_panel()

    @Slot()
    def _on_close(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        if self.isVisible():
            self._slide_out()

    def _refresh(self) -> None:
        self._clock_label.setText(time.strftime("%H:%M"))
        self._refresh_uptime()
        self._refresh_mode()
        self._refresh_system()
        self._refresh_mic()
        self._refresh_models()
        self._refresh_foreground()
        self._refresh_activity()

    def _refresh_uptime(self) -> None:
        secs = int(time.time() - self._started)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            text = f"{h}h {m}m"
        elif m:
            text = f"{m}m {s}s"
        else:
            text = f"{s}s"
        self._uptime_value.setText(text)

    def _refresh_mode(self) -> None:
        if self._sm is None:
            self._mode_value.setText("—")
            return
        try:
            mode = getattr(self._sm, "mode", None)
            cs = getattr(self._sm, "conversational_state", None)
            mode_name = getattr(mode, "name", str(mode)) if mode else "?"
            cs_name = getattr(cs, "name", str(cs)) if cs else ""
            if cs_name and mode_name == "ACTIVE":
                self._mode_value.setText(f"{mode_name.title()} · {cs_name.title()}")
            else:
                self._mode_value.setText(mode_name.title())
        except Exception:  # noqa: BLE001
            self._mode_value.setText("?")

    def _refresh_system(self) -> None:
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            self._last_cpu_pct = cpu
            self._cpu_value.setText(f"{cpu:.0f}%")
            self._cpu_meter.set_value(cpu)
            used_gb = (mem.total - mem.available) / (1024**3)
            total_gb = mem.total / (1024**3)
            self._ram_value.setText(f"{mem.percent:.0f}%")
            self._ram_sub.setText(f"{used_gb:.1f} / {total_gb:.1f} GB")
            self._ram_meter.set_value(mem.percent)
        except Exception:  # noqa: BLE001
            self._cpu_value.setText("?")
            self._ram_value.setText("?")

    def _refresh_mic(self) -> None:
        if self._amplitude_latch is None:
            self._mic_value.setText("idle")
            self._mic_meter.set_value(0.0)
            return
        try:
            amp = float(self._amplitude_latch.latest())
        except Exception:  # noqa: BLE001
            amp = 0.0
        pct = max(0.0, min(100.0, amp * 100.0))
        if pct < 2:
            self._mic_value.setText("silent")
        elif pct < 15:
            self._mic_value.setText(f"{pct:.0f}%")
        else:
            self._mic_value.setText(f"{pct:.0f}% (talking)")
        self._mic_meter.set_value(pct)

    def _refresh_models(self) -> None:
        if self._config_provider is None:
            return
        try:
            cfg = self._config_provider()
        except Exception:  # noqa: BLE001
            return
        try:
            main_model = getattr(cfg.llm, "model", "?")
            self._main_model_value.setText(str(main_model))
            rcfg = getattr(cfg, "research", None)
            planner = (getattr(rcfg, "planner_model", "") or "main") if rcfg else "main"
            worker = (getattr(rcfg, "worker_model", "") or "main") if rcfg else "main"
            self._planner_value.setText(f"planner: {planner}   worker: {worker}")
        except Exception:  # noqa: BLE001
            pass

    def _refresh_foreground(self) -> None:
        try:
            title = _foreground_window_title()
        except Exception:  # noqa: BLE001
            title = ""
        self._fg_value.setText(title or "—")

    def _refresh_activity(self) -> None:
        if self._notes_count_provider is not None:
            try:
                n = int(self._notes_count_provider())
                self._notes_value.setText(str(n))
            except Exception:  # noqa: BLE001
                self._notes_value.setText("?")
        if self._deep_research_count_provider is not None:
            try:
                total, paused = self._deep_research_count_provider()
                self._dr_value.setText(str(total))
                self._dr_sub.setText(f"{paused} paused" if paused else "")
            except Exception:  # noqa: BLE001
                self._dr_value.setText("?")

    # ----------------------------------------------------- geometry / anim

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


def _foreground_window_title() -> str:
    """Best-effort Windows-only foreground window title."""
    try:
        import ctypes
        import ctypes.wintypes as wt

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value.strip()
    except Exception:  # noqa: BLE001
        return ""
