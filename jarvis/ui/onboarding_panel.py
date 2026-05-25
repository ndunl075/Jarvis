"""First-run onboarding — friendly 3-step intro to Jarvis.

Auto-shown on first launch when ``general.first_run_completed`` is False.
After the user clicks **Finish** or **Skip**, the flag flips True and the
panel is dismissed; re-openable any time from the tray menu → **Show
tutorial**.

The three steps:
  1. **Welcome + mic test** — live waveform from the amplitude latch
     proves the mic is hot.
  2. **Wake word test** — subscribes to ConversationalStateChanged →
     LISTENING and marks success the first time it fires.
  3. **Try a command** — buttons that open the Help panel or the
     Command palette so the user can see what's possible right away.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import (
    QMetaObject,
    Qt,
    QTimer,
    Q_ARG,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from jarvis.core.events import ConversationalStateChanged, EventBus

log = logging.getLogger(__name__)

_BG = "#0d0d0d"
_BORDER = "#1f1f1f"
_TEXT = "#ffffff"
_TEXT_DIM = "#808080"
_TEXT_MID = "#b0b0b0"
_CYAN = "#38f4ff"
_GREEN = "#3dd97a"
_AMBER = "#e6b94a"

_PANEL_W = 620
_PANEL_H = 520

_BTN_PRIMARY = (
    f"QPushButton {{ color:#000; background:{_CYAN}; border:none; "
    "padding:8px 18px; font-size:10pt; font-weight:500; }}"
    f"QPushButton:hover {{ background:#7df8ff; }}"
)
_BTN_SECONDARY = (
    f"QPushButton {{ color:{_TEXT}; background:transparent; "
    f"border:1px solid {_BORDER}; padding:8px 18px; font-size:10pt; }}"
    f"QPushButton:hover {{ border:1px solid {_CYAN}; color:{_CYAN}; }}"
)
_BTN_LINK = (
    f"QPushButton {{ color:{_TEXT_DIM}; background:transparent; border:none; "
    "font-size:9pt; }}"
    f"QPushButton:hover {{ color:{_TEXT}; }}"
)


class _MicBars(QWidget):
    """Simple amplitude bars driven by ~30 Hz QTimer reads off a latch."""

    def __init__(
        self,
        amplitude_latch: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._latch = amplitude_latch
        self._level: float = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

    def _tick(self) -> None:
        if self._latch is None:
            self._level = 0.0
        else:
            try:
                raw = float(self._latch.latest())
            except Exception:  # noqa: BLE001
                raw = 0.0
            # EMA smoothing
            self._level = 0.75 * self._level + 0.25 * raw
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            rect = self.rect()
            p.fillRect(rect, QColor("#0a0a0a"))
            n_bars = 24
            gap = 4
            bar_w = (rect.width() - gap * (n_bars + 1)) / n_bars
            center_y = rect.center().y()
            for i in range(n_bars):
                pos = i / max(1, n_bars - 1)
                # Bell-like response so center bars react more strongly.
                weight = 1.0 - abs(pos - 0.5) * 1.6
                weight = max(0.15, weight)
                h = max(2.0, weight * self._level * rect.height())
                x = gap + i * (bar_w + gap)
                color = QColor(_CYAN)
                p.fillRect(int(x), int(center_y - h / 2), int(bar_w), int(h), color)
        finally:
            p.end()


class OnboardingPanel(QWidget):
    """Standalone 3-step intro window."""

    _sig_open = Signal()
    _sig_wake_seen = Signal()

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        amplitude_latch: object | None = None,
        on_finished: Callable[[], None],
        on_open_help: Callable[[], None] | None = None,
        on_open_command_palette: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._bus = bus
        self._amplitude_latch = amplitude_latch
        self._on_finished = on_finished
        self._on_open_help = on_open_help
        self._on_open_command_palette = on_open_command_palette
        self._wake_unsubscribe = None
        self._wake_detected: bool = False

        self._build_ui()
        self._sig_open.connect(self._on_open)
        self._sig_wake_seen.connect(self._on_wake_seen_qt)
        self.hide()

    # ------------------------------------------------------------ public

    def open_panel(self) -> None:
        self._sig_open.emit()

    def close_panel(self) -> None:
        self._cleanup_subscriptions()
        if self.isVisible():
            self.hide()

    # ----------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self.setFixedSize(_PANEL_W, _PANEL_H)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background:{_BG}; border:1px solid {_CYAN}; "
            "border-radius:8px; }}"
        )
        flay = QVBoxLayout(frame)
        flay.setContentsMargins(28, 24, 28, 20)
        flay.setSpacing(14)

        title = QLabel("Welcome to Jarvis")
        title.setStyleSheet(
            f"color:{_CYAN}; font-size:18pt; font-weight:300; "
            "letter-spacing:2px; background:transparent;"
        )
        flay.addWidget(title)

        self._step_label = QLabel("Step 1 of 3")
        self._step_label.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; font-weight:300; "
            "letter-spacing:3px; background:transparent;"
        )
        flay.addWidget(self._step_label)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_step_welcome())
        self._stack.addWidget(self._build_step_wake())
        self._stack.addWidget(self._build_step_command())
        flay.addWidget(self._stack, 1)

        # Footer: Skip on the left, Back/Next/Finish on the right.
        footer = QHBoxLayout()
        self._skip_btn = QPushButton("Skip")
        self._skip_btn.setStyleSheet(_BTN_LINK)
        self._skip_btn.clicked.connect(self._finish)
        footer.addWidget(self._skip_btn)
        footer.addStretch(1)
        self._back_btn = QPushButton("Back")
        self._back_btn.setStyleSheet(_BTN_SECONDARY)
        self._back_btn.clicked.connect(self._on_back)
        footer.addWidget(self._back_btn)
        self._next_btn = QPushButton("Next")
        self._next_btn.setStyleSheet(_BTN_PRIMARY)
        self._next_btn.clicked.connect(self._on_next)
        footer.addWidget(self._next_btn)
        flay.addLayout(footer)

        root.addWidget(frame)
        self._refresh_nav()

    def _build_step_welcome(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        intro = QLabel(
            "Jarvis is a private, local voice assistant. Nothing leaves "
            "your computer unless you ask it to.\n\n"
            "Let's make sure your microphone is working. Speak — the bars "
            "below should react to your voice."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            f"color:{_TEXT_MID}; font-size:11pt; background:transparent;"
        )
        lay.addWidget(intro)

        self._bars = _MicBars(amplitude_latch=self._amplitude_latch)
        lay.addWidget(self._bars)

        tip = QLabel(
            "Not reacting? Check Settings → Voice → Input device, or pick "
            "your headset's microphone there. You can come back here from "
            "the tray menu."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:9pt; background:transparent;"
        )
        lay.addWidget(tip)
        lay.addStretch(1)
        return w

    def _build_step_wake(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        head = QLabel("Say “Hey Jarvis”")
        head.setStyleSheet(
            f"color:{_TEXT}; font-size:14pt; font-weight:300; background:transparent;"
        )
        lay.addWidget(head)

        explain = QLabel(
            "Try saying it now. As soon as the wake-word detector hears "
            "you, the orb on your desktop will glow cyan and this step "
            "will check off — meaning Jarvis is listening for your next "
            "command."
        )
        explain.setWordWrap(True)
        explain.setStyleSheet(
            f"color:{_TEXT_MID}; font-size:11pt; background:transparent;"
        )
        lay.addWidget(explain)

        self._wake_indicator = QLabel("Waiting for “Hey Jarvis”…")
        self._wake_indicator.setStyleSheet(
            f"color:{_AMBER}; font-size:11pt; font-style:italic; "
            "background:transparent;"
        )
        lay.addWidget(self._wake_indicator)

        miss = QLabel(
            "If nothing happens after a few tries, lower the wake-word "
            "sensitivity in Settings → Voice and try again. You can also "
            "Skip this step."
        )
        miss.setWordWrap(True)
        miss.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:9pt; background:transparent;"
        )
        lay.addWidget(miss)
        lay.addStretch(1)
        return w

    def _build_step_command(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        head = QLabel("Try a command")
        head.setStyleSheet(
            f"color:{_TEXT}; font-size:14pt; font-weight:300; background:transparent;"
        )
        lay.addWidget(head)

        intro = QLabel(
            "You're all set. Here are two great ways to explore what "
            "Jarvis can do:"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            f"color:{_TEXT_MID}; font-size:11pt; background:transparent;"
        )
        lay.addWidget(intro)

        suggestions = QLabel(
            "•  Say <b>“what can you do”</b> for a plain-English list.<br>"
            "•  Press <b>Ctrl+Shift+P</b> to open the command palette.<br>"
            "•  Try one of these right now to see how it feels:"
        )
        suggestions.setWordWrap(True)
        suggestions.setStyleSheet(
            f"color:{_TEXT_MID}; font-size:10pt; background:transparent;"
        )
        suggestions.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(suggestions)

        btn_row = QHBoxLayout()
        help_btn = QPushButton("Open “What can I say?”")
        help_btn.setStyleSheet(_BTN_SECONDARY)
        help_btn.clicked.connect(self._open_help)
        btn_row.addWidget(help_btn)

        palette_btn = QPushButton("Open command palette")
        palette_btn.setStyleSheet(_BTN_SECONDARY)
        palette_btn.clicked.connect(self._open_palette)
        btn_row.addWidget(palette_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        lay.addStretch(1)
        outro = QLabel(
            "When you're ready, click <b>Finish</b> below. You can re-open "
            "this tutorial any time from the tray menu."
        )
        outro.setWordWrap(True)
        outro.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:9pt; background:transparent;"
        )
        outro.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(outro)
        return w

    # ----------------------------------------------------------- events

    @Slot()
    def _on_open(self) -> None:
        self._stack.setCurrentIndex(0)
        self._wake_detected = False
        self._wake_indicator.setText("Waiting for “Hey Jarvis”…")
        self._wake_indicator.setStyleSheet(
            f"color:{_AMBER}; font-size:11pt; font-style:italic; "
            "background:transparent;"
        )
        self._refresh_nav()
        self._bars.start()
        self._maybe_subscribe_wake()
        self._center_on_screen()
        self.show()
        self.raise_()

    def _on_back(self) -> None:
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
        self._refresh_nav()

    def _on_next(self) -> None:
        idx = self._stack.currentIndex()
        if idx >= self._stack.count() - 1:
            self._finish()
            return
        self._stack.setCurrentIndex(idx + 1)
        self._refresh_nav()

    def _refresh_nav(self) -> None:
        idx = self._stack.currentIndex()
        last = self._stack.count() - 1
        self._step_label.setText(f"Step {idx + 1} of {self._stack.count()}")
        self._back_btn.setVisible(idx > 0)
        self._next_btn.setText("Finish" if idx == last else "Next")

    def _finish(self) -> None:
        try:
            self._on_finished()
        except Exception:  # noqa: BLE001
            log.exception("onboarding finish callback raised")
        self.close_panel()

    def _open_help(self) -> None:
        if self._on_open_help is not None:
            self._on_open_help()

    def _open_palette(self) -> None:
        if self._on_open_command_palette is not None:
            self._on_open_command_palette()

    # ----------------------------------------- wake-word event bridge

    def _maybe_subscribe_wake(self) -> None:
        if self._bus is None or self._wake_unsubscribe is not None:
            return
        try:
            self._wake_unsubscribe = self._bus.subscribe(
                ConversationalStateChanged, self._on_cs_event
            )
        except Exception:  # noqa: BLE001
            log.exception("onboarding: failed to subscribe to wake bus")

    def _on_cs_event(self, event: ConversationalStateChanged) -> None:
        # Bus thread; marshal to Qt.
        try:
            new_name = event.new.name
        except Exception:  # noqa: BLE001
            return
        if new_name != "LISTENING":
            return
        self._sig_wake_seen.emit()

    @Slot()
    def _on_wake_seen_qt(self) -> None:
        if self._wake_detected:
            return
        self._wake_detected = True
        self._wake_indicator.setText("✓ Heard you, sir.")
        self._wake_indicator.setStyleSheet(
            f"color:{_GREEN}; font-size:11pt; background:transparent;"
        )

    def _cleanup_subscriptions(self) -> None:
        self._bars.stop()
        if self._wake_unsubscribe is not None:
            try:
                self._wake_unsubscribe()
            except Exception:  # noqa: BLE001
                pass
            self._wake_unsubscribe = None

    # ---------------------------------------------------------- geometry

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.left() + (geo.width() - self.width()) // 2
        y = geo.top() + (geo.height() - self.height()) // 2
        self.move(x, y)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._finish()
        else:
            super().keyPressEvent(event)
