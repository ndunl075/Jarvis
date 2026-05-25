"""System tray icon for Jarvis. Phase 5, [IMPL] + threading bridge.

Reflects Mode in icon + menu, dispatches menu actions back through
StateMachine.set_mode(), and re-renders on ModeChanged events from the bus.

Owns no audio modules: every action goes through the state machine, and
state observation goes through the event bus. Tray-side changes therefore
remain consistent with hotkey-side and pipeline-side changes automatically.

Threading bridge
----------------
Two loops run concurrently in production:

  - The audio asyncio loop: owns StateMachine, LifecycleManager, EventBus
    subscribers. Single-threaded by contract (StateMachine docstring is
    explicit on this).
  - The Qt main thread: owns this TrayIcon (a QObject) and any future UI.
    Qt requires that QObject UI mutations happen on the thread that
    constructed them.

Two directions to bridge:

  - Tray -> audio (menu click -> set_mode): the slot fires on the Qt
    thread. We marshal to the audio loop with
    asyncio.run_coroutine_threadsafe(_set_mode_coro, audio_loop), passing
    the new Mode value. The resulting ModeChanged is the visible effect.
    The audio loop is injected at construction (NOT
    asyncio.get_event_loop(), which is deprecated and brittle).

  - Audio -> tray (ModeChanged -> repaint icon/menu): the bus subscriber
    runs on the audio loop's thread. It cannot touch the QSystemTrayIcon
    directly. We post via QMetaObject.invokeMethod(self, "_on_mode_name",
    Qt.QueuedConnection, Q_ARG(str, name)). The slot runs on the next
    Qt event-loop tick. The Mode enum is marshalled as its .name string
    because str is already a registered metatype; saves wrestling with
    qRegisterMetaType for our 3-value enum.

A _alive flag, flipped to False in close(), guards the slot against
late-arriving queued calls after teardown — common at shutdown when a
final ModeChanged fires while the tray is being destroyed.

Tray availability
-----------------
If QSystemTrayIcon.isSystemTrayAvailable() returns False, this module
cannot do its job. We surface that to the user with a QMessageBox and
the composition root quits. Per Phase 5 scope: no windowed-status
fallback (deferred to Phase 7 polish). 99%+ of Windows installs have a
functioning tray.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Q_ARG, QMetaObject, Qt, Slot
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QMessageBox, QSystemTrayIcon

from jarvis.core.config import HotkeysConfig
from jarvis.core.events import EventBus, ModeChanged
from jarvis.core.state_machine import Mode, StateMachine

log = logging.getLogger(__name__)


def ensure_system_tray_available() -> bool:
    """Returns True if the system tray is usable. Otherwise pops a one-time
    QMessageBox.warning explaining that Jarvis cannot run without one, and
    returns False so the composition root can quit cleanly. Safe to call
    after QApplication construction; not before."""
    if QSystemTrayIcon.isSystemTrayAvailable():
        return True
    QMessageBox.warning(
        None,  # type: ignore[arg-type]
        "Jarvis",
        "Jarvis requires a system tray, but none is available on this "
        "desktop. Jarvis will exit.",
    )
    return False


def _paint_circle_icon(color: QColor, *, bar: bool = False) -> QIcon:
    """Programmatic tray icon: filled circle, optionally with a horizontal
    bar across it (for the MUTED variant). Replaceable with PNG assets
    later by swapping the body of _build_icons."""
    pix = QPixmap(32, 32)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    try:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(4, 4, 24, 24)
        if bar:
            p.setBrush(QColor(255, 255, 255))
            p.drawRect(6, 14, 20, 4)
    finally:
        p.end()
    return QIcon(pix)


class TrayIcon(QSystemTrayIcon):
    """Tray icon + menu. Lives on the Qt main thread. Constructed AFTER
    QApplication so QSystemTrayIcon's metaobject machinery is initialized."""

    def __init__(
        self,
        *,
        sm: StateMachine,
        bus: EventBus,
        audio_loop: asyncio.AbstractEventLoop,
        hotkeys: HotkeysConfig,
        on_open_settings: Callable[[], None],
        on_quit: Callable[[], None],
        on_mode_request: Callable[[Mode], Any] | None = None,
        on_open_dashboard: Callable[[], None] | None = None,
        on_open_notes: Callable[[], None] | None = None,
        on_open_help: Callable[[], None] | None = None,
        on_open_clipboard_history: Callable[[], None] | None = None,
        on_open_logs: Callable[[], None] | None = None,
        on_open_command_palette: Callable[[], None] | None = None,
        on_open_tutorial: Callable[[], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._sm = sm
        self._bus = bus
        self._audio_loop = audio_loop
        self._hotkeys = hotkeys
        self._on_open_settings = on_open_settings
        self._on_quit = on_quit
        self._on_open_dashboard = on_open_dashboard
        self._on_open_notes = on_open_notes
        self._on_open_help = on_open_help
        self._on_open_clipboard_history = on_open_clipboard_history
        self._on_open_logs = on_open_logs
        self._on_open_command_palette = on_open_command_palette
        self._on_open_tutorial = on_open_tutorial
        # Optional coroutine-returning hook the composition root uses to
        # route mode requests through ModeCoordinator. When None (tests,
        # legacy callers) we fall back to direct sm.set_mode, preserving
        # the pre-Phase-6 behaviour.
        self._on_mode_request = on_mode_request
        self._alive: bool = True

        self._icons: dict[Mode, QIcon] = self._build_icons()
        self._menu = QMenu()
        self._status_action: QAction = QAction(self._menu)
        self._status_action.setEnabled(False)
        self._mode_action: QAction = QAction(self._menu)  # Mute/Unmute
        self._sleep_action: QAction = QAction(self._menu)  # Sleep/Wake
        self._settings_action: QAction = QAction("Settings…", self._menu)
        self._command_palette_action: QAction = QAction("Command palette…", self._menu)
        self._dashboard_action: QAction = QAction("Show dashboard", self._menu)
        self._notes_action: QAction = QAction("Open notes", self._menu)
        self._clipboard_action: QAction = QAction("Clipboard history", self._menu)
        self._logs_action: QAction = QAction("Show logs", self._menu)
        self._help_action: QAction = QAction("What can I say?", self._menu)
        self._tutorial_action: QAction = QAction("Show tutorial", self._menu)
        self._about_action: QAction = QAction("About", self._menu)
        self._quit_action: QAction = QAction("Quit Jarvis", self._menu)

        self._mode_action.triggered.connect(self._on_mode_action_clicked)
        self._sleep_action.triggered.connect(self._on_sleep_action_clicked)
        self._settings_action.triggered.connect(self._on_settings_clicked)
        self._command_palette_action.triggered.connect(self._on_command_palette_clicked)
        self._dashboard_action.triggered.connect(self._on_dashboard_clicked)
        self._notes_action.triggered.connect(self._on_notes_clicked)
        self._clipboard_action.triggered.connect(self._on_clipboard_clicked)
        self._logs_action.triggered.connect(self._on_logs_clicked)
        self._help_action.triggered.connect(self._on_help_clicked)
        self._tutorial_action.triggered.connect(self._on_tutorial_clicked)
        self._about_action.triggered.connect(self._on_about_clicked)
        self._quit_action.triggered.connect(self._on_quit_clicked)

        self._assemble_menu()
        self.setContextMenu(self._menu)
        self.setToolTip("Jarvis")

        self._unsubscribe_mode = bus.subscribe(ModeChanged, self._on_mode_event)
        self._render_for_mode(self._sm.mode)

    # -- bus subscriber (audio thread) ------------------------------------

    def _on_mode_event(self, event: ModeChanged) -> None:
        """Runs on the audio loop's thread. Posts the work to the Qt main
        thread via a queued connection. The Mode enum crosses as a string
        because str is a registered Qt metatype."""
        if not self._alive:
            return
        QMetaObject.invokeMethod(
            self,
            "_on_mode_name",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, event.new.name),
        )

    @Slot(str)
    def _on_mode_name(self, name: str) -> None:
        """Runs on the Qt main thread. Guarded against late-arriving calls
        post-close()."""
        if not self._alive:
            return
        try:
            mode = Mode[name]
        except KeyError:
            log.warning("tray received unknown mode name: %r", name)
            return
        self._render_for_mode(mode)

    # -- rendering --------------------------------------------------------

    def _render_for_mode(self, mode: Mode) -> None:
        self.setIcon(self._icons[mode])
        self._status_action.setText(f"Status: {mode.name}")
        self._refresh_toggle_actions(mode)

    def _refresh_toggle_actions(self, mode: Mode) -> None:
        """One of Mute/Unmute is shown, never both; same for Sleep/Wake.
        Hotkey labels come from injected HotkeysConfig — tray does NOT
        register the global hotkey itself (ui/hotkeys.py owns that)."""
        mute_hk = _format_hotkey(self._hotkeys.mute)
        if mode is Mode.MUTED:
            self._mode_action.setText(_with_shortcut("Unmute", mute_hk))
            self._mode_action.setData(Mode.ACTIVE.name)
        else:  # ACTIVE or SLEEPING
            self._mode_action.setText(_with_shortcut("Mute", mute_hk))
            self._mode_action.setData(Mode.MUTED.name)

        if mode is Mode.SLEEPING:
            self._sleep_action.setText("Wake")
            self._sleep_action.setData(Mode.ACTIVE.name)
        else:  # ACTIVE or MUTED
            self._sleep_action.setText("Sleep")
            self._sleep_action.setData(Mode.SLEEPING.name)

    def _assemble_menu(self) -> None:
        m = self._menu
        m.addAction(self._status_action)
        m.addSeparator()
        m.addAction(self._mode_action)
        m.addAction(self._sleep_action)
        m.addSeparator()
        if self._on_open_help is not None:
            m.addAction(self._help_action)
        if self._on_open_command_palette is not None:
            cp_hk = _format_hotkey(getattr(self._hotkeys, "command_palette", "") or "")
            self._command_palette_action.setText(
                _with_shortcut("Command palette…", cp_hk)
            )
            m.addAction(self._command_palette_action)
        if self._on_open_dashboard is not None:
            m.addAction(self._dashboard_action)
        if self._on_open_notes is not None:
            m.addAction(self._notes_action)
        if self._on_open_clipboard_history is not None:
            m.addAction(self._clipboard_action)
        if self._on_open_logs is not None:
            m.addAction(self._logs_action)
        if self._on_open_tutorial is not None:
            m.addAction(self._tutorial_action)
        any_extra = any(
            cb is not None
            for cb in (
                self._on_open_help,
                self._on_open_command_palette,
                self._on_open_dashboard,
                self._on_open_notes,
                self._on_open_clipboard_history,
                self._on_open_logs,
                self._on_open_tutorial,
            )
        )
        if any_extra:
            m.addSeparator()
        settings_hk = _format_hotkey(self._hotkeys.open_settings)
        self._settings_action.setText(_with_shortcut("Settings…", settings_hk))
        m.addAction(self._settings_action)
        m.addAction(self._about_action)
        m.addSeparator()
        m.addAction(self._quit_action)

    def _build_icons(self) -> dict[Mode, QIcon]:
        return {
            Mode.ACTIVE: _paint_circle_icon(QColor(70, 130, 220)),
            Mode.MUTED: _paint_circle_icon(QColor(200, 60, 60), bar=True),
            Mode.SLEEPING: _paint_circle_icon(QColor(120, 120, 120)),
        }

    # -- menu actions (Qt thread) ----------------------------------------

    def _on_mode_action_clicked(self) -> None:
        target_name = self._mode_action.data()
        if isinstance(target_name, str):
            self._dispatch_set_mode(Mode[target_name])

    def _on_sleep_action_clicked(self) -> None:
        target_name = self._sleep_action.data()
        if isinstance(target_name, str):
            self._dispatch_set_mode(Mode[target_name])

    def _on_settings_clicked(self) -> None:
        try:
            self._on_open_settings()
        except Exception:
            log.exception("on_open_settings callback raised")

    def _on_dashboard_clicked(self) -> None:
        if self._on_open_dashboard is None:
            return
        try:
            self._on_open_dashboard()
        except Exception:
            log.exception("on_open_dashboard callback raised")

    def _on_notes_clicked(self) -> None:
        if self._on_open_notes is None:
            return
        try:
            self._on_open_notes()
        except Exception:
            log.exception("on_open_notes callback raised")

    def _on_help_clicked(self) -> None:
        if self._on_open_help is None:
            return
        try:
            self._on_open_help()
        except Exception:
            log.exception("on_open_help callback raised")

    def _on_command_palette_clicked(self) -> None:
        if self._on_open_command_palette is None:
            return
        try:
            self._on_open_command_palette()
        except Exception:
            log.exception("on_open_command_palette callback raised")

    def _on_clipboard_clicked(self) -> None:
        if self._on_open_clipboard_history is None:
            return
        try:
            self._on_open_clipboard_history()
        except Exception:
            log.exception("on_open_clipboard_history callback raised")

    def _on_logs_clicked(self) -> None:
        if self._on_open_logs is None:
            return
        try:
            self._on_open_logs()
        except Exception:
            log.exception("on_open_logs callback raised")

    def _on_tutorial_clicked(self) -> None:
        if self._on_open_tutorial is None:
            return
        try:
            self._on_open_tutorial()
        except Exception:
            log.exception("on_open_tutorial callback raised")

    def _on_about_clicked(self) -> None:
        QMessageBox.about(
            None,  # type: ignore[arg-type]
            "About Jarvis",
            "Jarvis — local Windows voice assistant.",
        )

    def _on_quit_clicked(self) -> None:
        try:
            self._on_quit()
        except Exception:
            log.exception("on_quit callback raised")

    def _dispatch_set_mode(self, new: Mode) -> None:
        """Marshal a Mode request onto the audio loop. Uses the injected
        on_mode_request coroutine when present (production: routes through
        ModeCoordinator), otherwise falls back to direct sm.set_mode.
        Fire-and-forget; exceptions on the audio side are logged there,
        not propagated back to the Qt thread."""
        if self._on_mode_request is not None:
            request = self._on_mode_request
            async def _coro() -> None:
                result = request(new)
                if asyncio.iscoroutine(result):
                    await result
        else:
            async def _coro() -> None:
                self._sm.set_mode(new)
        try:
            asyncio.run_coroutine_threadsafe(_coro(), self._audio_loop)
        except Exception:
            log.exception("failed to dispatch set_mode(%s) to audio loop", new)

    # -- shutdown --------------------------------------------------------

    def close(self) -> None:
        """Unsubscribe from the bus and stop accepting queued invocations.
        Safe to call multiple times. Called by the composition root during
        shutdown; the queued ModeChanged that arrives mid-teardown will be
        no-op'd by the _alive guard in _on_mode_name."""
        if not self._alive:
            return
        self._alive = False
        try:
            self._unsubscribe_mode()
        except Exception:
            log.exception("tray unsubscribe failed")
        self.hide()


# --- helpers --------------------------------------------------------------


def _format_hotkey(raw: str | None) -> str:
    """Render config hotkey strings (e.g. "ctrl+shift+m") in the
    capitalized form users expect to see in menus ("Ctrl+Shift+M"). None
    or empty -> empty string (no shortcut shown)."""
    if not raw:
        return ""
    parts = [p.strip().capitalize() for p in raw.split("+") if p.strip()]
    return "+".join(parts)


def _with_shortcut(label: str, shortcut: str) -> str:
    """Append a right-aligned shortcut to a Qt menu item's text. Qt menus
    render the segment after a tab as right-aligned shortcut text. Pure
    display — does NOT bind the key; ui/hotkeys.py owns global binding."""
    if not shortcut:
        return label
    return f"{label}\t{shortcut}"
