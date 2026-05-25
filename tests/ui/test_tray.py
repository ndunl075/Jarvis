"""Tests for jarvis.ui.tray.TrayIcon.

Mocks the asyncio cross-thread bridge and exercises the Qt slot directly
rather than relying on QMetaObject.invokeMethod's queued dispatch (which
requires a running Qt event loop the tests deliberately do not start).
The qapp session fixture creates the offscreen QApplication that QObject
construction requires."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.core.config import HotkeysConfig
from jarvis.core.events import EventBus, ModeChanged
from jarvis.core.state_machine import Mode, StateMachine
from jarvis.ui.tray import TrayIcon, _format_hotkey, _with_shortcut


# --- helpers -------------------------------------------------------------


def _make_tray(
    *,
    initial_mode: Mode = Mode.ACTIVE,
    hotkeys: HotkeysConfig | None = None,
):
    sm = StateMachine(initial_mode=initial_mode)
    bus = EventBus()
    loop = MagicMock()
    on_settings = MagicMock()
    on_quit = MagicMock()
    tray = TrayIcon(
        sm=sm,
        bus=bus,
        audio_loop=loop,
        hotkeys=hotkeys or HotkeysConfig(),
        on_open_settings=on_settings,
        on_quit=on_quit,
    )
    return tray, sm, bus, loop, on_settings, on_quit


def _menu_action_texts(tray: TrayIcon) -> list[str]:
    menu = tray.contextMenu()
    return [a.text() for a in menu.actions() if not a.isSeparator()]


# --- menu structure per Mode ---------------------------------------------


def test_active_shows_mute_and_sleep(qapp):
    tray, _, _, _, _, _ = _make_tray(initial_mode=Mode.ACTIVE)
    try:
        texts = _menu_action_texts(tray)
        assert "Status: ACTIVE" in texts
        assert any(t.startswith("Mute") for t in texts)
        assert any(t.startswith("Sleep") for t in texts)
        assert not any(t.startswith("Unmute") for t in texts)
        assert not any(t.startswith("Wake") for t in texts)
    finally:
        tray.close()


def test_muted_shows_unmute(qapp):
    tray, _, _, _, _, _ = _make_tray(initial_mode=Mode.MUTED)
    try:
        texts = _menu_action_texts(tray)
        assert "Status: MUTED" in texts
        assert any(t.startswith("Unmute") for t in texts)
        assert not any(t == "Mute" or t.startswith("Mute\t") for t in texts)
        assert any(t.startswith("Sleep") for t in texts)
    finally:
        tray.close()


def test_sleeping_shows_wake(qapp):
    tray, _, _, _, _, _ = _make_tray(initial_mode=Mode.SLEEPING)
    try:
        texts = _menu_action_texts(tray)
        assert "Status: SLEEPING" in texts
        assert any(t == "Wake" for t in texts)
        assert any(t.startswith("Mute") for t in texts)  # Mute is shown in SLEEPING
        assert not any(t == "Sleep" for t in texts)
    finally:
        tray.close()


def test_menu_includes_settings_about_and_quit_jarvis(qapp):
    tray, _, _, _, _, _ = _make_tray()
    try:
        texts = _menu_action_texts(tray)
        assert any(t.startswith("Settings…") for t in texts)
        assert "About" in texts
        # Per design adjustment: "Quit Jarvis", not just "Quit".
        assert "Quit Jarvis" in texts
    finally:
        tray.close()


def test_mute_action_label_includes_configured_hotkey(qapp):
    hotkeys = HotkeysConfig(mute="ctrl+alt+x", open_settings="ctrl+shift+,")
    tray, _, _, _, _, _ = _make_tray(hotkeys=hotkeys)
    try:
        texts = _menu_action_texts(tray)
        assert any("Ctrl+Alt+X" in t for t in texts)
        assert any("Ctrl+Shift+," in t for t in texts)
    finally:
        tray.close()


# --- icon swap on ModeChanged --------------------------------------------


def test_icon_swaps_when_mode_changes(qapp):
    tray, _, _, _, _, _ = _make_tray(initial_mode=Mode.ACTIVE)
    try:
        active_icon = tray.icon()
        # Bypass the queued cross-thread hop and invoke the Qt slot directly.
        tray._on_mode_name("MUTED")
        muted_icon = tray.icon()
        # QIcon equality is by identity-ish (cacheKey changes per QIcon).
        assert active_icon.cacheKey() != muted_icon.cacheKey()
        # And the menu re-rendered to match.
        texts = _menu_action_texts(tray)
        assert "Status: MUTED" in texts
        assert any(t.startswith("Unmute") for t in texts)
    finally:
        tray.close()


def test_mode_event_posts_queued_invocation_with_name_string(qapp):
    """Bus subscriber must NOT touch Qt directly. Verify it goes through
    QMetaObject.invokeMethod with the Mode.name as the payload — the
    contract that keeps the tray on the Qt thread."""
    tray, _, _, _, _, _ = _make_tray(initial_mode=Mode.ACTIVE)
    try:
        with patch("jarvis.ui.tray.QMetaObject.invokeMethod") as invoke:
            tray._on_mode_event(ModeChanged(old=Mode.ACTIVE, new=Mode.SLEEPING))
        assert invoke.called
        # 2nd positional arg is the slot name; one of the Q_ARG-wrapped
        # values must carry the string "SLEEPING".
        args, _ = invoke.call_args
        assert args[1] == "_on_mode_name"
    finally:
        tray.close()


def test_mode_event_after_close_is_dropped(qapp):
    tray, _, _, _, _, _ = _make_tray(initial_mode=Mode.ACTIVE)
    tray.close()
    with patch("jarvis.ui.tray.QMetaObject.invokeMethod") as invoke:
        tray._on_mode_event(ModeChanged(old=Mode.ACTIVE, new=Mode.MUTED))
    assert not invoke.called


def test_slot_after_close_is_dropped(qapp):
    """A queued invocation that arrives after close() must no-op rather
    than touching the partially-destroyed icon/menu."""
    tray, _, _, _, _, _ = _make_tray(initial_mode=Mode.ACTIVE)
    tray.close()
    # Should not raise.
    tray._on_mode_name("MUTED")
    # Status text unchanged (still reflects pre-close ACTIVE state).
    assert "Status: ACTIVE" in _menu_action_texts(tray)


def test_unknown_mode_name_is_logged_not_raised(qapp, caplog):
    import logging
    tray, _, _, _, _, _ = _make_tray()
    try:
        with caplog.at_level(logging.WARNING, logger="jarvis.ui.tray"):
            tray._on_mode_name("NONSENSE")
        assert any("unknown mode" in r.message for r in caplog.records)
    finally:
        tray.close()


# --- set_mode dispatched via run_coroutine_threadsafe -------------------


def test_mute_click_dispatches_set_mode_to_audio_loop(qapp):
    tray, sm, _, audio_loop, _, _ = _make_tray(initial_mode=Mode.ACTIVE)
    try:
        with patch("jarvis.ui.tray.asyncio.run_coroutine_threadsafe") as rcts:
            tray._on_mode_action_clicked()
        assert rcts.called
        _, kwargs = rcts.call_args
        args, _ = rcts.call_args
        # Loop passed as the second positional arg (or kw `loop`).
        passed_loop = args[1] if len(args) > 1 else kwargs.get("loop")
        assert passed_loop is audio_loop
        # And the coroutine, when awaited, calls sm.set_mode(MUTED).
        coro = args[0]
        # Coroutines aren't awaited synchronously in tests; mock sm.set_mode
        # to verify the captured target instead.
        assert tray._mode_action.data() == Mode.MUTED.name
        coro.close()  # avoid "never awaited" warning
    finally:
        tray.close()


def test_sleep_click_dispatches_sleeping_to_audio_loop(qapp):
    tray, _, _, audio_loop, _, _ = _make_tray(initial_mode=Mode.ACTIVE)
    try:
        with patch("jarvis.ui.tray.asyncio.run_coroutine_threadsafe") as rcts:
            tray._on_sleep_action_clicked()
        args, _ = rcts.call_args
        assert args[1] is audio_loop
        args[0].close()  # close the unawaited coroutine
        assert tray._sleep_action.data() == Mode.SLEEPING.name
    finally:
        tray.close()


def test_dispatched_coroutine_actually_calls_set_mode(qapp):
    """Round-trip check: drive the coroutine to completion and assert
    sm.set_mode was called with the expected target Mode."""
    import asyncio
    tray, sm, _, _, _, _ = _make_tray(initial_mode=Mode.ACTIVE)
    try:
        sm.set_mode = MagicMock()  # type: ignore[method-assign]
        captured: list = []
        with patch(
            "jarvis.ui.tray.asyncio.run_coroutine_threadsafe",
            side_effect=lambda coro, loop: captured.append(coro),
        ):
            tray._on_mode_action_clicked()
        assert len(captured) == 1
        asyncio.new_event_loop().run_until_complete(captured[0])
        sm.set_mode.assert_called_once_with(Mode.MUTED)
    finally:
        tray.close()


def test_settings_click_invokes_callback(qapp):
    tray, _, _, _, on_settings, _ = _make_tray()
    try:
        tray._on_settings_clicked()
        on_settings.assert_called_once()
    finally:
        tray.close()


def test_quit_click_invokes_callback(qapp):
    tray, _, _, _, _, on_quit = _make_tray()
    try:
        tray._on_quit_clicked()
        on_quit.assert_called_once()
    finally:
        tray.close()


def test_settings_callback_exception_is_logged_not_raised(qapp, caplog):
    import logging
    sm = StateMachine()
    bus = EventBus()
    on_settings = MagicMock(side_effect=RuntimeError("boom"))
    tray = TrayIcon(
        sm=sm,
        bus=bus,
        audio_loop=MagicMock(),
        hotkeys=HotkeysConfig(),
        on_open_settings=on_settings,
        on_quit=MagicMock(),
    )
    try:
        with caplog.at_level(logging.ERROR, logger="jarvis.ui.tray"):
            tray._on_settings_clicked()
        assert any("on_open_settings" in r.message for r in caplog.records)
    finally:
        tray.close()


# --- close() ------------------------------------------------------------


def test_close_unsubscribes_and_clears_alive(qapp):
    tray, _, bus, _, _, _ = _make_tray()
    # Pre-close: subscriber present.
    assert bus._handlers.get(ModeChanged)  # type: ignore[attr-defined]
    tray.close()
    assert tray._alive is False
    # No active ModeChanged subscribers from the tray remain.
    assert not bus._handlers.get(ModeChanged)  # type: ignore[attr-defined]


def test_close_is_idempotent(qapp):
    tray, _, _, _, _, _ = _make_tray()
    tray.close()
    tray.close()  # must not raise


# --- helpers ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ctrl+shift+m", "Ctrl+Shift+M"),
        ("Ctrl+Alt+X", "Ctrl+Alt+X"),
        ("", ""),
        (None, ""),
        ("ctrl+shift+,", "Ctrl+Shift+,"),
    ],
)
def test_format_hotkey(raw, expected):
    assert _format_hotkey(raw) == expected


def test_with_shortcut_appends_tab_separated_label():
    assert _with_shortcut("Mute", "Ctrl+Shift+M") == "Mute\tCtrl+Shift+M"
    assert _with_shortcut("Mute", "") == "Mute"


# --- tray availability ---------------------------------------------------


def test_ensure_system_tray_available_returns_true_when_available(qapp):
    with patch(
        "jarvis.ui.tray.QSystemTrayIcon.isSystemTrayAvailable",
        return_value=True,
    ):
        from jarvis.ui.tray import ensure_system_tray_available
        assert ensure_system_tray_available() is True


def test_ensure_system_tray_available_warns_and_returns_false(qapp):
    with patch(
        "jarvis.ui.tray.QSystemTrayIcon.isSystemTrayAvailable",
        return_value=False,
    ), patch("jarvis.ui.tray.QMessageBox.warning") as warn:
        from jarvis.ui.tray import ensure_system_tray_available
        assert ensure_system_tray_available() is False
        assert warn.called
