"""Tests for JarvisApp — the handlers that used to be closures in run().

Every test here exercises a code path that could not previously be
reached: as nested functions inside a 611-line run(), these handlers only
existed once the audio thread, the Qt UI, the LLM client and the tool
registry had all been built. As methods, they need a bare JarvisApp() with
two or three attributes set on it.

Nothing here constructs a real audio stack, a real Qt widget or a real
event loop. The point of the class is that it no longer has to.

Covered:
  - _on_quit: the ordered shutdown sequence, and that it stays ordered
  - _on_quit: the tool confirmer is closed FIRST, before the audio join
  - _on_quit: stop_event is signalled through the loop, never called direct
  - _on_quit: idempotence, and that a failing tray.hide() does not abort it
  - _on_config_change: diffing, snapshot advance, ConfigChanged publish
  - _request_mode: returns the coordinator's coroutine, does not await it
  - _tray_open / _tray_open_palette: the not-yet-built-panel case
  - _open_settings: lazy creation, reuse, and what it wires the window to
  - _on_test_voice / _submit_palette_text: Qt → audio-loop dispatch
  - _on_onboarding_finished, _set_deep_research_ultra
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.app import _AUDIO_SHUTDOWN_TIMEOUT, JarvisApp
from jarvis.core.config import JarvisConfig
from jarvis.core.events import ConfigChanged
from jarvis.core.state_machine import Mode
from tests._typing import as_mock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The panels _on_quit closes conditionally, and the method it closes each
# with. Order matters: this is the teardown order run() had.
_OPTIONAL_PANELS = [
    ("deep_research_panel", "close_panel"),
    ("notes_panel", "close_panel"),
    ("dashboard_panel", "close_panel"),
    ("help_panel", "close_panel"),
    ("clipboard_panel", "close_panel"),
    ("log_panel", "close_panel"),
    ("command_palette", "close_palette"),
    ("onboarding_panel", "close_panel"),
]


def _quittable_app(order: list[str] | None = None, *, with_panels: bool = True) -> JarvisApp:
    """A JarvisApp with only the attributes _on_quit touches."""
    app = JarvisApp()

    def rec(name):
        if order is None:
            return None
        # Bound to a local so the narrowing above survives into the
        # closure; pyright re-widens a captured name back to its
        # declared type inside a lambda.
        sink = order
        return lambda *a, **k: sink.append(name)

    app.confirmer = MagicMock()
    app.confirmer.close = MagicMock(side_effect=rec("confirmer_close"))
    app.audio_loop = MagicMock()
    app.audio_loop.call_soon_threadsafe = MagicMock(side_effect=rec("stop_event_signalled"))
    app.stop_event = MagicMock()
    app.audio_thread = MagicMock()
    app.audio_thread.is_alive.return_value = True
    app.audio_thread.join = MagicMock(side_effect=rec("thread_join"))
    app.qt_app = MagicMock()
    app.qt_app.quit = MagicMock(side_effect=rec("qt_quit"))

    app.tray = MagicMock()
    app.tray.hide = MagicMock(side_effect=rec("tray_hide"))
    app.tray.close = MagicMock(side_effect=rec("tray_close"))
    app.orb = MagicMock()
    app.orb.close = MagicMock(side_effect=rec("orb_close"))
    app.hotkeys = MagicMock()
    app.hotkeys.close = MagicMock(side_effect=rec("hotkeys_close"))
    app.research_panel = MagicMock()
    app.research_panel.close_panel = MagicMock(side_effect=rec("research_close"))

    if with_panels:
        for name, closer in _OPTIONAL_PANELS:
            panel = MagicMock()
            setattr(panel, closer, MagicMock(side_effect=rec(name)))
            setattr(app, name, panel)
        app.settings_window = MagicMock()
        app.settings_window.close = MagicMock(side_effect=rec("settings_window"))

    return app


# ---------------------------------------------------------------------------
# _on_quit — the ordered shutdown sequence
# ---------------------------------------------------------------------------


def test_on_quit_runs_the_documented_shutdown_order():
    """The sequence in the module docstring, asserted end to end.

    Signal the audio loop, hide the tray, join the audio thread, close the
    Qt-owned objects, quit the app. Reordering any of it deadlocks or
    crashes on exit, and no other test in the suite would notice.
    """
    order: list[str] = []
    app = _quittable_app(order)

    app._on_quit()

    assert order == [
        "confirmer_close",
        "stop_event_signalled",
        "tray_hide",
        "thread_join",
        "tray_close",
        "orb_close",
        "hotkeys_close",
        "research_close",
        "deep_research_panel",
        "notes_panel",
        "dashboard_panel",
        "help_panel",
        "clipboard_panel",
        "log_panel",
        "command_palette",
        "onboarding_panel",
        "settings_window",
        "qt_quit",
    ]


def test_on_quit_signals_stop_event_through_the_loop_not_directly():
    """stop_event lives on the audio loop; the Qt thread must not touch it.

    call_soon_threadsafe hands the *bound set method* to the loop. Calling
    stop_event.set() here instead would set an asyncio.Event from the wrong
    thread, which is exactly the bug this asserts against.
    """
    app = _quittable_app()

    app._on_quit()

    as_mock(app.audio_loop.call_soon_threadsafe).assert_called_once_with(app.stop_event.set)
    as_mock(app.stop_event.set).assert_not_called()


def test_on_quit_joins_with_the_shutdown_timeout_before_quitting():
    app = _quittable_app()

    app._on_quit()

    as_mock(app.audio_thread.join).assert_called_once_with(timeout=_AUDIO_SHUTDOWN_TIMEOUT)
    assert _AUDIO_SHUTDOWN_TIMEOUT == 10.0


def test_on_quit_skips_the_join_when_the_thread_already_stopped():
    order: list[str] = []
    app = _quittable_app(order)
    as_mock(app.audio_thread.is_alive).return_value = False

    app._on_quit()

    as_mock(app.audio_thread.join).assert_not_called()
    assert "thread_join" not in order
    assert order[-1] == "qt_quit"


def test_on_quit_survives_a_failing_tray_hide():
    """tray.hide() is best-effort; the audio stack must still be drained."""
    order: list[str] = []
    app = _quittable_app(order)
    as_mock(app.tray.hide).side_effect = RuntimeError("tray already gone")

    app._on_quit()

    # confirmer_close now precedes it; the point stands — signalling the
    # audio loop happens before the join, whatever the tray did.
    assert order.index("stop_event_signalled") < order.index("thread_join")
    assert "thread_join" in order
    assert order[-1] == "qt_quit"


def test_on_quit_closes_the_confirmer_before_signalling_the_audio_loop():
    """Shutdown-during-prompt, at the composition root.

    An open confirmation dialog parks a coroutine on the audio loop
    waiting for a verdict. If _on_quit signalled the loop and joined the
    thread while that prompt was still up, the join would sit there until
    the prompt's own watchdog expired. Closing the confirmer first denies
    the prompt and releases the coroutine before shutdown even begins, so
    the join has nothing to wait for.
    """
    order: list[str] = []
    app = _quittable_app(order)

    app._on_quit()

    assert order.index("confirmer_close") < order.index("stop_event_signalled")
    assert order.index("confirmer_close") < order.index("thread_join")


def test_on_quit_tolerates_a_confirmer_that_was_never_built():
    """The tray-unavailable exit and every partial boot leave it None."""
    app = _quittable_app()
    app.confirmer = None

    app._on_quit()

    as_mock(app.qt_app.quit).assert_called_once()


def test_on_quit_survives_a_failing_confirmer_close():
    """A confirmer that raises on the way out must not strand the app
    with a live tray icon and no window."""
    order: list[str] = []
    app = _quittable_app(order)
    as_mock(app.confirmer).close = MagicMock(side_effect=RuntimeError("boom"))

    app._on_quit()

    assert order[0] == "stop_event_signalled"
    as_mock(app.qt_app.quit).assert_called_once()


def test_build_confirmer_installs_it_on_the_registry():
    """Step 9b. The registry fails closed without one, so the only thing
    standing between a confirmable tool and a spurious refusal is this
    call happening — and happening before the audio thread starts."""
    app = JarvisApp()
    app.registry = MagicMock()

    with patch("jarvis.app.QtToolConfirmer") as ctor:
        app._build_confirmer()

    ctor.assert_called_once_with()
    assert app.confirmer is ctor.return_value
    app.registry.set_confirmer.assert_called_once_with(ctor.return_value)


def test_start_builds_the_confirmer_before_booting_the_audio_stack():
    """Ordering, asserted rather than commented.

    The audio thread is the thing that can execute a tool. If the
    confirmer were built with the rest of the Qt UI (step 12), there
    would be a live window in which a spoken "type ..." was refused for
    the wrong reason — no confirmer, rather than no approval.
    """
    order: list[str] = []
    app = JarvisApp()

    def rec(name, ret=None):
        def _f(*a, **k):
            order.append(name)
            return ret
        return _f

    with (
        patch.object(JarvisApp, "_load_config", rec("load_config")),
        patch.object(JarvisApp, "_build_audio_stack", rec("build_audio_stack")),
        patch.object(JarvisApp, "_create_qt_app", rec("create_qt_app")),
        patch.object(JarvisApp, "_build_confirmer", rec("build_confirmer")),
        patch.object(JarvisApp, "_boot_audio_stack", rec("boot_audio_stack")),
        patch.object(JarvisApp, "_build_ui", rec("build_ui")),
        patch("jarvis.app.ensure_system_tray_available", return_value=True),
        patch("jarvis.app._wire_event_logging"),
    ):
        app.bus = MagicMock()
        app.qt_app = MagicMock()
        app.qt_app.exec.return_value = 0
        assert app.start() == 0

    assert order.index("create_qt_app") < order.index("build_confirmer")
    assert order.index("build_confirmer") < order.index("boot_audio_stack")
    assert order.index("build_confirmer") < order.index("build_ui")


def test_on_quit_tolerates_panels_that_were_never_built():
    """The tray can be quit before _build_ui finished creating the panels."""
    app = _quittable_app(with_panels=False)

    app._on_quit()

    as_mock(app.qt_app.quit).assert_called_once()


def test_on_quit_tolerates_a_research_panel_that_was_never_built():
    """_build_ui shows the tray — Quit action live — before it builds the
    research panel, so _on_quit has to guard it like every sibling."""
    app = _quittable_app(with_panels=False)
    app.research_panel = None

    app._on_quit()

    as_mock(app.qt_app.quit).assert_called_once()


def test_on_quit_is_idempotent():
    """Tray menu and window-close can both fire it; the second must no-op."""
    app = _quittable_app()

    app._on_quit()
    app._on_quit()

    assert as_mock(app.qt_app.quit).call_count == 1
    assert as_mock(app.audio_loop.call_soon_threadsafe).call_count == 1
    assert as_mock(app.audio_thread.join).call_count == 1


# ---------------------------------------------------------------------------
# _on_config_change — Settings edits become a ConfigChanged on the bus
# ---------------------------------------------------------------------------


def _config_app() -> JarvisApp:
    app = JarvisApp()
    app.cfg = JarvisConfig()
    app.bus = MagicMock()
    app._cfg_snapshot = app.cfg.model_dump(mode="json")
    return app


def test_on_config_change_publishes_only_the_changed_fields():
    app = _config_app()
    app.cfg.tts.speed = 1.4

    app._on_config_change()

    as_mock(app.bus.publish).assert_called_once()
    event = as_mock(app.bus.publish).call_args[0][0]
    assert isinstance(event, ConfigChanged)
    assert event.changed_fields == ("tts.speed",)
    assert event.old.tts.speed != 1.4
    assert event.new.tts.speed == pytest.approx(1.4)


def test_on_config_change_is_a_no_op_when_nothing_changed():
    app = _config_app()

    app._on_config_change()

    as_mock(app.bus.publish).assert_not_called()


def test_on_config_change_advances_the_snapshot():
    """Two edits must publish two distinct diffs, not the same one twice."""
    app = _config_app()

    app.cfg.tts.speed = 1.4
    app._on_config_change()
    app.cfg.tts.volume = 0.5
    app._on_config_change()

    assert [c[0][0].changed_fields for c in as_mock(app.bus.publish).call_args_list] == [
        ("tts.speed",),
        ("tts.volume",),
    ]


def test_on_config_change_resizes_the_research_panel():
    app = _config_app()
    app.research_panel = MagicMock()
    app.cfg.ui.research_panel_width = 600

    app._on_config_change()

    app.research_panel.set_panel_width.assert_called_once_with(600)


def test_on_config_change_leaves_the_research_panel_alone_for_other_fields():
    app = _config_app()
    app.research_panel = MagicMock()
    app.cfg.tts.speed = 1.4

    app._on_config_change()

    app.research_panel.set_panel_width.assert_not_called()


def test_on_research_panel_width_persists_a_user_drag():
    app = _config_app()
    app.research_panel = MagicMock()

    app._on_research_panel_width(555)

    assert app.cfg.ui.research_panel_width == 555
    assert as_mock(app.bus.publish).call_args[0][0].changed_fields == ("ui.research_panel_width",)


def test_on_research_panel_width_ignores_a_drag_that_changed_nothing():
    app = _config_app()

    app._on_research_panel_width(app.cfg.ui.research_panel_width)

    as_mock(app.bus.publish).assert_not_called()


# ---------------------------------------------------------------------------
# _request_mode — tray/hotkey mode changes cross to the audio loop
# ---------------------------------------------------------------------------


def test_request_mode_returns_the_coordinators_coroutine_unawaited():
    """The Qt thread must hand a coroutine back, not run one.

    TrayIcon/HotkeyManager wrap this in run_coroutine_threadsafe; awaiting
    or running it here would drive the ModeCoordinator from the wrong
    thread.
    """
    app = JarvisApp()
    app.mode_coord = MagicMock()
    app.mode_coord.request = AsyncMock()

    coro = app._request_mode(Mode.SLEEPING)

    app.mode_coord.request.assert_called_once_with(Mode.SLEEPING)
    assert asyncio.iscoroutine(coro)
    coro.close()


# ---------------------------------------------------------------------------
# Tray panel handlers
# ---------------------------------------------------------------------------


def test_tray_open_opens_the_panel():
    app = JarvisApp()
    panel = MagicMock()

    app._tray_open(panel)

    panel.open_panel.assert_called_once_with()


def test_tray_open_ignores_a_panel_that_does_not_exist_yet():
    """The tray is built before the panels, so its menu can fire early."""
    JarvisApp()._tray_open(None)  # must not raise


def test_tray_open_palette_reads_the_attribute_at_call_time():
    app = JarvisApp()
    app._tray_open_palette()  # no palette yet: no-op

    app.command_palette = MagicMock()
    app._tray_open_palette()

    app.command_palette.open_palette.assert_called_once_with()


# ---------------------------------------------------------------------------
# _open_settings — lazy window, and what it is wired to
# ---------------------------------------------------------------------------


def test_open_settings_creates_once_and_raises_thereafter():
    app = JarvisApp()
    app.cfg = JarvisConfig()
    app.voices_dir = Path("/voices")

    with patch("jarvis.app.SettingsWindow") as window_cls:
        app._open_settings()
        first = as_mock(app.settings_window)
        app._open_settings()

    assert window_cls.call_count == 1
    assert app.settings_window is first
    assert first.show.call_count == 2
    assert first.raise_.call_count == 2
    assert first.activateWindow.call_count == 2


def test_open_settings_wires_the_window_to_this_apps_handlers():
    app = JarvisApp()
    app.cfg = JarvisConfig()
    # A bare str, not a Path: _open_settings only forwards this value to
    # the (patched) SettingsWindow, and the assertion below compares what
    # arrived against the literal it was given.
    app.voices_dir = "/voices"  # type: ignore[assignment]

    with patch("jarvis.app.SettingsWindow") as window_cls:
        app._open_settings()

    kwargs = window_cls.call_args.kwargs
    assert kwargs["config"] is app.cfg
    assert kwargs["voices_dir"] == "/voices"
    assert kwargs["on_change"] == app._on_config_change
    assert kwargs["on_test_voice"] == app._on_test_voice


def test_open_settings_any_thread_defers_to_the_qt_thread():
    """Hotkeys fire on pynput's thread; SettingsWindow may only be touched
    on the Qt thread, so this must go through the event loop."""
    app = JarvisApp()

    with patch("jarvis.app.QTimer") as timer:
        app._open_settings_any_thread()

    timer.singleShot.assert_called_once_with(0, app._open_settings)


# ---------------------------------------------------------------------------
# Qt → audio-loop dispatch
# ---------------------------------------------------------------------------


def test_on_test_voice_dispatches_speak_onto_the_audio_loop():
    app = JarvisApp()
    app.tts = MagicMock()
    app.tts.speak = AsyncMock()
    app.audio_loop = MagicMock(spec=asyncio.AbstractEventLoop)

    with patch("asyncio.run_coroutine_threadsafe") as dispatch:
        app._on_test_voice("hello, this is jarvis")

    assert dispatch.call_count == 1
    coro, loop = dispatch.call_args[0]
    assert loop is app.audio_loop
    assert asyncio.iscoroutine(coro)
    coro.close()


def _closing_boom(coro, loop):
    """Stand in for a dead loop: reject the coroutine, but close it first
    so the test does not leak a never-awaited coroutine warning."""
    coro.close()
    raise RuntimeError("loop closed")


def test_on_test_voice_swallows_a_dispatch_failure():
    app = JarvisApp()
    app.tts = MagicMock()
    app.tts.speak = AsyncMock()
    app.audio_loop = MagicMock(spec=asyncio.AbstractEventLoop)

    with patch("asyncio.run_coroutine_threadsafe", side_effect=_closing_boom):
        app._on_test_voice("hello")  # must not propagate


def test_submit_palette_text_dispatches_onto_the_audio_loop():
    app = JarvisApp()
    app.audio_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    app._palette_producer = MagicMock()

    with patch("asyncio.run_coroutine_threadsafe") as dispatch:
        app._submit_palette_text("what is the weather")

    coro, loop = dispatch.call_args[0]
    assert loop is app.audio_loop
    assert asyncio.iscoroutine(coro)
    coro.close()


def test_submit_palette_text_swallows_a_dispatch_failure():
    app = JarvisApp()
    app.audio_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    app._palette_producer = MagicMock()

    with patch("asyncio.run_coroutine_threadsafe", side_effect=_closing_boom):
        app._submit_palette_text("hello")  # must not propagate


async def test_consume_palette_text_drains_the_producer():
    app = JarvisApp()
    seen: list[str] = []

    async def producer(transcription: str) -> AsyncIterator[str]:
        seen.append(transcription)
        yield "one"
        yield "two"

    app._palette_producer = producer
    await app._consume_palette_text("open spotify")

    assert seen == ["open spotify"]


async def test_consume_palette_text_swallows_a_producer_failure():
    app = JarvisApp()

    async def producer(transcription: str) -> AsyncIterator[str]:
        raise RuntimeError("router exploded")
        yield  # pragma: no cover

    app._palette_producer = producer
    await app._consume_palette_text("open spotify")  # must not propagate


# ---------------------------------------------------------------------------
# Onboarding + deep-research toggles
# ---------------------------------------------------------------------------


def test_on_onboarding_finished_marks_first_run_complete_once():
    app = _config_app()
    app.cfg.general.first_run_completed = False

    app._on_onboarding_finished()

    assert app.cfg.general.first_run_completed is True
    published = as_mock(app.bus.publish).call_args[0][0]
    assert published.changed_fields == ("general.first_run_completed",)

    app._on_onboarding_finished()
    assert as_mock(app.bus.publish).call_count == 1


def test_on_onboarding_finished_survives_a_failing_persist():
    app = _config_app()
    app.cfg.general.first_run_completed = False
    as_mock(app.bus.publish).side_effect = RuntimeError("bus is gone")

    app._on_onboarding_finished()  # must not propagate

    assert app.cfg.general.first_run_completed is True


def test_set_deep_research_ultra_saves_and_confirms():
    app = JarvisApp()
    app.cfg = JarvisConfig()

    with patch("jarvis.core.config.save_config") as save:
        on = app._set_deep_research_ultra(True)
        assert app.cfg.research.ultra_enabled is True
        off = app._set_deep_research_ultra(False)

    assert app.cfg.research.ultra_enabled is False
    assert save.call_count == 2
    assert "Ultra is on" in on
    assert "Ultra is off" in off


def test_dr_counts_and_notes_count_read_the_stores():
    app = JarvisApp()

    # list[Any]: _dr_counts only reads .status and _notes_count only takes
    # len(), so mocks stand in for the real DeepResearchState / Note the
    # stores return, and a list of those is invariant.
    def _sessions() -> list[Any]:
        return [
            MagicMock(status="paused"),
            MagicMock(status="running"),
            MagicMock(status="paused"),
        ]

    def _notes() -> list[Any]:
        return ["a", "b"]

    app._list_dr = _sessions
    app._list_notes = _notes

    assert app._dr_counts() == (3, 2)
    assert app._notes_count() == 2


def test_take_note_delegates_to_the_notes_panel():
    app = JarvisApp()
    app.notes_panel = MagicMock()
    app.notes_panel.create_and_show.return_value = "Noted, sir."

    assert app._take_note("Shopping", "milk") == "Noted, sir."
    app.notes_panel.create_and_show.assert_called_once_with("Shopping", "milk")
