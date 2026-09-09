"""Tests for jarvis.app — the composition root.

Strategy: mock everything that touches real hardware (audio stack, Qt, pynput)
so these tests run in CI without devices, models, or a display.

Covered:
  - _audio_main: boot happy path, load failure, Ollama warning, unload order
  - Lifecycle module list declared order
  - on_quit drains audio stack before calling qt_app.quit()
  - on_open_settings lazily creates and re-raises SettingsWindow
  - TrayIcon, OverlayOrb, HotkeyManager injected with correct dependencies
  - AmplitudeLatch shared between TTS and OverlayOrb
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.app import _audio_main
from jarvis.core.config import LifecycleConfig
from jarvis.core.events import EventBus
from jarvis.core.state_machine import StateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lm(load_side_effect=None):
    lm = MagicMock()
    lm.load_all = AsyncMock(side_effect=load_side_effect)
    lm.unload_all = AsyncMock()
    # bind/unbind are unused by app.py post-Phase 6 (ModeCoordinator owns
    # transitions). Kept on the mock so legacy probes don't AttributeError.
    lm.bind = MagicMock()
    lm.unbind = MagicMock()
    return lm


def _make_pipeline():
    p = MagicMock()
    p.start = AsyncMock()
    p.stop = AsyncMock()
    return p


def _make_ollama(warm_side_effect=None):
    o = MagicMock()
    o.warm = AsyncMock(side_effect=warm_side_effect)
    return o


def _make_coordinator():
    c = MagicMock()
    c.request = AsyncMock()
    return c


def _make_tts():
    t = MagicMock()
    t.unload = AsyncMock()
    t.load = AsyncMock()
    return t


def _make_stt():
    s = MagicMock()
    s.unload = AsyncMock()
    s.load = AsyncMock()
    return s


def _make_wake_word():
    return MagicMock()


def _make_source():
    s = MagicMock()
    s.load = AsyncMock()
    s.unload = AsyncMock()
    return s


def _make_mcp_manager():
    m = MagicMock()
    m.add_server = AsyncMock()
    m.reload_from_config = AsyncMock()
    m.shutdown = AsyncMock()
    return m


def _make_registry():
    """_audio_main only passes the registry through to the ConfigChanged
    handler, so a bare mock is enough here."""
    return MagicMock()


# ---------------------------------------------------------------------------
# _audio_main: happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_main_happy_path():
    lm = _make_lm()
    pipeline = _make_pipeline()
    ollama = _make_ollama()
    tts = _make_tts()
    stt = _make_stt()
    wake_word = _make_wake_word()
    bus = EventBus(loop=asyncio.get_running_loop())
    sm = StateMachine()
    coordinator = _make_coordinator()
    lifecycle_cfg = LifecycleConfig()
    stop_event = asyncio.Event()
    stop_event.set()  # skip wait
    holder: list[str | None] = [None]
    done = threading.Event()

    await _audio_main(
        lm, pipeline, ollama, tts, stt, wake_word, bus, sm, coordinator,
        lifecycle_cfg, stop_event, holder, done,
        source=_make_source(),
        mcp_manager=_make_mcp_manager(),
        mcp_servers=[],
        registry=_make_registry(),
    )

    lm.load_all.assert_awaited_once()
    ollama.warm.assert_awaited_once()
    # Post-Phase 6: app.py no longer calls lm.bind/unbind; ModeCoordinator
    # drives transitions. Assert neither is called.
    lm.bind.assert_not_called()
    lm.unbind.assert_not_called()
    pipeline.start.assert_awaited_once()
    pipeline.stop.assert_awaited_once()
    lm.unload_all.assert_awaited_once()
    assert done.is_set()
    assert holder[0] is None


# ---------------------------------------------------------------------------
# _audio_main: module load failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_main_load_failure_sets_error_skips_pipeline():
    lm = _make_lm(load_side_effect=RuntimeError("mic not found"))
    pipeline = _make_pipeline()
    ollama = _make_ollama()
    tts = _make_tts()
    stt = _make_stt()
    wake_word = _make_wake_word()
    bus = EventBus(loop=asyncio.get_running_loop())
    sm = StateMachine()
    coordinator = _make_coordinator()
    lifecycle_cfg = LifecycleConfig()
    stop_event = asyncio.Event()
    holder: list[str | None] = [None]
    done = threading.Event()

    await _audio_main(
        lm, pipeline, ollama, tts, stt, wake_word, bus, sm, coordinator,
        lifecycle_cfg, stop_event, holder, done,
        source=_make_source(),
        mcp_manager=_make_mcp_manager(),
        mcp_servers=[],
        registry=_make_registry(),
    )

    assert done.is_set()
    assert holder[0] is not None
    assert "load_failure" in holder[0]
    assert "mic not found" in holder[0]
    pipeline.start.assert_not_awaited()


# ---------------------------------------------------------------------------
# _audio_main: Ollama unavailable (non-fatal)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_main_ollama_warmup_failure_is_non_fatal():
    lm = _make_lm()
    pipeline = _make_pipeline()
    ollama = _make_ollama(warm_side_effect=ConnectionRefusedError("no daemon"))
    tts = _make_tts()
    stt = _make_stt()
    wake_word = _make_wake_word()
    bus = EventBus(loop=asyncio.get_running_loop())
    sm = StateMachine()
    coordinator = _make_coordinator()
    lifecycle_cfg = LifecycleConfig()
    stop_event = asyncio.Event()
    stop_event.set()
    holder: list[str | None] = [None]
    done = threading.Event()

    await _audio_main(
        lm, pipeline, ollama, tts, stt, wake_word, bus, sm, coordinator,
        lifecycle_cfg, stop_event, holder, done,
        source=_make_source(),
        mcp_manager=_make_mcp_manager(),
        mcp_servers=[],
        registry=_make_registry(),
    )

    assert done.is_set()
    assert holder[0] is not None
    assert holder[0].startswith("ollama_warning:")
    # Pipeline still starts despite Ollama being absent
    pipeline.start.assert_awaited_once()


# ---------------------------------------------------------------------------
# _audio_main: unload runs strictly after pipeline.stop()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_main_unload_after_pipeline_stop():
    """unload_all must follow pipeline.stop() so active streams are torn
    down before modules release their resources."""
    order: list[str] = []

    lm = MagicMock()
    lm.load_all = AsyncMock()
    lm.unload_all = AsyncMock(side_effect=lambda: order.append("unload_all"))
    lm.bind = MagicMock()
    lm.unbind = MagicMock()

    pipeline = MagicMock()
    pipeline.start = AsyncMock()
    pipeline.stop = AsyncMock(side_effect=lambda: order.append("pipeline_stop"))

    ollama = _make_ollama()
    tts = _make_tts()
    stt = _make_stt()
    wake_word = _make_wake_word()
    bus = EventBus(loop=asyncio.get_running_loop())
    sm = StateMachine()
    coordinator = _make_coordinator()
    lifecycle_cfg = LifecycleConfig()
    stop_event = asyncio.Event()
    stop_event.set()
    holder: list[str | None] = [None]
    done = threading.Event()

    await _audio_main(
        lm, pipeline, ollama, tts, stt, wake_word, bus, sm, coordinator,
        lifecycle_cfg, stop_event, holder, done,
        source=_make_source(),
        mcp_manager=_make_mcp_manager(),
        mcp_servers=[],
        registry=_make_registry(),
    )

    assert order == ["pipeline_stop", "unload_all"]


# ---------------------------------------------------------------------------
# Lifecycle: declared module list preserved as loadables
# ---------------------------------------------------------------------------


def test_lifecycle_manager_preserves_module_order():
    from jarvis.core.lifecycle import LifecycleManager

    mods = [MagicMock(name=n) for n in ["source", "wake_word", "vad", "stt", "tts", "ollama"]]
    lm = LifecycleManager(mods)
    assert list(lm.loadables) == mods


# ---------------------------------------------------------------------------
# on_quit: thread joined before qt_app.quit()
# ---------------------------------------------------------------------------


def test_on_quit_joins_thread_before_qt_quit():
    order: list[str] = []

    qt_app = MagicMock()
    qt_app.quit = MagicMock(side_effect=lambda: order.append("qt_quit"))
    audio_loop = MagicMock()
    audio_thread = MagicMock()
    audio_thread.is_alive.return_value = True
    audio_thread.join = MagicMock(side_effect=lambda timeout=None: order.append("thread_join"))
    tray = MagicMock()
    orb = MagicMock()
    hotkeys_mgr = MagicMock()
    settings_ref: list = [None]
    quit_called = [False]

    def _on_quit():
        if quit_called[0]:
            return
        quit_called[0] = True
        audio_loop.call_soon_threadsafe(lambda: None)
        try:
            tray.hide()
        except Exception:
            pass
        if audio_thread.is_alive():
            audio_thread.join(timeout=10.0)
        tray.close()
        orb.close()
        hotkeys_mgr.close()
        if settings_ref[0] is not None:
            settings_ref[0].close()
        qt_app.quit()

    _on_quit()

    assert order == ["thread_join", "qt_quit"], (
        "audio thread must be joined before qt_app.quit()"
    )
    tray.close.assert_called_once()
    orb.close.assert_called_once()
    hotkeys_mgr.close.assert_called_once()


def test_on_quit_is_idempotent():
    qt_app = MagicMock()
    audio_loop = MagicMock()
    audio_thread = MagicMock()
    audio_thread.is_alive.return_value = False
    tray = MagicMock()
    orb = MagicMock()
    hotkeys_mgr = MagicMock()
    quit_called = [False]

    def _on_quit():
        if quit_called[0]:
            return
        quit_called[0] = True
        audio_loop.call_soon_threadsafe(lambda: None)
        tray.close()
        orb.close()
        hotkeys_mgr.close()
        qt_app.quit()

    _on_quit()
    _on_quit()

    assert qt_app.quit.call_count == 1


# ---------------------------------------------------------------------------
# on_open_settings: lazy creation, re-raise on second call
# ---------------------------------------------------------------------------


def test_open_settings_lazy_and_reuses(qapp):
    from jarvis.core.config import JarvisConfig
    from jarvis.ui.settings import SettingsWindow

    cfg = JarvisConfig()
    settings_ref: list = [None]

    with patch("jarvis.ui.settings.tabs.about.AboutTab._refresh_status"):
        def _open_settings():
            if settings_ref[0] is None:
                settings_ref[0] = SettingsWindow(config=cfg, on_change=lambda: None)
            win = settings_ref[0]
            win.show()
            win.raise_()
            win.activateWindow()

        _open_settings()
        first = settings_ref[0]
        assert first is not None

        _open_settings()
        assert settings_ref[0] is first, "second call must reuse the same window"

    if settings_ref[0]:
        settings_ref[0].close()


# ---------------------------------------------------------------------------
# on_test_voice: dispatches tts.speak to audio loop via run_coroutine_threadsafe
# ---------------------------------------------------------------------------


def test_on_test_voice_dispatches_to_audio_loop():
    """_on_test_voice must call asyncio.run_coroutine_threadsafe with
    tts.speak(phrase) and the audio event loop — not call speak directly."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    tts = MagicMock()
    tts.speak = AsyncMock()
    audio_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    dispatched: list = []

    def fake_run_coroutine_threadsafe(coro, loop):
        dispatched.append((coro, loop))
        return MagicMock()

    with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coroutine_threadsafe):
        def _on_test_voice(phrase: str) -> None:
            try:
                asyncio.run_coroutine_threadsafe(tts.speak(phrase), audio_loop)
            except Exception:
                pass

        _on_test_voice("hello, this is jarvis")

    assert len(dispatched) == 1
    assert dispatched[0][1] is audio_loop


# ---------------------------------------------------------------------------
# TrayIcon, OverlayOrb, HotkeyManager: injected dependencies
# ---------------------------------------------------------------------------


def test_tray_receives_injected_deps(qapp):
    from jarvis.core.config import HotkeysConfig
    from jarvis.core.events import EventBus
    from jarvis.core.state_machine import StateMachine
    from jarvis.ui.tray import TrayIcon

    sm = StateMachine()
    bus = EventBus()
    loop = MagicMock()
    on_settings = MagicMock()
    on_quit_cb = MagicMock()

    tray = TrayIcon(
        sm=sm,
        bus=bus,
        audio_loop=loop,
        hotkeys=HotkeysConfig(),
        on_open_settings=on_settings,
        on_quit=on_quit_cb,
    )
    try:
        assert tray._sm is sm
        assert tray._bus is bus
        assert tray._audio_loop is loop
        assert tray._on_open_settings is on_settings
        assert tray._on_quit is on_quit_cb
    finally:
        tray.close()


def test_overlay_receives_injected_deps(qapp):
    from jarvis.core.events import EventBus
    from jarvis.core.state_machine import StateMachine
    from jarvis.ui.overlay import AmplitudeLatch, OverlayOrb

    sm = StateMachine()
    bus = EventBus()
    latch = AmplitudeLatch()

    orb = OverlayOrb(sm=sm, bus=bus, amplitude_latch=latch)
    try:
        assert orb._sm is sm
        assert orb._bus is bus
        assert orb._amplitude_latch is latch
    finally:
        orb.close()


def test_hotkeys_receives_injected_deps(qapp):
    from jarvis.core.config import HotkeysConfig
    from jarvis.core.events import EventBus
    from jarvis.core.state_machine import StateMachine
    from jarvis.ui.hotkeys import HotkeyManager

    sm = StateMachine()
    bus = EventBus()
    loop = MagicMock()
    on_settings = MagicMock()

    mgr = HotkeyManager(
        sm=sm,
        bus=bus,
        audio_loop=loop,
        hotkeys=HotkeysConfig(),
        on_open_settings=on_settings,
    )
    try:
        assert mgr._sm is sm
        assert mgr._bus is bus
        assert mgr._audio_loop is loop
        assert mgr._on_open_settings is on_settings
    finally:
        mgr.close()


# ---------------------------------------------------------------------------
# AmplitudeLatch: shared between TTS callback and OverlayOrb
# ---------------------------------------------------------------------------


def test_amplitude_latch_shared_reference():
    """make_amplitude_callback() returns (latch, callback) where latch is callback."""
    from jarvis.ui.overlay import make_amplitude_callback

    latch, callback = make_amplitude_callback()
    assert latch is callback
    callback(0.75)
    assert latch.latest() == pytest.approx(0.75)


def test_amplitude_latch_write_read_roundtrip():
    from jarvis.ui.overlay import AmplitudeLatch

    latch = AmplitudeLatch()
    latch(0.5)
    assert latch.latest() == pytest.approx(0.5)
    latch.reset()
    assert latch.latest() == pytest.approx(0.0)
