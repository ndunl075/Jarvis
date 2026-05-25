"""Tests for jarvis.ui.hotkeys.HotkeyManager.

pynput is mocked entirely — no real key registration in CI. The lazy
import inside register_all() makes patching pynput.keyboard.GlobalHotKeys
the natural seam."""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from jarvis.core.config import HotkeysConfig
from jarvis.core.events import EventBus, ModeChanged, WakeWordDetected
from jarvis.core.state_machine import Mode, StateMachine
from jarvis.ui.hotkeys import HotkeyManager, _to_pynput_hotkey


# --- pynput translation -------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ctrl+shift+m", "<ctrl>+<shift>+m"),
        ("CTRL+SHIFT+M", "<ctrl>+<shift>+m"),
        ("ctrl+space", "<ctrl>+<space>"),
        ("ctrl+shift+,", "<ctrl>+<shift>+,"),
        ("f9", "<f9>"),
        ("ctrl+alt+f12", "<ctrl>+<alt>+<f12>"),
        ("win+l", "<cmd>+l"),
        ("super+space", "<cmd>+<space>"),
        ("escape", "<esc>"),
        # Plain letter passes through.
        ("m", "m"),
        # Whitespace tolerated around plus signs.
        ("ctrl + shift + m", "<ctrl>+<shift>+m"),
    ],
)
def test_translation_table(raw, expected):
    assert _to_pynput_hotkey(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "+", "++"])
def test_translation_rejects_empty_or_punctuation_only(raw):
    with pytest.raises(ValueError):
        _to_pynput_hotkey(raw)


# --- fakes / helpers ----------------------------------------------------


class _FakeGlobalHotKeys:
    """Stand-in for pynput.keyboard.GlobalHotKeys. Records the action
    dict, exposes start/stop counters, and lets tests trigger
    individual callbacks to simulate a key event."""

    instances: list[_FakeGlobalHotKeys] = []

    def __init__(self, actions: dict[str, object]) -> None:
        self.actions = actions
        self.start_calls = 0
        self.stop_calls = 0
        _FakeGlobalHotKeys.instances.append(self)

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def fire(self, key: str) -> None:
        """Test helper: invoke the registered action for `key` on the
        calling thread, mimicking pynput's hook-thread invocation."""
        self.actions[key]()


def _install_fake_pynput() -> _FakeGlobalHotKeys:
    """Inject a fake pynput.keyboard module into sys.modules so the
    lazy import inside register_all() finds our stub. Returns the
    fake class so the test can assert on its state."""
    pkg = types.ModuleType("pynput")
    kb = types.ModuleType("pynput.keyboard")
    kb.GlobalHotKeys = _FakeGlobalHotKeys  # type: ignore[attr-defined]
    pkg.keyboard = kb  # type: ignore[attr-defined]
    _FakeGlobalHotKeys.instances.clear()
    return kb


@pytest.fixture
def fake_pynput(monkeypatch):
    kb = _install_fake_pynput()
    monkeypatch.setitem(sys.modules, "pynput", sys.modules.get("pynput") or types.ModuleType("pynput"))
    monkeypatch.setitem(sys.modules, "pynput.keyboard", kb)
    yield _FakeGlobalHotKeys


def _make_mgr(
    *,
    sm: StateMachine | None = None,
    hotkeys: HotkeysConfig | None = None,
    audio_loop=None,
    on_open_settings=None,
):
    sm = sm or StateMachine()
    bus = EventBus()
    loop = audio_loop if audio_loop is not None else MagicMock()
    on_settings = on_open_settings or MagicMock()
    mgr = HotkeyManager(
        sm=sm,
        bus=bus,
        audio_loop=loop,
        hotkeys=hotkeys or HotkeysConfig(),
        on_open_settings=on_settings,
    )
    return mgr, sm, bus, loop, on_settings


# --- registration --------------------------------------------------------


def test_register_all_starts_listener_with_configured_bindings(fake_pynput):
    mgr, *_ = _make_mgr()
    mgr.register_all()
    assert len(fake_pynput.instances) == 1
    inst = fake_pynput.instances[0]
    assert inst.start_calls == 1
    # Default config: mute + open_settings (push_to_talk is None).
    assert set(inst.actions.keys()) == {
        "<ctrl>+<shift>+m",
        "<ctrl>+<shift>+,",
    }


def test_register_all_skips_empty_bindings(fake_pynput):
    """push_to_talk=None is the off-by-default state. It must not show
    up as a registered key."""
    hk = HotkeysConfig(mute="ctrl+shift+m", push_to_talk=None,
                       open_settings="ctrl+shift+,")
    mgr, *_ = _make_mgr(hotkeys=hk)
    mgr.register_all()
    inst = fake_pynput.instances[0]
    assert all("push" not in k for k in inst.actions)


def test_register_all_with_push_to_talk_set_includes_it(fake_pynput):
    hk = HotkeysConfig(mute="ctrl+shift+m",
                       push_to_talk="ctrl+space",
                       open_settings="ctrl+shift+,")
    mgr, *_ = _make_mgr(hotkeys=hk)
    mgr.register_all()
    inst = fake_pynput.instances[0]
    assert "<ctrl>+<space>" in inst.actions


def test_register_all_logs_and_skips_unparseable_binding(
    fake_pynput, caplog
):
    """A bad binding must NOT block the others from registering."""
    import logging
    hk = HotkeysConfig(mute="", push_to_talk=None,
                       open_settings="ctrl+shift+,")
    mgr, *_ = _make_mgr(hotkeys=hk)
    with caplog.at_level(logging.WARNING, logger="jarvis.ui.hotkeys"):
        mgr.register_all()
    inst = fake_pynput.instances[0]
    # Only open_settings was registered; mute was empty and skipped.
    assert list(inst.actions.keys()) == ["<ctrl>+<shift>+,"]


def test_register_all_with_no_valid_bindings_skips_listener(
    fake_pynput, caplog
):
    """If every binding is empty, don't bother starting a listener.
    The composition root still gets a working HotkeyManager — it just
    does nothing."""
    import logging
    hk = HotkeysConfig(mute="", push_to_talk=None, open_settings="")
    mgr, *_ = _make_mgr(hotkeys=hk)
    with caplog.at_level(logging.INFO, logger="jarvis.ui.hotkeys"):
        mgr.register_all()
    assert fake_pynput.instances == []


def test_register_all_idempotent(fake_pynput, caplog):
    import logging
    mgr, *_ = _make_mgr()
    mgr.register_all()
    with caplog.at_level(logging.INFO, logger="jarvis.ui.hotkeys"):
        mgr.register_all()
    assert len(fake_pynput.instances) == 1  # not a second listener


def test_register_all_missing_pynput_logs_and_returns(monkeypatch, caplog):
    """A headless install without pynput must not crash the app — it
    just disables hotkeys."""
    import logging
    # Force the lazy import to fail.
    monkeypatch.setitem(sys.modules, "pynput.keyboard", None)
    mgr, *_ = _make_mgr()
    with caplog.at_level(logging.ERROR, logger="jarvis.ui.hotkeys"):
        mgr.register_all()
    assert any(
        "pynput not installed" in r.message for r in caplog.records
    )


# --- action handlers ----------------------------------------------------


def test_mute_hotkey_toggles_active_to_muted(fake_pynput):
    """Tap when ACTIVE -> MUTED via the audio loop."""
    sm = StateMachine(initial_mode=Mode.ACTIVE)
    # Real loop in a thread so run_coroutine_threadsafe actually runs.
    loop = asyncio.new_event_loop()
    import threading
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        mgr, _, _, _, _ = _make_mgr(sm=sm, audio_loop=loop)
        mgr.register_all()
        fake_pynput.instances[0].fire("<ctrl>+<shift>+m")
        # Give the coroutine a moment to execute.
        for _ in range(50):
            if sm.mode is Mode.MUTED:
                break
            import time
            time.sleep(0.005)
        assert sm.mode is Mode.MUTED
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=1.0)
        loop.close()


def test_mute_hotkey_toggles_muted_to_active(fake_pynput):
    sm = StateMachine(initial_mode=Mode.MUTED)
    loop = asyncio.new_event_loop()
    import threading
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        mgr, *_ = _make_mgr(sm=sm, audio_loop=loop)
        mgr.register_all()
        fake_pynput.instances[0].fire("<ctrl>+<shift>+m")
        for _ in range(50):
            if sm.mode is Mode.ACTIVE:
                break
            import time
            time.sleep(0.005)
        assert sm.mode is Mode.ACTIVE
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=1.0)
        loop.close()


def test_mute_hotkey_when_sleeping_is_noop(fake_pynput, caplog):
    """Mute is meaningless when the mic is already cold; no dispatch
    happens, the audio loop sees nothing."""
    import logging
    sm = StateMachine(initial_mode=Mode.SLEEPING)
    loop = MagicMock()
    mgr, *_ = _make_mgr(sm=sm, audio_loop=loop)
    mgr.register_all()
    with caplog.at_level(logging.INFO, logger="jarvis.ui.hotkeys"):
        fake_pynput.instances[0].fire("<ctrl>+<shift>+m")
    # Nothing scheduled onto the audio loop.
    assert sm.mode is Mode.SLEEPING


def test_mute_hotkey_dispatch_uses_audio_loop(fake_pynput):
    """The set_mode coroutine MUST go through run_coroutine_threadsafe
    on the audio loop, not be invoked directly on the pynput thread."""
    sm = StateMachine(initial_mode=Mode.ACTIVE)
    loop = MagicMock()
    mgr, *_ = _make_mgr(sm=sm, audio_loop=loop)
    mgr.register_all()
    with patch(
        "jarvis.ui.hotkeys.asyncio.run_coroutine_threadsafe"
    ) as rcs:
        fake_pynput.instances[0].fire("<ctrl>+<shift>+m")
    assert rcs.call_count == 1
    _, kwargs_or_pos = rcs.call_args
    # Called as run_coroutine_threadsafe(coro, loop).
    assert rcs.call_args.args[1] is loop


def test_push_to_talk_publishes_synthetic_wake_event(fake_pynput):
    """Phase 5 tap-to-listen: PTT publishes WakeWordDetected so the
    pipeline takes the same path it would for "hey jarvis"."""
    import threading
    import time
    sm = StateMachine(initial_mode=Mode.ACTIVE)
    hk = HotkeysConfig(mute="ctrl+shift+m",
                       push_to_talk="ctrl+space",
                       open_settings="ctrl+shift+,")
    mgr, _, bus, _, _ = _make_mgr(sm=sm, hotkeys=hk)
    seen: list[WakeWordDetected] = []
    bus.subscribe(WakeWordDetected, lambda e: seen.append(e))

    # Real loop in a background thread so bus.publish (which schedules
    # the dispatch coroutine on the bound loop) actually runs.
    loop = asyncio.new_event_loop()
    bus.bind_loop(loop)
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        mgr.register_all()
        fake_pynput.instances[0].fire("<ctrl>+<space>")
        for _ in range(50):
            if seen:
                break
            time.sleep(0.005)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=1.0)
        loop.close()
    assert len(seen) == 1
    assert seen[0].confidence == 1.0


def test_push_to_talk_ignored_when_not_active(fake_pynput):
    """No synthetic wake events while MUTED or SLEEPING."""
    sm = StateMachine(initial_mode=Mode.MUTED)
    hk = HotkeysConfig(mute="ctrl+shift+m",
                       push_to_talk="ctrl+space",
                       open_settings="ctrl+shift+,")
    mgr, _, bus, _, _ = _make_mgr(sm=sm, hotkeys=hk)
    seen: list[WakeWordDetected] = []
    bus.subscribe(WakeWordDetected, lambda e: seen.append(e))
    mgr.register_all()
    fake_pynput.instances[0].fire("<ctrl>+<space>")
    assert seen == []


def test_settings_hotkey_invokes_injected_callback(fake_pynput):
    on_settings = MagicMock()
    mgr, *_ = _make_mgr(on_open_settings=on_settings)
    mgr.register_all()
    fake_pynput.instances[0].fire("<ctrl>+<shift>+,")
    on_settings.assert_called_once_with()


def test_settings_callback_exception_does_not_crash_hook(
    fake_pynput, caplog
):
    """A raising callback must not bring down the pynput hook thread
    (which would lose all subsequent hotkeys)."""
    import logging
    on_settings = MagicMock(side_effect=RuntimeError("settings boom"))
    mgr, *_ = _make_mgr(on_open_settings=on_settings)
    mgr.register_all()
    with caplog.at_level(logging.ERROR, logger="jarvis.ui.hotkeys"):
        fake_pynput.instances[0].fire("<ctrl>+<shift>+,")
    assert any(
        "on_open_settings callback raised" in r.message
        for r in caplog.records
    )


def test_actions_no_op_after_close(fake_pynput):
    """A late-arriving hook call after close() must NOT touch the SM
    or fire the callback. _alive guard on every action."""
    sm = StateMachine(initial_mode=Mode.ACTIVE)
    on_settings = MagicMock()
    mgr, *_ = _make_mgr(sm=sm, on_open_settings=on_settings)
    mgr.register_all()
    inst = fake_pynput.instances[0]
    mgr.close()
    inst.fire("<ctrl>+<shift>+m")
    inst.fire("<ctrl>+<shift>+,")
    on_settings.assert_not_called()
    assert sm.mode is Mode.ACTIVE


# --- shutdown ------------------------------------------------------------


def test_close_stops_the_listener(fake_pynput):
    mgr, *_ = _make_mgr()
    mgr.register_all()
    inst = fake_pynput.instances[0]
    mgr.close()
    assert inst.stop_calls == 1


def test_close_is_idempotent(fake_pynput):
    mgr, *_ = _make_mgr()
    mgr.register_all()
    inst = fake_pynput.instances[0]
    mgr.close()
    mgr.close()  # must not raise; must not double-stop
    assert inst.stop_calls == 1


def test_close_without_register_is_safe():
    """A construction that never called register_all (e.g. settings
    UI deferred startup) must still close cleanly."""
    mgr, *_ = _make_mgr()
    mgr.close()  # must not raise
