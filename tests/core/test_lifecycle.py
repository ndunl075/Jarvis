"""Tests for jarvis.core.lifecycle.

Coverage:
- Load order: declared order on load_all, reverse on unload_all.
- Idempotency: load() / unload() on already-loaded / already-unloaded skips.
- load_all fail-fast with rollback: failure of N triggers unload of 1..N-1
  in reverse, original exception propagates, later loadables not touched.
- unload_all best-effort: failure of one does not stop the rest.
- transition_to_mode: every (old, new) Mode pair drives the right action
  (load / unload / no-op), with the documented asymmetric failure policy.
- register() appends and the new entry participates in lifecycle.
- bind() subscribes to ModeChanged and unbind() removes it.
- The full Phase 1 smoke test: config + events + state machine + lifecycle
  composed together.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.core.config import JarvisConfig, load_config, save_config
from jarvis.core.events import ConfigChanged, EventBus, ModeChanged
from jarvis.core.lifecycle import LifecycleManager, Loadable
from jarvis.core.state_machine import (
    ConversationalState,
    Mode,
    StateMachine,
)

# --- fakes ----------------------------------------------------------------


class FakeLoadable:
    """A minimal Loadable that records every load/unload call into a shared
    log so tests can assert ordering. Optionally fails on load or unload."""

    def __init__(
        self,
        name: str,
        log: list[tuple[str, str]],
        *,
        fail_on_load: bool = False,
        fail_on_unload: bool = False,
    ) -> None:
        self.name = name
        self.is_loaded = False
        self._log = log
        self._fail_on_load = fail_on_load
        self._fail_on_unload = fail_on_unload

    async def load(self) -> None:
        self._log.append(("load", self.name))
        if self._fail_on_load:
            raise RuntimeError(f"{self.name}: load failed")
        self.is_loaded = True

    async def unload(self) -> None:
        self._log.append(("unload", self.name))
        if self._fail_on_unload:
            raise RuntimeError(f"{self.name}: unload failed")
        self.is_loaded = False


def test_fakeloadable_satisfies_protocol():
    # runtime_checkable Protocol verifies the structural shape.
    f = FakeLoadable("x", [])
    assert isinstance(f, Loadable)


# --- ordering -------------------------------------------------------------


async def test_load_all_loads_in_declared_order():
    log: list[tuple[str, str]] = []
    a, b, c = (FakeLoadable(n, log) for n in "abc")
    lm = LifecycleManager([a, b, c])
    await lm.load_all()
    assert log == [("load", "a"), ("load", "b"), ("load", "c")]
    assert all(x.is_loaded for x in (a, b, c))


async def test_unload_all_unloads_in_reverse_order():
    log: list[tuple[str, str]] = []
    a, b, c = (FakeLoadable(n, log) for n in "abc")
    lm = LifecycleManager([a, b, c])
    await lm.load_all()
    log.clear()
    await lm.unload_all()
    assert log == [("unload", "c"), ("unload", "b"), ("unload", "a")]
    assert not any(x.is_loaded for x in (a, b, c))


# --- idempotency ----------------------------------------------------------


async def test_load_all_skips_already_loaded():
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    lm = LifecycleManager([a])
    await lm.load_all()
    await lm.load_all()  # second call must not re-call load()
    assert log == [("load", "a")]


async def test_unload_all_skips_already_unloaded():
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    lm = LifecycleManager([a])
    await lm.unload_all()  # never loaded; nothing should run
    assert log == []


# --- load_all fail-fast with rollback ------------------------------------


async def test_load_all_rolls_back_on_failure():
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    b = FakeLoadable("b", log)
    boom = FakeLoadable("boom", log, fail_on_load=True)
    d = FakeLoadable("d", log)
    lm = LifecycleManager([a, b, boom, d])

    with pytest.raises(RuntimeError, match="boom: load failed"):
        await lm.load_all()

    # a and b loaded, boom attempted, d never reached.
    # Rollback unloaded b then a in reverse.
    assert log == [
        ("load", "a"),
        ("load", "b"),
        ("load", "boom"),
        ("unload", "b"),
        ("unload", "a"),
    ]
    assert not a.is_loaded
    assert not b.is_loaded
    assert not boom.is_loaded
    assert not d.is_loaded


async def test_load_all_rollback_failures_logged_not_propagated(
    caplog: pytest.LogCaptureFixture,
):
    import logging

    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log, fail_on_unload=True)  # rollback unload will fail
    boom = FakeLoadable("boom", log, fail_on_load=True)
    lm = LifecycleManager([a, boom])

    with caplog.at_level(logging.ERROR, logger="jarvis.core.lifecycle"):
        with pytest.raises(RuntimeError, match="boom: load failed"):
            await lm.load_all()

    assert any("unload failed for 'a'" in r.message for r in caplog.records)


# --- unload_all best-effort -----------------------------------------------


async def test_unload_all_continues_past_failure():
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    b = FakeLoadable("b", log, fail_on_unload=True)
    c = FakeLoadable("c", log)
    lm = LifecycleManager([a, b, c])
    await lm.load_all()
    log.clear()

    await lm.unload_all()  # must not raise

    # Reverse order: c, b (fails), a (still attempted).
    assert log == [("unload", "c"), ("unload", "b"), ("unload", "a")]
    assert not a.is_loaded
    assert b.is_loaded  # the failed unload left it marked loaded; its choice
    assert not c.is_loaded


# --- register -------------------------------------------------------------


async def test_register_appends_and_participates():
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    lm = LifecycleManager([a])
    b = FakeLoadable("b", log)
    lm.register(b)
    await lm.load_all()
    assert log == [("load", "a"), ("load", "b")]
    log.clear()
    await lm.unload_all()
    assert log == [("unload", "b"), ("unload", "a")]


# --- transition_to_mode dispatch -----------------------------------------


@pytest.mark.parametrize(
    "old,new,expect",
    [
        # Loading transitions
        (Mode.SLEEPING, Mode.ACTIVE, "load"),
        (Mode.SLEEPING, Mode.MUTED, "load"),
        # Unloading transitions
        (Mode.ACTIVE, Mode.SLEEPING, "unload"),
        (Mode.MUTED, Mode.SLEEPING, "unload"),
        # No-op transitions on Loadables
        (Mode.ACTIVE, Mode.MUTED, "noop"),
        (Mode.MUTED, Mode.ACTIVE, "noop"),
        # Same-mode no-ops
        (Mode.ACTIVE, Mode.ACTIVE, "noop"),
        (Mode.MUTED, Mode.MUTED, "noop"),
        (Mode.SLEEPING, Mode.SLEEPING, "noop"),
    ],
)
async def test_transition_to_mode_dispatches_correctly(
    old: Mode, new: Mode, expect: str
):
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    b = FakeLoadable("b", log)
    lm = LifecycleManager([a, b])

    # If the transition will unload, we need them loaded first.
    if expect == "unload":
        await lm.load_all()
        log.clear()

    await lm.transition_to_mode(old, new)

    if expect == "load":
        assert log == [("load", "a"), ("load", "b")]
    elif expect == "unload":
        assert log == [("unload", "b"), ("unload", "a")]
    else:
        assert log == []


async def test_transition_to_loading_uses_failfast():
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    boom = FakeLoadable("boom", log, fail_on_load=True)
    lm = LifecycleManager([a, boom])

    with pytest.raises(RuntimeError, match="boom"):
        await lm.transition_to_mode(Mode.SLEEPING, Mode.ACTIVE)

    # Rollback ran: a got unloaded.
    assert not a.is_loaded


async def test_transition_to_unloading_uses_best_effort():
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    b = FakeLoadable("b", log, fail_on_unload=True)
    c = FakeLoadable("c", log)
    lm = LifecycleManager([a, b, c])
    await lm.load_all()
    log.clear()

    await lm.transition_to_mode(Mode.ACTIVE, Mode.SLEEPING)  # must not raise
    assert ("unload", "a") in log  # reached past the failing b


# --- bus binding ----------------------------------------------------------


async def test_bind_subscribes_to_modechanged_and_drives_transitions():
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    bus = EventBus()
    lm = LifecycleManager([a], bus=bus)
    lm.bind(bus)

    # Simulate the SM emitting a SLEEPING->ACTIVE transition.
    await bus.publish_and_wait(ModeChanged(old=Mode.SLEEPING, new=Mode.ACTIVE))
    assert log == [("load", "a")]
    assert a.is_loaded

    # And ACTIVE->SLEEPING.
    await bus.publish_and_wait(ModeChanged(old=Mode.ACTIVE, new=Mode.SLEEPING))
    assert log[-1] == ("unload", "a")
    assert not a.is_loaded


async def test_unbind_removes_subscription():
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    bus = EventBus()
    lm = LifecycleManager([a], bus=bus)
    lm.bind(bus)
    lm.unbind()

    await bus.publish_and_wait(ModeChanged(old=Mode.SLEEPING, new=Mode.ACTIVE))
    assert log == []


async def test_bind_is_idempotent_replaces_prior_subscription():
    log: list[tuple[str, str]] = []
    a = FakeLoadable("a", log)
    bus = EventBus()
    lm = LifecycleManager([a], bus=bus)
    lm.bind(bus)
    lm.bind(bus)  # second bind must not double-fire on each ModeChanged

    await bus.publish_and_wait(ModeChanged(old=Mode.SLEEPING, new=Mode.ACTIVE))
    # Only one load call, not two.
    assert log == [("load", "a")]


# --- Phase 1 smoke test --------------------------------------------------
# BUILD.md § Phase 1: "a smoke test shows config changes emitting events that
# the state machine and a fake module can react to."


async def test_phase1_smoke_config_events_state_machine_lifecycle(tmp_path):
    # 1. Round-trip a config change through disk.
    cfg_path = tmp_path / "config.json"
    cfg = load_config(cfg_path)
    cfg.llm.temperature = 0.42
    save_config(cfg, cfg_path)
    reloaded = load_config(cfg_path)
    assert reloaded.llm.temperature == 0.42

    # 2. Compose the event bus + state machine + lifecycle manager.
    bus = EventBus()
    log: list[tuple[str, str]] = []
    fake_module = FakeLoadable("fake_audio", log)
    lm = LifecycleManager([fake_module], bus=bus)
    lm.bind(bus)
    sm = StateMachine(bus=bus)

    # 3. Subscribe a fake config-change observer (simulating the future
    #    ConfigManager wrapper); also subscribe a state-observer.
    config_events_seen: list[ConfigChanged] = []
    bus.subscribe(ConfigChanged, lambda e: config_events_seen.append(e))

    state_history: list[tuple[Mode, Mode]] = []
    bus.subscribe(ModeChanged, lambda e: state_history.append((e.old, e.new)))

    # 4. Emulate a future ConfigManager: emit ConfigChanged for the
    #    temperature change above. This proves the wiring path the
    #    Contract-2 design note documented.
    old_cfg = JarvisConfig()
    new_cfg = JarvisConfig()
    new_cfg.llm.temperature = 0.42
    await bus.publish_and_wait(
        ConfigChanged(old=old_cfg, new=new_cfg, changed_fields=("llm.temperature",))
    )
    assert len(config_events_seen) == 1
    assert "llm.temperature" in config_events_seen[0].changed_fields

    # 5. Drive the state machine through the full lifecycle.
    sm.set_mode(Mode.SLEEPING)
    sm.set_mode(Mode.ACTIVE)  # SLEEPING -> ACTIVE: lifecycle should load
    sm.set_conversational_state(ConversationalState.LISTENING)
    sm.set_mode(Mode.SLEEPING)  # ACTIVE -> SLEEPING: lifecycle should unload
                                #                     and force CS to IDLE first

    # Drain any tasks the bus scheduled.
    for _ in range(10):
        await asyncio.sleep(0)

    # Three Mode transitions (the sm forces CS->IDLE before each, but those
    # only emit when CS isn't already IDLE).
    assert state_history == [
        (Mode.ACTIVE, Mode.SLEEPING),
        (Mode.SLEEPING, Mode.ACTIVE),
        (Mode.ACTIVE, Mode.SLEEPING),
    ]

    # The fake module participated in lifecycle in the right places:
    # - ACTIVE -> SLEEPING (initial): no-op, nothing was loaded yet
    # - SLEEPING -> ACTIVE: load
    # - ACTIVE -> SLEEPING: unload
    assert log == [("load", "fake_audio"), ("unload", "fake_audio")]
    assert not fake_module.is_loaded

    # Final settled state.
    assert sm.mode is Mode.SLEEPING
    assert sm.conversational_state is ConversationalState.IDLE
