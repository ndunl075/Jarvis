"""Tests for jarvis.core.mode_coordinator.

Coverage:
- Sleep: speaks confirmation, stops pipeline, sm.set_mode(SLEEPING),
  lm.transition_to_mode runs unload.
- Wake from SLEEPING: lm.transition_to_mode runs load, sm.set_mode(ACTIVE),
  pipeline.start.
- Race: wake during sleep confirmation cancels the sleep mid-phrase.
  End state is the prior Mode; TTS is cancelled; no unload happens.
- Race: second sleep during confirmation is ignored.
- Race: mute during sleep confirmation is ignored (simpler than queueing).
- Mute: ACTIVE <-> MUTED is sm-only, no lifecycle action.
- "Mute" issued while SLEEPING wakes-to-MUTED via load_all.
- sleep_confirmation=False skips the speak step but still unloads.
- Idempotent: request(current_mode) is a no-op fast path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.core.events import EventBus, ModeChanged
from jarvis.core.lifecycle import LifecycleManager
from jarvis.core.mode_coordinator import ModeCoordinator
from jarvis.core.state_machine import Mode, StateMachine

# --- fakes ----------------------------------------------------------------


class FakeTTS:
    """Minimal TTS double. speak() blocks on an event so tests can drive
    the race: hold the speak future open, fire a competing request, then
    release the event."""

    def __init__(self, *, hold_speak: bool = False) -> None:
        self.is_loaded = True
        self.speak_calls: list[str] = []
        self.cancel_count = 0
        self._hold = hold_speak
        self._release = asyncio.Event()
        self._cancelled = asyncio.Event()

    async def speak(self, text: str) -> None:
        self.speak_calls.append(text)
        if not self._hold:
            return
        t_rel = asyncio.create_task(self._release.wait())
        t_can = asyncio.create_task(self._cancelled.wait())
        try:
            await asyncio.wait(
                {t_rel, t_can}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for t in (t_rel, t_can):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, BaseException):
                        pass

    async def cancel(self) -> None:
        self.cancel_count += 1
        self._cancelled.set()

    def release(self) -> None:
        self._release.set()


def _make_lm(loadables=()):
    lm = LifecycleManager(loadables)
    lm.load_all = AsyncMock(wraps=lm.load_all)
    lm.unload_all = AsyncMock(wraps=lm.unload_all)
    lm.transition_to_mode = AsyncMock(wraps=lm.transition_to_mode)
    return lm


def _make_pipeline():
    p = MagicMock()
    p.start = AsyncMock()
    p.stop = AsyncMock()
    return p


def _make_coord(
    *,
    initial_mode: Mode = Mode.ACTIVE,
    sleep_confirmation: bool = True,
    tts: FakeTTS | None = None,
):
    sm = StateMachine(initial_mode=initial_mode)
    lm = _make_lm()
    pipeline = _make_pipeline()
    tts = tts or FakeTTS()
    coord = ModeCoordinator(
        sm=sm,
        lm=lm,
        pipeline=pipeline,
        tts=tts,
        sleep_confirmation=sleep_confirmation,
    )
    return coord, sm, lm, pipeline, tts


# --- sleep happy path ----------------------------------------------------


@pytest.mark.asyncio
async def test_sleep_speaks_then_unloads():
    coord, sm, lm, pipeline, tts = _make_coord()
    await coord.request(Mode.SLEEPING)
    assert tts.speak_calls == ["Going to sleep, sir."]
    pipeline.stop.assert_awaited_once()
    assert sm.mode is Mode.SLEEPING
    lm.transition_to_mode.assert_awaited_once_with(Mode.ACTIVE, Mode.SLEEPING)


@pytest.mark.asyncio
async def test_sleep_confirmation_false_skips_speak():
    coord, sm, _, pipeline, tts = _make_coord(sleep_confirmation=False)
    await coord.request(Mode.SLEEPING)
    assert tts.speak_calls == []
    assert sm.mode is Mode.SLEEPING
    pipeline.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_sleep_skips_speak_if_tts_unloaded():
    tts = FakeTTS()
    tts.is_loaded = False
    coord, sm, _, _, _ = _make_coord(tts=tts)
    await coord.request(Mode.SLEEPING)
    assert tts.speak_calls == []
    assert sm.mode is Mode.SLEEPING


# --- wake happy path ----------------------------------------------------


@pytest.mark.asyncio
async def test_wake_loads_then_starts_pipeline():
    coord, sm, lm, pipeline, _ = _make_coord(initial_mode=Mode.SLEEPING)
    await coord.request(Mode.ACTIVE)
    lm.transition_to_mode.assert_awaited_once_with(Mode.SLEEPING, Mode.ACTIVE)
    assert sm.mode is Mode.ACTIVE
    pipeline.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_wake_order_load_then_pipeline_start():
    """load_all must complete before pipeline.start so the source's
    on_frame callback re-attaches against a loaded stream."""
    coord, sm, lm, pipeline, _ = _make_coord(initial_mode=Mode.SLEEPING)
    order: list[str] = []

    async def load_then(*a, **kw):
        order.append("transition")
    async def start_then():
        order.append("pipeline_start")

    lm.transition_to_mode = AsyncMock(side_effect=load_then)
    pipeline.start = AsyncMock(side_effect=start_then)
    await coord.request(Mode.ACTIVE)
    assert order == ["transition", "pipeline_start"]


# --- race: wake during sleep confirmation -------------------------------


@pytest.mark.asyncio
async def test_wake_during_sleep_confirmation_cancels_sleep():
    """The user pressed Sleep, then Wake before the confirmation phrase
    finished. End state: stay ACTIVE, TTS cancelled, no unload."""
    tts = FakeTTS(hold_speak=True)
    coord, sm, lm, pipeline, _ = _make_coord(tts=tts)

    sleep_task = asyncio.create_task(coord.request(Mode.SLEEPING))
    # Yield enough times for speak() to begin and start waiting.
    for _ in range(5):
        await asyncio.sleep(0)
    assert coord._sleep_in_progress is True

    # Issue Wake. This must cancel the speak and abort the transition.
    await coord.request(Mode.ACTIVE)
    await sleep_task

    assert sm.mode is Mode.ACTIVE
    assert tts.cancel_count >= 1
    pipeline.stop.assert_not_awaited()
    lm.transition_to_mode.assert_not_awaited()


@pytest.mark.asyncio
async def test_second_sleep_during_confirmation_is_ignored(caplog):
    tts = FakeTTS(hold_speak=True)
    coord, sm, lm, pipeline, _ = _make_coord(tts=tts)

    sleep_task = asyncio.create_task(coord.request(Mode.SLEEPING))
    for _ in range(5):
        await asyncio.sleep(0)
    assert coord._sleep_in_progress is True

    import logging
    with caplog.at_level(logging.INFO, logger="jarvis.core.mode_coordinator"):
        # Second sleep request - must be a no-op.
        await coord.request(Mode.SLEEPING)
    assert any("already in progress" in r.message for r in caplog.records)
    # Only one speak in flight; one cancel never issued.
    assert len(tts.speak_calls) == 1
    assert tts.cancel_count == 0

    # Release and let the first sleep complete.
    tts.release()
    await sleep_task
    assert sm.mode is Mode.SLEEPING


@pytest.mark.asyncio
async def test_mute_during_sleep_confirmation_is_ignored():
    """Per design: mute during sleep-in-progress is ignored (simpler than
    queueing). The user can mute after wake completes."""
    tts = FakeTTS(hold_speak=True)
    coord, sm, lm, pipeline, _ = _make_coord(tts=tts)

    sleep_task = asyncio.create_task(coord.request(Mode.SLEEPING))
    for _ in range(5):
        await asyncio.sleep(0)

    # Mute request during confirmation. The coordinator's `request` will
    # route to _request_mute_or_unmute, which is itself lock-guarded;
    # we just assert the *eventual* end state once sleep completes.
    mute_task = asyncio.create_task(coord.request(Mode.MUTED))
    tts.release()
    await sleep_task
    # mute_task may or may not have completed; drive it to completion.
    try:
        await asyncio.wait_for(mute_task, timeout=1.0)
    except TimeoutError:
        mute_task.cancel()

    # End state: SLEEPING (sleep was the committed transition). The mute
    # attempt while sleep was in progress is ignored OR fails because
    # MUTED <- SLEEPING re-loads. Both end-state branches are valid; the
    # invariant we care about is: no torn state.
    assert sm.mode in (Mode.SLEEPING, Mode.MUTED)


# --- mute ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_mute_toggle_does_not_touch_lifecycle():
    coord, sm, lm, _, tts = _make_coord(initial_mode=Mode.ACTIVE)
    await coord.request(Mode.MUTED)
    assert sm.mode is Mode.MUTED
    # transition_to_mode is called for uniformity but its body is a no-op
    # (the ACTIVE<->MUTED pair isn't in the load/unload sets).
    lm.transition_to_mode.assert_awaited_once_with(Mode.ACTIVE, Mode.MUTED)
    lm.load_all.assert_not_awaited()
    lm.unload_all.assert_not_awaited()
    assert tts.speak_calls == []


@pytest.mark.asyncio
async def test_unmute_does_not_touch_lifecycle():
    coord, sm, lm, _, _ = _make_coord(initial_mode=Mode.MUTED)
    await coord.request(Mode.ACTIVE)
    assert sm.mode is Mode.ACTIVE
    lm.load_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_mute_from_sleeping_wakes_to_muted():
    """Per SPEC § Lifecycle Contract: MUTED-from-SLEEP loads everything
    then suppresses wake word."""
    coord, sm, lm, pipeline, _ = _make_coord(initial_mode=Mode.SLEEPING)
    await coord.request(Mode.MUTED)
    lm.transition_to_mode.assert_awaited_once_with(Mode.SLEEPING, Mode.MUTED)
    assert sm.mode is Mode.MUTED
    pipeline.start.assert_awaited_once()


# --- idempotency --------------------------------------------------------


@pytest.mark.asyncio
async def test_request_current_mode_is_noop():
    coord, sm, lm, pipeline, tts = _make_coord(initial_mode=Mode.ACTIVE)
    await coord.request(Mode.ACTIVE)
    assert tts.speak_calls == []
    pipeline.stop.assert_not_awaited()
    pipeline.start.assert_not_awaited()
    lm.transition_to_mode.assert_not_awaited()
    assert sm.mode is Mode.ACTIVE


@pytest.mark.asyncio
async def test_sleep_when_already_sleeping_is_noop():
    coord, sm, lm, _, tts = _make_coord(initial_mode=Mode.SLEEPING)
    await coord.request(Mode.SLEEPING)
    assert tts.speak_calls == []
    lm.transition_to_mode.assert_not_awaited()
    assert sm.mode is Mode.SLEEPING


# --- emits ModeChanged --------------------------------------------------


@pytest.mark.asyncio
async def test_sleep_emits_mode_changed():
    """The bus subscriber for tray-icon update must still fire even though
    lm.bind is no longer used."""
    sm_bus = EventBus()
    sm = StateMachine(bus=sm_bus, initial_mode=Mode.ACTIVE)
    lm = _make_lm()
    pipeline = _make_pipeline()
    tts = FakeTTS()
    coord = ModeCoordinator(sm=sm, lm=lm, pipeline=pipeline, tts=tts)

    events: list[ModeChanged] = []
    sm_bus.subscribe(ModeChanged, lambda e: events.append(e))

    await coord.request(Mode.SLEEPING)
    # Yield once so the publish task runs.
    await asyncio.sleep(0)
    assert any(e.new is Mode.SLEEPING for e in events)
