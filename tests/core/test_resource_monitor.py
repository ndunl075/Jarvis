"""Tests for jarvis.core.resource_monitor.ResourceMonitor.

Strategy for avoiding real 30-minute waits:
  - Inject _check_interval_s=0.001 so the poll loop fires almost immediately.
  - Inject _now as a FakeTime counter so we can advance "time" without
    actually waiting.
  - All tests are async (asyncio_mode="auto" in pyproject.toml).

Coverage:
  - Disabled config: loop runs but request() never called.
  - Enabled + timeout elapsed: request(SLEEPING) called exactly once,
    then timer resets so a second poll doesn't double-fire.
  - Enabled + activity resets timer: CS transition to non-IDLE resets
    the clock; request() is NOT called within the original window.
  - Mode != ACTIVE: no sleep request even after timeout.
  - close() cancels the background task cleanly.
  - close() is idempotent (second call does not raise).
  - start() is idempotent (second call does not create a duplicate task).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from jarvis.core.events import ConversationalStateChanged, EventBus
from jarvis.core.resource_monitor import ResourceMonitor
from jarvis.core.state_machine import ConversationalState, Mode, StateMachine

# --- helpers ------------------------------------------------------------------


class FakeTime:
    """Controllable monotonic clock for injecting into ResourceMonitor._now."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _make_sm(mode: Mode = Mode.ACTIVE) -> StateMachine:
    return StateMachine(initial_mode=mode)


def _make_monitor(
    *,
    bus: EventBus,
    coordinator,
    sm: StateMachine,
    enabled: bool = True,
    timeout_minutes: int = 1,
    clock: FakeTime | None = None,
) -> ResourceMonitor:
    if clock is None:
        clock = FakeTime()
    return ResourceMonitor(
        bus=bus,
        coordinator=coordinator,
        sm=sm,
        auto_sleep_enabled=enabled,
        idle_timeout_minutes=timeout_minutes,
        _check_interval_s=0.001,
        _now=clock,
    )


def _make_bus() -> EventBus:
    # Every caller below is an `async def` test (pytest-asyncio auto mode), so
    # there is always a running loop to bind to. Binding explicitly keeps the
    # bus usable from the worker threads the monitor spawns.
    return EventBus(loop=asyncio.get_running_loop())


# --- tests --------------------------------------------------------------------


async def test_disabled_monitor_never_fires():
    """With auto_sleep_enabled=False the coordinator is never called."""
    bus = _make_bus()
    sm = _make_sm()
    coord = MagicMock()
    coord.request = AsyncMock()
    clock = FakeTime()

    monitor = _make_monitor(
        bus=bus, coordinator=coord, sm=sm, enabled=False, clock=clock
    )
    monitor.start()
    # Advance clock well past a 1-minute threshold.
    clock.advance(120.0)
    # Give the loop several poll cycles.
    await asyncio.sleep(0.05)
    monitor.close()

    coord.request.assert_not_awaited()


async def test_enabled_monitor_fires_after_timeout():
    """request(SLEEPING) is called once the idle threshold is exceeded."""
    bus = _make_bus()
    sm = _make_sm()
    coord = MagicMock()
    coord.request = AsyncMock()
    clock = FakeTime()

    monitor = _make_monitor(
        bus=bus, coordinator=coord, sm=sm, enabled=True, timeout_minutes=1, clock=clock
    )
    monitor.start()
    # 1 min = 60 s; advance past it.
    clock.advance(61.0)
    await asyncio.sleep(0.05)
    monitor.close()

    coord.request.assert_awaited_once_with(Mode.SLEEPING)


async def test_timer_resets_after_firing_no_double_fire():
    """After request() fires, the internal timer resets so a second poll
    cycle does not immediately fire again."""
    bus = _make_bus()
    sm = _make_sm()
    coord = MagicMock()
    coord.request = AsyncMock()
    clock = FakeTime()

    monitor = _make_monitor(
        bus=bus, coordinator=coord, sm=sm, enabled=True, timeout_minutes=1, clock=clock
    )
    monitor.start()
    # First fire: advance past threshold.
    clock.advance(61.0)
    await asyncio.sleep(0.05)
    # After firing, the monitor resets _last_activity_ts to clock() = 61.
    # Advance only 30 more seconds — still under 60 s since reset.
    clock.advance(30.0)
    await asyncio.sleep(0.05)
    monitor.close()

    # Should have fired exactly once, not twice.
    assert coord.request.await_count == 1


async def test_activity_resets_timer():
    """A ConversationalStateChanged to non-IDLE resets the clock so the
    monitor does NOT fire within the original window."""
    bus = _make_bus()
    sm = _make_sm()
    coord = MagicMock()
    coord.request = AsyncMock()
    clock = FakeTime()

    monitor = _make_monitor(
        bus=bus, coordinator=coord, sm=sm, enabled=True, timeout_minutes=1, clock=clock
    )
    monitor.start()

    # Advance 40 s (under 60 s threshold), then simulate activity.
    clock.advance(40.0)
    bus.publish(
        ConversationalStateChanged(
            old=ConversationalState.IDLE, new=ConversationalState.LISTENING
        )
    )
    # Give the subscriber a chance to run.
    await asyncio.sleep(0.01)

    # Advance another 40 s — only 40 s since last activity, still under 60 s.
    clock.advance(40.0)
    await asyncio.sleep(0.05)
    monitor.close()

    coord.request.assert_not_awaited()


async def test_activity_to_idle_does_not_reset_timer():
    """CS transition back to IDLE does NOT reset the timer — IDLE means
    the conversation just ended, which is exactly when inactivity counts."""
    bus = _make_bus()
    sm = _make_sm()
    coord = MagicMock()
    coord.request = AsyncMock()
    clock = FakeTime()

    monitor = _make_monitor(
        bus=bus, coordinator=coord, sm=sm, enabled=True, timeout_minutes=1, clock=clock
    )
    monitor.start()

    # Advance 40 s, then fire an IDLE transition (should NOT reset timer).
    clock.advance(40.0)
    bus.publish(
        ConversationalStateChanged(
            old=ConversationalState.SPEAKING, new=ConversationalState.IDLE
        )
    )
    await asyncio.sleep(0.01)

    # Advance another 25 s — 65 s total from start, threshold is 60 s.
    clock.advance(25.0)
    await asyncio.sleep(0.05)
    monitor.close()

    coord.request.assert_awaited_once_with(Mode.SLEEPING)


async def test_non_active_mode_does_not_fire():
    """Auto-sleep does not trigger when Mode != ACTIVE (e.g. SLEEPING)."""
    bus = _make_bus()
    sm = _make_sm(mode=Mode.SLEEPING)
    coord = MagicMock()
    coord.request = AsyncMock()
    clock = FakeTime()

    monitor = _make_monitor(
        bus=bus, coordinator=coord, sm=sm, enabled=True, timeout_minutes=1, clock=clock
    )
    monitor.start()
    clock.advance(120.0)
    await asyncio.sleep(0.05)
    monitor.close()

    coord.request.assert_not_awaited()


async def test_muted_mode_does_not_fire():
    """Auto-sleep does not trigger when Mode == MUTED."""
    bus = _make_bus()
    sm = _make_sm(mode=Mode.MUTED)
    coord = MagicMock()
    coord.request = AsyncMock()
    clock = FakeTime()

    monitor = _make_monitor(
        bus=bus, coordinator=coord, sm=sm, enabled=True, timeout_minutes=1, clock=clock
    )
    monitor.start()
    clock.advance(120.0)
    await asyncio.sleep(0.05)
    monitor.close()

    coord.request.assert_not_awaited()


async def test_close_cancels_task():
    """close() stops the background task; further poll cycles do not run."""
    bus = _make_bus()
    sm = _make_sm()
    coord = MagicMock()
    coord.request = AsyncMock()
    clock = FakeTime()

    monitor = _make_monitor(
        bus=bus, coordinator=coord, sm=sm, enabled=True, timeout_minutes=1, clock=clock
    )
    monitor.start()
    task = monitor._task
    assert task is not None

    monitor.close()
    await asyncio.sleep(0.02)

    assert task.cancelled() or task.done()
    # Advance time and confirm no request fires after close.
    clock.advance(120.0)
    await asyncio.sleep(0.02)
    coord.request.assert_not_awaited()


async def test_close_is_idempotent():
    """Calling close() twice does not raise."""
    bus = _make_bus()
    sm = _make_sm()
    coord = MagicMock()
    coord.request = AsyncMock()

    monitor = _make_monitor(bus=bus, coordinator=coord, sm=sm)
    monitor.start()
    monitor.close()
    monitor.close()  # must not raise


async def test_start_is_idempotent():
    """Calling start() twice does not create a second background task."""
    bus = _make_bus()
    sm = _make_sm()
    coord = MagicMock()
    coord.request = AsyncMock()

    monitor = _make_monitor(bus=bus, coordinator=coord, sm=sm)
    monitor.start()
    first_task = monitor._task
    monitor.start()
    assert monitor._task is first_task
    monitor.close()


async def test_close_before_start_is_safe():
    """close() on a never-started monitor does not raise."""
    bus = _make_bus()
    sm = _make_sm()
    coord = MagicMock()
    coord.request = AsyncMock()

    monitor = _make_monitor(bus=bus, coordinator=coord, sm=sm)
    monitor.close()  # must not raise
