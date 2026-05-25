"""ResourceMonitor: idle-time auto-sleep trigger.

Subscribes to ConversationalStateChanged on the bus. Any CS transition
into a non-IDLE state (LISTENING, THINKING, SPEAKING) resets the idle
clock, because the system is actively processing user input.  When the
clock exceeds idle_timeout_minutes while Mode is ACTIVE, the monitor
calls coordinator.request(Mode.SLEEPING) to hand off to ModeCoordinator
for the speak-then-unload sequence.

Disabled by default (auto_sleep_enabled=False).  Enable via
Settings → General → "Auto-sleep when idle".

Low-battery and Windows-user-idle rules are scaffolded in LifecycleConfig
but are no-ops in this implementation (Phase 6 Task 2 scope is idle time
only; the other rules are reserved for a later task).

Threading
---------
start() / close() are called from the audio asyncio loop (inside
_audio_main).  The bus subscriber (_on_cs_changed) is also dispatched on
the audio loop.  No cross-thread coordination is needed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Protocol

from jarvis.core.events import ConversationalStateChanged, EventBus
from jarvis.core.state_machine import ConversationalState, Mode

log = logging.getLogger(__name__)

# How often the monitor loop wakes to check elapsed idle time.
# 10 s gives ≤10 s over-sleep vs. the configured threshold; fine for
# a "30 minute" granularity feature.  Injected as _check_interval_s
# in the constructor so tests can pass a tiny value without sleeping.
_IDLE_CHECK_INTERVAL_S: float = 10.0


class _Coordinator(Protocol):
    async def request(self, target: Mode) -> None: ...


class _SM(Protocol):
    @property
    def mode(self) -> Mode: ...


class ResourceMonitor:
    """Idle-time auto-sleep trigger.

    Parameters
    ----------
    bus:
        EventBus to subscribe ConversationalStateChanged from.
    coordinator:
        ModeCoordinator (or compatible) whose request() is called on
        sleep trigger.
    sm:
        StateMachine (or compatible) whose .mode property is checked
        before firing sleep so we don't trigger from MUTED or SLEEPING.
    auto_sleep_enabled:
        When False the monitor runs its loop but never fires.
    idle_timeout_minutes:
        Minutes of inactivity before sleep is requested.
    _check_interval_s:
        Internal poll cadence (seconds).  Defaults to 10 s; pass a
        small value in tests to avoid actually waiting.
    _now:
        Callable returning current time as a float (seconds).  Defaults
        to time.monotonic.  Pass a controlled callable in tests.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        coordinator: _Coordinator,
        sm: _SM,
        auto_sleep_enabled: bool,
        idle_timeout_minutes: int,
        _check_interval_s: float = _IDLE_CHECK_INTERVAL_S,
        _now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bus = bus
        self._coordinator = coordinator
        self._sm = sm
        self._auto_sleep_enabled = auto_sleep_enabled
        self._idle_timeout_s = idle_timeout_minutes * 60.0
        self._check_interval_s = _check_interval_s
        self._now = _now
        self._last_activity_ts: float = _now()
        self._task: asyncio.Task | None = None

        bus.subscribe(ConversationalStateChanged, self._on_cs_changed)

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        """Schedule the monitor loop on the running event loop. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._monitor_loop())

    def stop(self) -> None:
        """Alias for close()."""
        self.close()

    def reconfigure(self, *, auto_sleep_enabled: bool, idle_timeout_minutes: int) -> None:
        """Update auto-sleep settings in-place without restarting the loop."""
        self._auto_sleep_enabled = auto_sleep_enabled
        self._idle_timeout_s = idle_timeout_minutes * 60.0

    def close(self) -> None:
        """Cancel the monitor loop task. Idempotent."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    # -- bus subscriber ----------------------------------------------------

    def _on_cs_changed(self, event: ConversationalStateChanged) -> None:
        if event.new is not ConversationalState.IDLE:
            self._last_activity_ts = self._now()

    # -- monitor loop ------------------------------------------------------

    async def _monitor_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._check_interval_s)
                if not self._auto_sleep_enabled:
                    continue
                if self._sm.mode is not Mode.ACTIVE:
                    continue
                elapsed = self._now() - self._last_activity_ts
                if elapsed >= self._idle_timeout_s:
                    log.info(
                        "auto-sleep: idle for %.0f s (threshold %.0f s); "
                        "requesting sleep",
                        elapsed,
                        self._idle_timeout_s,
                    )
                    await self._coordinator.request(Mode.SLEEPING)
                    # Reset timer after firing so we don't spam sleep requests
                    # if coordinator is slow or the system stays ACTIVE.
                    self._last_activity_ts = self._now()
        except asyncio.CancelledError:
            pass
