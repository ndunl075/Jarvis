"""ModeCoordinator: serializes Mode requests, owns the sleep-confirmation
phrase, drives LifecycleManager and the audio pipeline around sleep/wake.

Phase 6 Task 1 ([ARCH]). Approved design with race rules:

  - Sleep is a two-phase action: (1) speak "Going to sleep, sir." through
    the TTS that is about to be unloaded; (2) stop the pipeline,
    transition to SLEEPING via StateMachine, and run LifecycleManager
    unload_all to free audio modules and evict the Ollama model.
  - The confirmation window is a race surface: the user can press
    Sleep / Mute / Wake again between (1) and (2). Rules:
      * Second Sleep during confirmation: ignored (logged).
      * Mute during confirmation: ignored (logged). Simpler than
        queueing; the user can mute after wake completes.
      * Wake during confirmation: cancels the sleep. TTS is aborted,
        no Mode transition happens, the system stays in its prior Mode
        (ACTIVE or MUTED). The user changed their mind — honour it.
  - Wake from SLEEPING is also two-phase: transition to ACTIVE, then
    drive LifecycleManager.load_all and restart the pipeline. Failure
    in load_all is fatal to the wake (re-raised after best-effort
    rollback inside the lifecycle manager); the SM state is rolled
    back to SLEEPING and the original error is propagated.
  - Mute toggles (ACTIVE <-> MUTED) bypass the sleep machinery entirely
    and just call sm.set_mode.

Why this lives outside LifecycleManager:
  - LifecycleManager is TTS-unaware and must stay so. The coordinator
    is the only object that owns both the lifecycle list and the TTS
    handle, so the speak-then-unload sequence belongs here.
  - The composition root constructs ONE coordinator and routes every
    Mode request (tray, hotkeys, future auto-sleep) through it. The
    StateMachine remains the source of truth for the current Mode;
    the coordinator is the sole writer to it for ACTIVE/SLEEPING
    transitions.

The audio loop is the single thread of execution: every request_*
coroutine is scheduled there via asyncio.run_coroutine_threadsafe by
the calling thread. An internal asyncio.Lock serializes request
processing so a "wake just after sleep" sequence cannot interleave."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from jarvis.core.lifecycle import LifecycleManager
from jarvis.core.state_machine import IllegalTransition, Mode, StateMachine

log = logging.getLogger(__name__)

SLEEP_CONFIRMATION_PHRASE = "Going to sleep, sir."


class _TTS(Protocol):
    """Subset of TextToSpeech the coordinator uses. Anything with these
    awaitable methods works -- production PiperTTS, test doubles, etc."""

    is_loaded: bool

    async def speak(self, text: str) -> None: ...
    async def cancel(self) -> None: ...


class _Pipeline(Protocol):
    """Subset of AudioPipeline the coordinator uses."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class ModeCoordinator:
    def __init__(
        self,
        *,
        sm: StateMachine,
        lm: LifecycleManager,
        pipeline: _Pipeline,
        tts: _TTS,
        sleep_confirmation: bool = True,
        sleep_phrase: str = SLEEP_CONFIRMATION_PHRASE,
    ) -> None:
        self._sm = sm
        self._lm = lm
        self._pipeline = pipeline
        self._tts = tts
        self._sleep_confirmation = sleep_confirmation
        self._sleep_phrase = sleep_phrase
        # Re-entrancy/race guard for the sleep-confirmation phase. Set
        # True between "begin speaking" and "transition committed or
        # cancelled". Mute/Sleep are ignored, Wake cancels.
        self._sleep_in_progress: bool = False
        # Flipped True when a Wake request arrives during the
        # confirmation phase. request_sleep checks this after speak()
        # returns/aborts and bails out of the transition.
        self._wake_during_sleep: bool = False
        # Serializes coordinator entry points so request_wake cannot
        # land between request_sleep's pre-checks and its commit.
        self._lock: asyncio.Lock = asyncio.Lock()

    # -- public entry point ------------------------------------------------

    async def request(self, target: Mode) -> None:
        """Route a Mode request to the right private handler. Called from
        the audio loop (after the caller's thread marshalled via
        run_coroutine_threadsafe). Idempotent: a request whose target
        equals the current Mode is a fast no-op."""
        current = self._sm.mode
        if target is current and not self._sleep_in_progress:
            return
        if target is Mode.SLEEPING:
            await self._request_sleep()
        elif target is Mode.ACTIVE and current is Mode.SLEEPING:
            await self._request_wake()
        elif target is Mode.ACTIVE and self._sleep_in_progress:
            # Wake-mid-confirmation: short-circuit the in-flight sleep.
            await self._request_wake()
        else:
            # ACTIVE <-> MUTED (no lifecycle work) and MUTED <- SLEEPING.
            # (target == SLEEPING and target == ACTIVE with current ==
            # SLEEPING are already handled above.)
            if current is Mode.SLEEPING and target is Mode.MUTED:
                # Treat "Mute" from SLEEPING as Wake-then-Mute. Per SPEC
                # § Lifecycle Contract this requires loading modules.
                await self._request_wake_to(Mode.MUTED)
                return
            await self._request_mute_or_unmute(target)

    # -- sleep -------------------------------------------------------------

    async def _request_sleep(self) -> None:
        async with self._lock:
            if self._sm.mode is Mode.SLEEPING:
                return
            if self._sleep_in_progress:
                log.info("sleep already in progress; ignoring duplicate request")
                return
            self._sleep_in_progress = True
            self._wake_during_sleep = False

        try:
            if self._sleep_confirmation and getattr(self._tts, "is_loaded", False):
                try:
                    await self._tts.speak(self._sleep_phrase)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception(
                        "sleep confirmation speak failed; continuing to unload"
                    )
            # Re-check after the speak yields. A Wake request landing on
            # the loop while speak was awaiting will have flipped this.
            if self._wake_during_sleep:
                log.info("sleep cancelled by wake request mid-confirmation")
                return

            async with self._lock:
                if self._wake_during_sleep:
                    log.info("sleep cancelled by wake request just before commit")
                    return
                await self._commit_sleep()
        finally:
            self._sleep_in_progress = False
            self._wake_during_sleep = False

    async def _commit_sleep(self) -> None:
        """Pipeline.stop -> sm.set_mode(SLEEPING) -> lm.unload.

        Pipeline must stop before unload so its frame loop does not feed
        a half-unloaded STT. sm.set_mode emits ModeChanged for the tray.
        unload_all is best-effort per LifecycleManager policy."""
        old = self._sm.mode
        try:
            await self._pipeline.stop()
        except Exception:
            log.exception("pipeline stop failed during sleep; continuing to unload")
        try:
            self._sm.set_mode(Mode.SLEEPING)
        except IllegalTransition:
            log.exception("illegal transition to SLEEPING from %s", old.name)
            return
        await self._lm.transition_to_mode(old, Mode.SLEEPING)

    # -- wake --------------------------------------------------------------

    async def _request_wake(self) -> None:
        """Wake to ACTIVE. If a sleep confirmation is mid-flight, abort
        it instead of running a real wake."""
        if self._sleep_in_progress:
            self._wake_during_sleep = True
            try:
                await self._tts.cancel()
            except Exception:
                log.exception("tts cancel during wake-mid-confirmation failed")
            return
        await self._request_wake_to(Mode.ACTIVE)

    async def _request_wake_to(self, target: Mode) -> None:
        """Wake from SLEEPING to `target` (ACTIVE or MUTED). lm.load_all
        first, sm.set_mode second, pipeline.start last."""
        async with self._lock:
            old = self._sm.mode
            if old is not Mode.SLEEPING:
                return
            try:
                await self._lm.transition_to_mode(old, target)
            except Exception:
                log.exception(
                    "load failed during wake to %s; staying SLEEPING", target.name
                )
                raise
            try:
                self._sm.set_mode(target)
            except IllegalTransition:
                log.exception(
                    "illegal transition to %s from %s after load",
                    target.name, old.name,
                )
                return
            try:
                await self._pipeline.start()
            except Exception:
                log.exception("pipeline start after wake failed")

    # -- mute / unmute -----------------------------------------------------

    async def _request_mute_or_unmute(self, target: Mode) -> None:
        async with self._lock:
            old = self._sm.mode
            if target is old:
                return
            try:
                self._sm.set_mode(target)
            except IllegalTransition:
                log.exception(
                    "illegal mute transition %s -> %s", old.name, target.name
                )
                return
            # No lifecycle work for ACTIVE <-> MUTED, but call the manager
            # for uniformity (it logs the no-op and returns).
            await self._lm.transition_to_mode(old, target)
