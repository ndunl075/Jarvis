"""Loadable protocol and LifecycleManager: orchestrates module load/unload
in declared order, in response to Mode transitions.

Design notes (kept here so future readers see the rationale):

- Loadable.load() / unload() must be idempotent. The protocol's `is_loaded`
  field is the source of truth; the manager additionally checks before
  calling, so a non-conforming implementation still won't see double-init.
- load_all is fail-fast with rollback: on failure of loadable N, unload
  loadables 1..N-1 in reverse and raise. Half-loaded states lie to the rest
  of the system. Rollback failures are logged, not propagated.
- transition_to_mode is asymmetric. Loading transitions (-> ACTIVE / ->
  MUTED-from-SLEEP) use load_all semantics (fail-fast, rollback). Unloading
  transitions (-> SLEEPING) are best-effort: log per-module failures,
  continue, reach the target Mode. User intent: "wake up" must succeed
  cleanly; "go to sleep" must always make progress.
- Ollama eviction is modeled as OllamaClient implementing Loadable. load()
  is a no-op (SPEC: "LLM loads lazily on first use"); unload() sends
  keep_alive:0. is_loaded tracks intent. Uniformity over special-casing.
- Load order is declared at the composition root: pass an ordered iterable
  to the constructor. register() appends for late additions. No decorators,
  no priority weights, no topological sort.
- Sequential, not concurrent: predictable ordering, no audio-device races
  on init.
- bind(bus) subscribes a handler that calls transition_to_mode on
  ModeChanged. transition_to_mode is also directly callable for tests.
- Module-specific runtime gating (e.g. "ignore wake-word while MUTED")
  is NOT a lifecycle concern; the audio pipeline subscribes to ModeChanged
  itself and handles that.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from jarvis.core.state_machine import Mode

if TYPE_CHECKING:
    from jarvis.core.events import EventBus, ModeChanged

log = logging.getLogger(__name__)


# --- protocol --------------------------------------------------------------


@runtime_checkable
class Loadable(Protocol):
    """A module that holds heavy resources (model weights, audio devices,
    network connections) and can be loaded / unloaded on demand.

    Contract:
    - load() and unload() are idempotent. Calling load() while is_loaded
      is True must be a no-op (and fast); same for unload() while False.
    - is_loaded must reflect the post-condition of the last successful
      load/unload call.
    - load() and unload() are coroutines so they can do I/O without
      blocking the loop. They should not rely on running in any particular
      thread (the lifecycle manager runs them on the asyncio loop).
    """

    name: str
    is_loaded: bool

    async def load(self) -> None: ...
    async def unload(self) -> None: ...


# --- which transitions touch which side -----------------------------------

# (old, new) Mode pairs that require load_all-on-the-Loadables.
# Per SPEC § Lifecycle Contract:
#   to ACTIVE (from SLEEP)  -> load
#   to MUTED  (from SLEEP)  -> "same as to ACTIVE then suppress wake word"
_LOADING_TRANSITIONS: frozenset[tuple[Mode, Mode]] = frozenset({
    (Mode.SLEEPING, Mode.ACTIVE),
    (Mode.SLEEPING, Mode.MUTED),
})

# (old, new) Mode pairs that require unload_all-on-the-Loadables.
#   to SLEEPING (from ACTIVE) -> unload + ollama evict
#   to SLEEPING (from MUTED)  -> unload + ollama evict
_UNLOADING_TRANSITIONS: frozenset[tuple[Mode, Mode]] = frozenset({
    (Mode.ACTIVE, Mode.SLEEPING),
    (Mode.MUTED, Mode.SLEEPING),
})

# Everything else (ACTIVE<->MUTED) is a Loadable-no-op; the audio pipeline
# handles wake-word suppression on its own.


# --- manager ---------------------------------------------------------------


class LifecycleManager:
    def __init__(
        self,
        loadables: Iterable[Loadable] = (),
        bus: EventBus | None = None,
    ) -> None:
        self._loadables: list[Loadable] = list(loadables)
        self._bus: EventBus | None = bus
        self._unsubscribe = None  # set by bind()

    # -- registration --

    def register(self, loadable: Loadable) -> None:
        """Append a loadable. New entry loads last and unloads first
        (reverse-order unload preserves the dependency-friendly invariant)."""
        self._loadables.append(loadable)

    @property
    def loadables(self) -> tuple[Loadable, ...]:
        return tuple(self._loadables)

    # -- load_all / unload_all --

    async def load_all(self) -> None:
        """Load every Loadable in declared order. Fail-fast: on the first
        failure, best-effort unload everything that successfully loaded in
        this call (in reverse), then re-raise the original exception."""
        loaded_so_far: list[Loadable] = []
        for ld in self._loadables:
            if ld.is_loaded:
                continue
            try:
                await ld.load()
            except Exception:
                log.error("load failed for %r; rolling back %d loaded module(s)",
                          ld.name, len(loaded_so_far))
                await self._best_effort_unload(reversed(loaded_so_far))
                raise
            loaded_so_far.append(ld)

    async def unload_all(self) -> None:
        """Unload every Loadable in reverse declared order. Best-effort:
        a failure on one Loadable does not stop the others."""
        await self._best_effort_unload(reversed(self._loadables))

    async def _best_effort_unload(self, loadables: Iterable[Loadable]) -> None:
        for ld in loadables:
            if not ld.is_loaded:
                continue
            try:
                await ld.unload()
            except Exception:
                log.exception("unload failed for %r; continuing", ld.name)

    # -- mode transitions --

    async def transition_to_mode(self, old: Mode, new: Mode) -> None:
        """Run the load/unload subset SPEC requires for this Mode pair.

        Loading transitions (fail-fast, rollback) and unloading transitions
        (best-effort) per the asymmetric policy in the module docstring.
        Same-Mode pairs and the ACTIVE<->MUTED pairs are no-ops here; the
        audio pipeline handles wake-word gating on its own."""
        if old is new:
            return
        pair = (old, new)
        if pair in _LOADING_TRANSITIONS:
            log.info("lifecycle: loading for transition %s -> %s", old.name, new.name)
            await self.load_all()
        elif pair in _UNLOADING_TRANSITIONS:
            log.info("lifecycle: unloading for transition %s -> %s", old.name, new.name)
            await self.unload_all()
        else:
            log.debug("lifecycle: no-op transition %s -> %s", old.name, new.name)

    # -- bus binding --

    def bind(self, bus: EventBus) -> None:
        """Subscribe to ModeChanged so transitions are auto-driven by the
        state machine. Idempotent: re-binding replaces the prior subscription."""
        from jarvis.core.events import ModeChanged  # late, avoids cycle

        if self._unsubscribe is not None:
            self._unsubscribe()
        self._bus = bus

        async def _on_mode_changed(event: ModeChanged) -> None:
            await self.transition_to_mode(event.old, event.new)

        self._unsubscribe = bus.subscribe(ModeChanged, _on_mode_changed)

    def unbind(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
