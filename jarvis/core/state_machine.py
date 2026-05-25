"""State machine: Mode + ConversationalState with explicit transition tables.

Design notes (kept here so future readers see the rationale):

- Two orthogonal axes per SPEC § State Machine. Mode is the top-level lifecycle
  axis (ACTIVE / MUTED / SLEEPING); ConversationalState is the per-interaction
  axis and is only meaningful when Mode == ACTIVE.
- Transition legality is encoded as `dict[State, frozenset[State]]` per axis,
  defined as module-level constants. O(1) lookup, immutable, and reads
  top-to-bottom like the SPEC bullet list. The "any state to IDLE" rule is
  expressed by adding IDLE to every CS source's allowed-targets set
  explicitly -- no implicit special case.
- This module owns *legality* only. Per-transition *actions* (unloading
  modules, evicting Ollama, cancelling TTS) are the responsibility of
  `lifecycle.py` and the audio pipeline, both of which subscribe to
  ModeChanged. The SM never imports from those layers.
- Same-state transitions (e.g. set_mode(ACTIVE) when already ACTIVE) are a
  no-op: no event, no exception. SPEC's "any state to IDLE" implies IDLE->IDLE
  is allowed; for symmetry and UI ergonomics (clicking "mute" while already
  muted shouldn't pop) the same rule applies on both axes.
- On Mode change: CS is forced to IDLE first (emitting ConversationalState-
  Changed if it wasn't already IDLE), then the mode flips and ModeChanged
  fires. Subscribers to ModeChanged thus see a coherent, settled snapshot.
- The bus is optional. Production wires it; pure transition-logic tests can
  pass nothing and assert observable state without spinning up a loop.
- IllegalTransition carries (axis, old, new) so handlers and tests can
  diagnose precisely without parsing strings.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from jarvis.core.events import EventBus

log = logging.getLogger(__name__)


# --- enums (also imported by core.events; do not redefine elsewhere) -------


class Mode(Enum):
    ACTIVE = "active"
    MUTED = "muted"
    SLEEPING = "sleeping"


class ConversationalState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


# --- transition tables -----------------------------------------------------

# SPEC § State Machine "Legal Mode transitions": ACTIVE<->MUTED, ACTIVE<->SLEEPING,
# MUTED<->SLEEPING. (Self-transitions are no-ops, handled before the lookup.)
_LEGAL_MODE_TRANSITIONS: dict[Mode, frozenset[Mode]] = {
    Mode.ACTIVE: frozenset({Mode.MUTED, Mode.SLEEPING}),
    Mode.MUTED: frozenset({Mode.ACTIVE, Mode.SLEEPING}),
    Mode.SLEEPING: frozenset({Mode.ACTIVE, Mode.MUTED}),
}

# SPEC § State Machine "Legal ConversationalState transitions". IDLE is in
# every source's set explicitly (the "any state to IDLE" rule made
# unambiguous, no hidden special case in the check function).
_LEGAL_CS_TRANSITIONS: dict[ConversationalState, frozenset[ConversationalState]] = {
    ConversationalState.IDLE: frozenset(
        {ConversationalState.LISTENING, ConversationalState.IDLE}
    ),
    ConversationalState.LISTENING: frozenset(
        {ConversationalState.THINKING, ConversationalState.IDLE}
    ),
    ConversationalState.THINKING: frozenset(
        {ConversationalState.SPEAKING, ConversationalState.IDLE}
    ),
    ConversationalState.SPEAKING: frozenset(
        {ConversationalState.IDLE, ConversationalState.LISTENING}
    ),
}


# --- exception -------------------------------------------------------------


Axis = Literal["mode", "conversational_state"]


class IllegalTransition(RuntimeError):  # noqa: N818  (name fixed by SPEC § State Machine)
    """Raised by StateMachine when a requested transition is not in the
    legal table for that axis. Attributes (axis, old, new) are queryable
    so tests and handlers can diagnose without parsing the message."""

    def __init__(self, axis: Axis, old: Enum, new: Enum) -> None:
        super().__init__(
            f"illegal {axis} transition: {old.name} -> {new.name}"
        )
        self.axis: Axis = axis
        self.old: Enum = old
        self.new: Enum = new


# --- state machine ---------------------------------------------------------


class StateMachine:
    """Owns the (Mode, ConversationalState) tuple. Validates transitions
    against the tables above; emits ModeChanged / ConversationalStateChanged
    on the supplied bus (if any).

    Not thread-safe by itself. The audio thread / asyncio loop is the sole
    intended driver in production; UI inputs cross to the loop via the bus
    or Qt-signal -> loop bridges before reaching this object.
    """

    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        initial_mode: Mode = Mode.ACTIVE,
        initial_conversational_state: ConversationalState = ConversationalState.IDLE,
    ) -> None:
        # SPEC: "ConversationalState is only meaningful when Mode == ACTIVE.
        # In MUTED or SLEEPING, ConversationalState is forced to IDLE."
        if (
            initial_mode is not Mode.ACTIVE
            and initial_conversational_state is not ConversationalState.IDLE
        ):
            raise ValueError(
                f"initial_conversational_state must be IDLE when "
                f"initial_mode is {initial_mode.name}"
            )
        self._bus = bus
        self._mode = initial_mode
        self._cs = initial_conversational_state

    # -- accessors --

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def conversational_state(self) -> ConversationalState:
        return self._cs

    # -- mode --

    def set_mode(self, new: Mode) -> None:
        old = self._mode
        if new is old:
            return  # no-op self-transition
        if new not in _LEGAL_MODE_TRANSITIONS[old]:
            raise IllegalTransition("mode", old, new)

        # Force CS to IDLE first so subscribers to ModeChanged see a settled
        # snapshot. If CS was already IDLE this emits nothing.
        if self._cs is not ConversationalState.IDLE:
            self._set_cs_internal(ConversationalState.IDLE)

        self._mode = new
        log.debug("mode transition: %s -> %s", old.name, new.name)
        self._emit_mode(old, new)

    # -- conversational state --

    def set_conversational_state(self, new: ConversationalState) -> None:
        # SPEC: CS transitions only meaningful in ACTIVE mode. Trying to
        # leave IDLE while not ACTIVE is a real bug -- raise.
        if self._mode is not Mode.ACTIVE and new is not ConversationalState.IDLE:
            raise IllegalTransition("conversational_state", self._cs, new)
        self._set_cs_internal(new)

    def _set_cs_internal(self, new: ConversationalState) -> None:
        old = self._cs
        if new is old:
            return  # no-op self-transition
        if new not in _LEGAL_CS_TRANSITIONS[old]:
            raise IllegalTransition("conversational_state", old, new)
        self._cs = new
        log.debug("conversational_state transition: %s -> %s", old.name, new.name)
        self._emit_cs(old, new)

    # -- emission --

    def _emit_mode(self, old: Mode, new: Mode) -> None:
        if self._bus is None:
            return
        # Late import to avoid a module-level cycle (events imports the
        # enums from this module).
        from jarvis.core.events import ModeChanged

        self._bus.publish(ModeChanged(old=old, new=new))

    def _emit_cs(self, old: ConversationalState, new: ConversationalState) -> None:
        if self._bus is None:
            return
        from jarvis.core.events import ConversationalStateChanged

        self._bus.publish(ConversationalStateChanged(old=old, new=new))
