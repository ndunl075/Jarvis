"""Tests for jarvis.core.state_machine.

Coverage matrix:
- Every legal Mode transition succeeds.
- Every illegal Mode transition raises (excluding self-transitions, which
  are documented no-ops).
- Every legal ConversationalState transition succeeds.
- Every illegal ConversationalState transition raises.
- Same-state transitions are no-ops on both axes.
- Mode change forces CS to IDLE before flipping mode (event ordering).
- CS transitions to non-IDLE while Mode != ACTIVE are illegal.
- Event emission only happens when a bus is wired.
- IllegalTransition carries (axis, old, new).
- Initial-state guard.
"""

from __future__ import annotations

from itertools import product

import pytest

from jarvis.core.events import (
    ConversationalStateChanged,
    EventBus,
    ModeChanged,
)
from jarvis.core.state_machine import (
    ConversationalState,
    IllegalTransition,
    Mode,
    StateMachine,
)

LEGAL_MODE_PAIRS = {
    (Mode.ACTIVE, Mode.MUTED),
    (Mode.ACTIVE, Mode.SLEEPING),
    (Mode.MUTED, Mode.ACTIVE),
    (Mode.MUTED, Mode.SLEEPING),
    (Mode.SLEEPING, Mode.ACTIVE),
    (Mode.SLEEPING, Mode.MUTED),
}

LEGAL_CS_PAIRS = {
    (ConversationalState.IDLE, ConversationalState.LISTENING),
    (ConversationalState.LISTENING, ConversationalState.THINKING),
    (ConversationalState.LISTENING, ConversationalState.IDLE),
    (ConversationalState.THINKING, ConversationalState.SPEAKING),
    (ConversationalState.THINKING, ConversationalState.IDLE),
    (ConversationalState.SPEAKING, ConversationalState.IDLE),
    (ConversationalState.SPEAKING, ConversationalState.LISTENING),
}


# --- legal transitions ----------------------------------------------------


@pytest.mark.parametrize("old,new", sorted(LEGAL_MODE_PAIRS, key=lambda p: (p[0].name, p[1].name)))
def test_every_legal_mode_transition_succeeds(old: Mode, new: Mode):
    sm = StateMachine(initial_mode=old)
    sm.set_mode(new)
    assert sm.mode is new


@pytest.mark.parametrize(
    "old,new",
    sorted(LEGAL_CS_PAIRS, key=lambda p: (p[0].name, p[1].name)),
)
def test_every_legal_cs_transition_succeeds(
    old: ConversationalState, new: ConversationalState
):
    sm = StateMachine(initial_conversational_state=old)
    sm.set_conversational_state(new)
    assert sm.conversational_state is new


# --- illegal transitions --------------------------------------------------


@pytest.mark.parametrize(
    "old,new",
    [
        (a, b)
        for a, b in product(Mode, Mode)
        if a is not b and (a, b) not in LEGAL_MODE_PAIRS
    ],
)
def test_every_illegal_mode_transition_raises(old: Mode, new: Mode):
    # The Mode table is fully connected aside from self-transitions, so this
    # parametrize set is empty today; the test exists so that if anyone narrows
    # the legal table in the future, the failure mode is enforced automatically.
    sm = StateMachine(initial_mode=old)
    with pytest.raises(IllegalTransition) as excinfo:
        sm.set_mode(new)
    assert excinfo.value.axis == "mode"
    assert excinfo.value.old is old
    assert excinfo.value.new is new


@pytest.mark.parametrize(
    "old,new",
    [
        (a, b)
        for a, b in product(ConversationalState, ConversationalState)
        if a is not b and (a, b) not in LEGAL_CS_PAIRS
    ],
)
def test_every_illegal_cs_transition_raises(
    old: ConversationalState, new: ConversationalState
):
    sm = StateMachine(initial_conversational_state=old)
    with pytest.raises(IllegalTransition) as excinfo:
        sm.set_conversational_state(new)
    assert excinfo.value.axis == "conversational_state"
    assert excinfo.value.old is old
    assert excinfo.value.new is new


# Specific call-outs from the SPEC's illegal list:
def test_idle_to_speaking_illegal():
    sm = StateMachine()
    with pytest.raises(IllegalTransition):
        sm.set_conversational_state(ConversationalState.SPEAKING)


def test_idle_to_thinking_illegal():
    sm = StateMachine()
    with pytest.raises(IllegalTransition):
        sm.set_conversational_state(ConversationalState.THINKING)


def test_listening_to_speaking_illegal():
    sm = StateMachine(initial_conversational_state=ConversationalState.LISTENING)
    with pytest.raises(IllegalTransition):
        sm.set_conversational_state(ConversationalState.SPEAKING)


def test_thinking_to_listening_illegal():
    sm = StateMachine(initial_conversational_state=ConversationalState.THINKING)
    with pytest.raises(IllegalTransition):
        sm.set_conversational_state(ConversationalState.LISTENING)


# --- self-transitions are no-ops -----------------------------------------


@pytest.mark.parametrize("mode", list(Mode))
async def test_self_mode_transition_is_noop(mode: Mode):
    bus = EventBus()
    seen: list[ModeChanged] = []
    bus.subscribe(ModeChanged, lambda e: seen.append(e))
    sm = StateMachine(bus=bus, initial_mode=mode)
    sm.set_mode(mode)
    await bus.publish_and_wait(ModeChanged(old=mode, new=mode))  # flush bus
    # The handler will see ONLY the explicit publish above (one event), not
    # one from the no-op set_mode call.
    assert len(seen) == 1


@pytest.mark.parametrize("cs", list(ConversationalState))
async def test_self_cs_transition_is_noop(cs: ConversationalState):
    bus = EventBus()
    seen: list[ConversationalStateChanged] = []
    bus.subscribe(ConversationalStateChanged, lambda e: seen.append(e))
    # IDLE under any mode; non-IDLE only under ACTIVE.
    sm = StateMachine(
        bus=bus,
        initial_mode=Mode.ACTIVE,
        initial_conversational_state=cs,
    )
    sm.set_conversational_state(cs)
    await bus.publish_and_wait(ConversationalStateChanged(old=cs, new=cs))
    assert len(seen) == 1


# --- mode change forces CS to IDLE first ---------------------------------


async def test_mode_change_forces_cs_to_idle_before_emitting_modechanged():
    bus = EventBus()
    events: list = []
    bus.subscribe(ConversationalStateChanged, lambda e: events.append(("cs", e)))
    bus.subscribe(ModeChanged, lambda e: events.append(("mode", e)))

    sm = StateMachine(bus=bus, initial_mode=Mode.ACTIVE)
    # Drive into a non-IDLE CS.
    sm.set_conversational_state(ConversationalState.LISTENING)
    sm.set_conversational_state(ConversationalState.THINKING)
    # Now flip the mode. CS must be forced to IDLE *first*.
    sm.set_mode(Mode.MUTED)

    # Allow scheduled tasks to run.
    import asyncio
    for _ in range(5):
        await asyncio.sleep(0)

    # Sequence: CS IDLE->LISTENING, CS LISTENING->THINKING, CS THINKING->IDLE,
    # then ModeChanged ACTIVE->MUTED.
    assert [tag for tag, _ in events] == ["cs", "cs", "cs", "mode"]
    assert events[2][1].new is ConversationalState.IDLE
    assert events[3][1].old is Mode.ACTIVE and events[3][1].new is Mode.MUTED
    # And the SM's settled state matches.
    assert sm.mode is Mode.MUTED
    assert sm.conversational_state is ConversationalState.IDLE


async def test_mode_change_when_cs_already_idle_emits_only_modechanged():
    bus = EventBus()
    events: list = []
    bus.subscribe(ConversationalStateChanged, lambda e: events.append(("cs", e)))
    bus.subscribe(ModeChanged, lambda e: events.append(("mode", e)))

    sm = StateMachine(bus=bus)
    sm.set_mode(Mode.SLEEPING)

    import asyncio
    for _ in range(5):
        await asyncio.sleep(0)

    assert [tag for tag, _ in events] == ["mode"]


# --- CS transitions while not ACTIVE -------------------------------------


def test_cs_to_non_idle_while_muted_raises():
    sm = StateMachine()
    sm.set_mode(Mode.MUTED)
    with pytest.raises(IllegalTransition) as excinfo:
        sm.set_conversational_state(ConversationalState.LISTENING)
    assert excinfo.value.axis == "conversational_state"


def test_cs_to_non_idle_while_sleeping_raises():
    sm = StateMachine()
    sm.set_mode(Mode.SLEEPING)
    with pytest.raises(IllegalTransition):
        sm.set_conversational_state(ConversationalState.LISTENING)


def test_cs_to_idle_while_muted_is_noop():
    sm = StateMachine()
    sm.set_mode(Mode.MUTED)
    # Already IDLE (forced by mode change). Setting IDLE again must not raise.
    sm.set_conversational_state(ConversationalState.IDLE)
    assert sm.conversational_state is ConversationalState.IDLE


# --- bus is optional ------------------------------------------------------


def test_no_bus_means_no_emission_and_no_loop_required():
    # No EventBus, no asyncio loop. Pure state transitions must work.
    sm = StateMachine()
    sm.set_conversational_state(ConversationalState.LISTENING)
    sm.set_conversational_state(ConversationalState.THINKING)
    sm.set_conversational_state(ConversationalState.SPEAKING)
    sm.set_conversational_state(ConversationalState.IDLE)
    sm.set_mode(Mode.MUTED)
    sm.set_mode(Mode.SLEEPING)
    sm.set_mode(Mode.ACTIVE)
    assert sm.mode is Mode.ACTIVE
    assert sm.conversational_state is ConversationalState.IDLE


# --- initial state guard --------------------------------------------------


@pytest.mark.parametrize("mode", [Mode.MUTED, Mode.SLEEPING])
def test_initial_non_active_with_non_idle_cs_rejected(mode: Mode):
    with pytest.raises(ValueError):
        StateMachine(
            initial_mode=mode,
            initial_conversational_state=ConversationalState.LISTENING,
        )


def test_default_initial_state_is_active_idle():
    sm = StateMachine()
    assert sm.mode is Mode.ACTIVE
    assert sm.conversational_state is ConversationalState.IDLE


# --- IllegalTransition surface --------------------------------------------


def test_illegal_transition_message_is_human_readable():
    sm = StateMachine()
    with pytest.raises(IllegalTransition) as excinfo:
        sm.set_conversational_state(ConversationalState.SPEAKING)
    msg = str(excinfo.value)
    assert "IDLE" in msg and "SPEAKING" in msg
    assert "conversational_state" in msg
