"""Tests for jarvis.ui.overlay.

The Qt event loop is NOT running in these tests — same constraint as
test_tray.py. Anything that depends on QMetaObject.invokeMethod queued
dispatch is exercised by calling the slot directly. The qapp fixture
gives us the offscreen QApplication QWidget construction requires."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.core.events import ConversationalStateChanged, EventBus
from jarvis.core.state_machine import ConversationalState, StateMachine
from jarvis.ui.overlay import (
    PALETTE_CYAN,
    AmplitudeLatch,
    OverlayOrb,
    _BREATH_HZ,
    _EMA_KEEP,
    _EMA_NEW,
    _FRAME_INTERVAL_MS,
    _LERP_FACTOR,
    _RING_TILT_AMPS,
    _RING_TILT_FREQS,
    _RING_TILT_OFFSETS,
    _RING_Y_SCALES,
    _STATE_TARGETS,
    make_amplitude_callback,
)


# --- AmplitudeLatch (no Qt needed) -------------------------------------


def test_amplitude_latch_starts_at_zero():
    latch = AmplitudeLatch()
    assert latch.latest() == 0.0


def test_amplitude_latch_writes_overwrite_previous():
    latch = AmplitudeLatch()
    latch(0.3)
    latch(0.7)
    latch(0.5)
    # Latch keeps the LATEST sample (not the max). Frames want freshness,
    # not history; older samples are useless once overtaken.
    assert latch.latest() == 0.5


def test_amplitude_latch_clamps_garbage_input():
    """The TTS callback is the writer. Defensive clamp guards the Qt
    reader against a misbehaving voice emitting out-of-range values."""
    latch = AmplitudeLatch()
    latch(-0.5)
    assert latch.latest() == 0.0
    latch(2.5)
    assert latch.latest() == 1.0


def test_amplitude_latch_reset_returns_to_zero():
    latch = AmplitudeLatch()
    latch(0.8)
    latch.reset()
    assert latch.latest() == 0.0


def test_amplitude_latch_is_callable_for_tts_wiring():
    """PiperTTS takes on_amplitude as Callable[[float], None]; the latch
    is a single object that satisfies both that contract and the Qt
    reader's needs, no wrapper required."""
    latch = AmplitudeLatch()
    latch(0.42)  # callable form (audio thread)
    assert latch.latest() == 0.42  # property form (Qt thread)


def test_make_amplitude_callback_returns_same_latch_object():
    """The helper returns (latch, callback) for explicit wiring in the
    composition root. Both are the same object since AmplitudeLatch
    satisfies the callback interface itself."""
    latch, cb = make_amplitude_callback()
    assert cb is latch
    cb(0.5)
    assert latch.latest() == 0.5


# --- constants tripwires -----------------------------------------------


def test_frame_interval_targets_30fps():
    """33 ms / frame ≈ 30 fps. Tripwire if someone bumps it without
    rethinking the GPU/CPU cost note in the module docstring."""
    assert 30 <= _FRAME_INTERVAL_MS <= 50


def test_ema_coefficients_sum_to_one():
    """KEEP + NEW = 1.0 by convention so the EMA stays bounded in [0, 1]
    for any input in [0, 1]."""
    assert abs((_EMA_KEEP + _EMA_NEW) - 1.0) < 1e-9


def test_lerp_factor_is_documented_value():
    assert _LERP_FACTOR == pytest.approx(0.08)


# --- palette tripwires ------------------------------------------------


def test_palette_cyan_has_required_colors():
    """PALETTE_CYAN must export all three palette colors from the design spec."""
    from PySide6.QtGui import QColor

    assert "core" in PALETTE_CYAN
    assert "primary" in PALETTE_CYAN
    assert "secondary" in PALETTE_CYAN
    assert PALETTE_CYAN["primary"] == QColor(0x38, 0xF4, 0xFF)
    assert PALETTE_CYAN["secondary"] == QColor(0x19, 0xA8, 0xFF)


# --- state target tripwires -------------------------------------------


def test_state_targets_cover_active_states():
    """Each non-IDLE ConversationalState must have a _StateTarget with all
    8 documented fields."""
    active = (
        ConversationalState.LISTENING,
        ConversationalState.THINKING,
        ConversationalState.SPEAKING,
    )
    fields = (
        "energy", "rotation_speed", "particle_speed", "shell_radius",
        "ring_spread", "filament_opacity", "core_scale", "bloom",
    )
    for state in active:
        assert state in _STATE_TARGETS, f"{state.name} missing from _STATE_TARGETS"
        t = _STATE_TARGETS[state]
        for field in fields:
            assert hasattr(t, field), f"{state.name} missing field {field!r}"


def test_state_target_rotation_speed_overrides():
    """LISTENING and THINKING rings are stationary; SPEAKING rings rotate slowly."""
    assert _STATE_TARGETS[ConversationalState.LISTENING].rotation_speed == 0.0
    assert _STATE_TARGETS[ConversationalState.THINKING].rotation_speed == 0.0
    assert _STATE_TARGETS[ConversationalState.SPEAKING].rotation_speed == pytest.approx(0.4)


def test_ring_y_scales_are_organic():
    """Updated Y-scales give each ring a different tilt angle so they look
    like independent orbital planes rather than concentric flat circles."""
    assert len(_RING_Y_SCALES) == 3
    assert _RING_Y_SCALES == pytest.approx((0.60, 0.40, 0.55))


def test_ring_tilt_wobble_constants_are_distinct():
    """All three rings must have different tilt parameters so their orbital
    planes precess independently."""
    assert len(_RING_TILT_FREQS) == 3
    assert len(_RING_TILT_AMPS) == 3
    assert len(_RING_TILT_OFFSETS) == 3
    # Frequencies must all differ so rings never lock in tilt phase.
    assert len(set(_RING_TILT_FREQS)) == 3


def test_ring_tilt_varies_over_time():
    """The effective Y-scale for each ring must change meaningfully across a
    simulated run, confirming precession is active."""
    import math

    def tilt_at(ring_idx: int, phase: float) -> float:
        base = _RING_Y_SCALES[ring_idx]
        amp  = _RING_TILT_AMPS[ring_idx]
        freq = _RING_TILT_FREQS[ring_idx]
        off  = _RING_TILT_OFFSETS[ring_idx]
        return base + amp * math.sin(phase * freq + off)

    for ring in range(3):
        values = [tilt_at(ring, p * 0.5) for p in range(200)]
        spread = max(values) - min(values)
        assert spread > 0.10, f"ring {ring} tilt barely moves: spread={spread:.3f}"

    # Rings must differ from each other at the same phase so they look
    # independent, not like copies of one another.
    for phase in (0.0, 1.5, 3.7, 6.0):
        t0 = tilt_at(0, phase)
        t1 = tilt_at(1, phase)
        t2 = tilt_at(2, phase)
        assert t0 != pytest.approx(t1, abs=0.02), f"ring 0/1 same tilt at phase {phase}"
        assert t0 != pytest.approx(t2, abs=0.02), f"ring 0/2 same tilt at phase {phase}"


def test_breath_hz_covers_active_states():
    """Every non-IDLE state must have a breathing-pulse frequency defined."""
    for state in (
        ConversationalState.LISTENING,
        ConversationalState.THINKING,
        ConversationalState.SPEAKING,
    ):
        assert state in _BREATH_HZ, f"{state.name} missing from _BREATH_HZ"
    # Frequencies increase from LISTENING → SPEAKING (more urgency).
    assert _BREATH_HZ[ConversationalState.LISTENING] < _BREATH_HZ[ConversationalState.THINKING]
    assert _BREATH_HZ[ConversationalState.THINKING] < _BREATH_HZ[ConversationalState.SPEAKING]


# --- OverlayOrb construction + lifecycle ------------------------------


def _make_orb(qapp, sm: StateMachine | None = None):
    sm = sm or StateMachine()
    bus = EventBus()
    latch = AmplitudeLatch()
    orb = OverlayOrb(sm=sm, bus=bus, amplitude_latch=latch)
    return orb, sm, bus, latch


def test_construction_hides_orb_when_initial_state_is_idle(qapp):
    """SM defaults to (ACTIVE, IDLE); orb should not be visible at
    construction."""
    orb, *_ = _make_orb(qapp)
    try:
        assert orb.isVisible() is False
    finally:
        orb.close()


def test_construction_starts_no_timer_when_idle(qapp):
    orb, *_ = _make_orb(qapp)
    try:
        assert orb._timer.isActive() is False
    finally:
        orb.close()


def test_widget_is_click_through_and_frameless(qapp):
    """The orb is decoration — clicks must pass through, no focus steal."""
    from PySide6.QtCore import Qt

    orb, *_ = _make_orb(qapp)
    try:
        assert orb.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        assert orb.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        assert orb.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        flags = orb.windowFlags()
        assert bool(flags & Qt.WindowType.FramelessWindowHint)
        assert bool(flags & Qt.WindowType.WindowStaysOnTopHint)
    finally:
        orb.close()


# --- state transitions -------------------------------------------------


def test_state_transition_to_listening_shows_orb_and_starts_timer(qapp):
    orb, *_ = _make_orb(qapp)
    try:
        orb._on_state_name("LISTENING")
        assert orb._state is ConversationalState.LISTENING
        assert orb._timer.isActive() is True
    finally:
        orb.close()


def test_state_transition_to_idle_hides_and_stops_timer(qapp):
    orb, *_ = _make_orb(qapp)
    try:
        orb._on_state_name("LISTENING")
        assert orb._timer.isActive() is True
        orb._on_state_name("IDLE")
        assert orb._timer.isActive() is False
        assert orb.isVisible() is False
    finally:
        orb.close()


def test_idle_transition_resets_displayed_amplitude(qapp):
    orb, _, _, latch = _make_orb(qapp)
    try:
        latch(0.9)
        orb._on_state_name("SPEAKING")
        orb._tick()
        assert orb._displayed_amplitude > 0
        orb._on_state_name("IDLE")
        assert orb._displayed_amplitude == 0.0
        assert latch.latest() == 0.0
    finally:
        orb.close()


def test_unknown_state_name_logs_and_ignores(qapp, caplog):
    import logging

    orb, *_ = _make_orb(qapp)
    try:
        with caplog.at_level(logging.WARNING, logger="jarvis.ui.overlay"):
            orb._on_state_name("BOGUS")
        assert any("unknown state name" in r.message for r in caplog.records)
        assert orb._state is ConversationalState.IDLE
    finally:
        orb.close()


# --- per-tick EMA + phase ---------------------------------------------


def test_tick_advances_phase(qapp):
    orb, *_ = _make_orb(qapp)
    try:
        orb._on_state_name("LISTENING")
        phase_before = orb._phase
        orb._tick()
        assert orb._phase > phase_before
    finally:
        orb.close()


def test_tick_smooths_amplitude_with_ema_during_speaking(qapp):
    """Single tick from displayed=0 toward latched=1.0 yields EMA_NEW."""
    orb, _, _, latch = _make_orb(qapp)
    try:
        orb._on_state_name("SPEAKING")
        latch(1.0)
        orb._displayed_amplitude = 0.0
        orb._tick()
        assert orb._displayed_amplitude == pytest.approx(_EMA_NEW, abs=1e-6)
    finally:
        orb.close()


def test_tick_decays_residual_amplitude_when_not_speaking(qapp):
    """SPEAKING → LISTENING shouldn't keep the orb mid-bulge. The
    displayed value decays by KEEP each non-SPEAKING tick."""
    orb, *_ = _make_orb(qapp)
    try:
        orb._on_state_name("LISTENING")
        orb._displayed_amplitude = 0.5
        orb._tick()
        assert orb._displayed_amplitude == pytest.approx(0.5 * _EMA_KEEP, abs=1e-6)
    finally:
        orb.close()


def test_tick_after_close_is_noop(qapp):
    orb, *_ = _make_orb(qapp)
    orb._on_state_name("LISTENING")
    orb.close()
    phase_at_close = orb._phase
    orb._tick()
    assert orb._phase == phase_at_close


# --- lerp convergence -------------------------------------------------


def test_lerp_converges_filament_opacity_toward_listening_target(qapp):
    """After enough ticks in LISTENING, non-oscillating render state fields
    should converge to their targets. core_scale oscillates due to the
    breathing pulse so we check it stays within the oscillation band instead."""
    orb, *_ = _make_orb(qapp)
    try:
        orb._on_state_name("LISTENING")
        target = _STATE_TARGETS[ConversationalState.LISTENING]
        breath_amp = 0.20 * target.core_scale  # max deviation from base
        for _ in range(200):  # ~6.6 s of simulated ticks
            orb._tick()
        # These fields don't oscillate — they should converge to the target.
        assert orb._filament_opacity == pytest.approx(target.filament_opacity, abs=0.01)
        assert orb._energy == pytest.approx(target.energy, abs=0.01)
        # core_scale oscillates around target.core_scale at the breath rate.
        lo = target.core_scale - breath_amp - 0.02
        hi = target.core_scale + breath_amp + 0.02
        assert lo <= orb._core_scale <= hi, (
            f"core_scale {orb._core_scale:.4f} outside breathing band [{lo:.4f}, {hi:.4f}]"
        )
    finally:
        orb.close()


def test_speaking_rotates_rings(qapp):
    """In SPEAKING state the ring phases should advance each tick once
    the rotation_speed has lerped away from zero."""
    orb, *_ = _make_orb(qapp)
    try:
        orb._on_state_name("SPEAKING")
        # Force render_rotation_speed to the SPEAKING target so we see
        # movement immediately without waiting for full lerp convergence.
        orb._render_rotation_speed = _STATE_TARGETS[ConversationalState.SPEAKING].rotation_speed
        phases_before = list(orb._ring_phases)
        orb._tick()
        assert orb._ring_phases[0] != phases_before[0]
    finally:
        orb.close()


def test_listening_rings_do_not_rotate(qapp):
    """In LISTENING state with rotation_speed=0 the ring phases must stay
    constant."""
    orb, *_ = _make_orb(qapp)
    try:
        orb._on_state_name("LISTENING")
        orb._render_rotation_speed = 0.0
        phases_before = list(orb._ring_phases)
        orb._tick()
        assert orb._ring_phases == phases_before
    finally:
        orb.close()


# --- multi-frequency ring rotation ------------------------------------


def test_ring_effective_speeds_vary_across_phases():
    """The superposition of three sine waves must produce speed variation
    across a sample of phases.  At base_speed=0.4 the result should range
    from approximately 0.15 to 0.65."""
    from jarvis.ui.overlay import _ring_effective_speeds

    base = 0.4
    import math
    samples = [
        _ring_effective_speeds(phase, base)[0]
        for phase in (i * 0.5 for i in range(100))
    ]
    lo, hi = min(samples), max(samples)
    # Must genuinely vary (not flat).
    assert hi - lo > 0.10, f"speed range too narrow: [{lo:.3f}, {hi:.3f}]"
    # Must stay in a reasonable envelope around the base.
    assert lo > 0.10, f"speed dipped too low: {lo:.3f}"
    assert hi < 0.70, f"speed went too high: {hi:.3f}"


def test_ring_effective_speeds_rings_differ_at_same_phase():
    """Ring 0 and ring 2 must have different effective speeds at the same
    phase so they drift apart rather than rotating in lockstep."""
    from jarvis.ui.overlay import _ring_effective_speeds

    # Test at several phases to rule out coincidental equality.
    for phase in (0.0, 1.0, 2.5, 4.7):
        s0, _s1, s2 = _ring_effective_speeds(phase, 0.4)
        assert s0 != pytest.approx(s2, abs=1e-6), (
            f"ring 0 and ring 2 have identical speed {s0:.6f} at phase {phase}"
        )


# --- shutdown discipline (mirrors tray) --------------------------------


def test_close_unsubscribes_from_bus(qapp):
    orb, *_ = _make_orb(qapp)
    calls: list[int] = []
    original = orb._unsubscribe

    def spy() -> None:
        calls.append(1)
        original()

    orb._unsubscribe = spy
    orb.close()
    assert calls == [1]
    assert orb._alive is False


def test_close_is_idempotent(qapp):
    orb, *_ = _make_orb(qapp)
    orb.close()
    orb.close()
    assert orb._alive is False


def test_close_stops_the_animation_timer(qapp):
    orb, *_ = _make_orb(qapp)
    orb._on_state_name("LISTENING")
    assert orb._timer.isActive() is True
    orb.close()
    assert orb._timer.isActive() is False


def test_late_state_event_after_close_does_not_apply(qapp):
    """The queued slot guard mirrors tray._on_mode_name's pattern."""
    orb, *_ = _make_orb(qapp)
    orb.close()
    orb._on_state_name("LISTENING")
    assert orb._timer.isActive() is False
    assert orb._state is ConversationalState.IDLE


# --- multi-monitor reposition ------------------------------------------


def test_screen_changed_repositions_orb(qapp):
    orb, *_ = _make_orb(qapp)
    try:
        with patch.object(orb, "_position_on_primary_screen") as repos:
            orb._on_screen_changed()
        repos.assert_called_once()
    finally:
        orb.close()


def test_geometry_changed_repositions_without_rebinding_signals(qapp):
    orb, *_ = _make_orb(qapp)
    try:
        with patch.object(orb, "_position_on_primary_screen") as repos:
            orb._on_geometry_changed()
        repos.assert_called_once()
    finally:
        orb.close()


def test_screen_changed_post_close_is_noop(qapp):
    orb, *_ = _make_orb(qapp)
    orb.close()
    with patch.object(orb, "_position_on_primary_screen") as repos:
        orb._on_screen_changed()
    repos.assert_not_called()
