"""Siri-style overlay orb. Phase 5, [ARCH] + [IMPL].

Frameless, transparent, always-on-top, click-through. Subscribes to
ConversationalStateChanged only — never reads Mode. When Mode is not
ACTIVE, ConversationalState is forced to IDLE by the state machine and
the orb hides on its own, so a single subscription is sufficient.

Animation states (drive paintEvent):

  IDLE       — hidden (window not visible)
  LISTENING  — rings stationary, gentle pulse; cyan palette
  THINKING   — rings stationary, slightly more energy
  SPEAKING   — rings rotate slowly, amplitude-reactive core pulse

Each tick lerps the current render state toward the active state's
_StateTarget at _LERP_FACTOR per frame. Ring rotation phases advance each
tick by the current (lerped) rotation_speed, giving smooth transitions
when switching between stationary and rotating states.

Render layers (all use CompositionMode_Plus for additive blending):
  1. Outer halo   — large QRadialGradient, primary → secondary, provides glow
  2. Orbital rings — three dashed ellipses with fake-3D Y-scale tilt
  3. Particles    — 36 fixed-seed radial gradients with per-particle twinkle
  4. Core         — bright filled circle, near-white center fading to cyan

Animation cadence: a single QTimer at ~33 ms (~30 fps).

Audio → Qt amplitude bridge
---------------------------
PiperTTS exposes an on_amplitude(float) callback fired ~30 Hz from its
synthesis thread. We cannot touch a QWidget from that thread, so the
overlay sets `self._amplitude_latch` (an AmplitudeLatch) as that
callback. The latch is a single shared float: TTS writes (audio thread),
the QTimer tick reads (Qt thread). Python attribute writes on a float
are atomic under the GIL, so no lock is needed for this single-writer
single-reader pattern.

The displayed value is an exponential moving average of the latch:
    displayed = EMA_KEEP * displayed + EMA_NEW * latched

State changes
-------------
Bus subscriber runs on the audio loop's thread. It marshals to the Qt
main thread via QMetaObject.invokeMethod + queued connection, same
pattern tray.py uses. State name crosses as a str (already a registered
Qt metatype).

Shutdown discipline
-------------------
Mirrors TrayIcon exactly:
  - _alive flag, flipped False in close()
  - close() unsubscribes from the bus
  - close() stops the QTimer
  - queued slot guards against late-arriving calls post-close
  - close() is idempotent

Multi-monitor
-------------
We listen to QGuiApplication.screenAdded / screenRemoved and to the
primary screen's geometryChanged signal to reposition on
resolution/dock/undock events.

Known limitation
----------------
A visible white bounding rectangle may appear around the orb on some
Windows configurations due to the Qt DWM compositor not honoring
WA_TranslucentBackground for child widgets. Deferred to v1.1 — fix path
is QWebEngineView with an HTML/CSS orb (Chromium handles per-pixel alpha
correctly), accepting the ~150 MB installer cost.
"""

from __future__ import annotations

import logging
import math
import os
import random as _random
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import (
    Q_ARG,
    QMetaObject,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QPainter,
    QPaintEvent,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from jarvis.core.events import ConversationalStateChanged, EventBus
from jarvis.core.state_machine import ConversationalState, StateMachine

log = logging.getLogger(__name__)

# Render cadence. 30 fps reads as fluid for a single-orb animation;
# anything higher is wasted GPU/CPU for no visible gain.
_FRAME_INTERVAL_MS: int = 33

# EMA smoothing for SPEAKING amplitude. Lower KEEP = snappier orb pulses.
_EMA_KEEP: float = 0.45
_EMA_NEW: float = 0.55

# How much smoothed amplitude drives each render channel during SPEAKING.
_SPEAK_AMP_ENERGY: float = 0.85
_SPEAK_AMP_CORE: float = 0.45
_SPEAK_AMP_BLOOM: float = 0.40
_SPEAK_AMP_SHELL: float = 0.15
_SPEAK_AMP_FILAMENT: float = 0.30
_SPEAK_AMP_PARTICLES: float = 0.50

# Widget window size. The orb fills most of this area; padding absorbs
# halo glow and particle scatter near the edges.
_WIDGET_DIAMETER: int = 320

# Position: distance from the bottom of the primary screen's work area
# to the orb's bottom edge. Centered horizontally. Bumped from 64→100
# to keep the larger orb clear of the taskbar.
_BOTTOM_MARGIN_PX: int = 100

# --------------------------------------------------------------------------
# Palette — PALETTES.cyan from cyber1443 / jarvis-ai-orb-web-animation
# --------------------------------------------------------------------------

_CORE_COLOR = QColor(0xEA, 0xFF, 0xFF)   # #eaffff — near-white
_PRIMARY     = QColor(0x38, 0xF4, 0xFF)  # #38f4ff — cyan
_SECONDARY   = QColor(0x19, 0xA8, 0xFF)  # #19a8ff — deeper blue

PALETTE_CYAN: dict[str, QColor] = {
    "core":      _CORE_COLOR,
    "primary":   _PRIMARY,
    "secondary": _SECONDARY,
}

# --------------------------------------------------------------------------
# Lerp factor and per-state render targets
# --------------------------------------------------------------------------

_LERP_FACTOR: float = 0.08


@dataclass(frozen=True)
class _StateTarget:
    """8-tuple of lerp-able render parameters for one ConversationalState."""

    energy: float          # drives core brightness
    rotation_speed: float  # ring phase advance per second (rad/s)
    particle_speed: float  # twinkle frequency multiplier
    shell_radius: float    # halo outer radius as fraction of widget_radius
    ring_spread: float     # ring radius scale (1.0 = nominal)
    filament_opacity: float  # master opacity for rings and particles
    core_scale: float      # core radius as fraction of nominal
    bloom: float           # halo brightness boost


_STATE_TARGETS: dict[ConversationalState, _StateTarget] = {
    ConversationalState.LISTENING: _StateTarget(
        energy=0.35,
        rotation_speed=0.0,    # stationary rings
        particle_speed=0.5,
        shell_radius=0.70,
        ring_spread=1.0,
        filament_opacity=0.55,
        core_scale=0.85,
        bloom=0.50,
    ),
    ConversationalState.THINKING: _StateTarget(
        energy=0.50,
        rotation_speed=0.0,    # stationary rings
        particle_speed=0.8,
        shell_radius=0.75,
        ring_spread=1.0,
        filament_opacity=0.65,
        core_scale=0.90,
        bloom=0.60,
    ),
    ConversationalState.SPEAKING: _StateTarget(
        energy=0.70,
        rotation_speed=0.4,    # slow rotation — not too fast
        particle_speed=1.2,
        shell_radius=0.85,
        ring_spread=1.0,
        filament_opacity=0.75,
        core_scale=1.00,
        bloom=0.80,
    ),
}

# --------------------------------------------------------------------------
# Ring geometry — fake-3D tilt via Y-axis compression
# --------------------------------------------------------------------------

# Semi-major axes as fractions of widget_radius
_RING_RADII: tuple[float, ...] = (0.60, 0.75, 0.90)
# Y-axis scale (< 1 = compressed, looks tilted in 3D). Base values;
# the actual tilt wobbles continuously — see _RING_TILT_*.
_RING_Y_SCALES: tuple[float, ...] = (0.60, 0.40, 0.55)

# Tilt precession: each ring's Y-scale oscillates slowly so the orbital
# plane appears to wobble in 3D space.  Frequencies are incommensurable
# with the rotation frequencies so tilt and spin never lock in phase.
_RING_TILT_FREQS:   tuple[float, ...] = (0.13, 0.09, 0.17)  # effective rad/s
_RING_TILT_AMPS:    tuple[float, ...] = (0.08, 0.12, 0.09)  # wobble amplitude
_RING_TILT_OFFSETS: tuple[float, ...] = (0.0,  2.1,  4.3)   # prevent sync

# --------------------------------------------------------------------------
# Particle precomputation — fixed seed keeps positions stable across frames
# --------------------------------------------------------------------------

_N_PARTICLES: int = 36


def _generate_particles(n: int, seed: int = 42) -> list[tuple[float, float, float]]:
    """Return list of (angle_rad, radius_fraction, a_seed) for n particles."""
    rng = _random.Random(seed)
    return [
        (
            rng.uniform(0.0, 2.0 * math.pi),  # polar angle
            rng.uniform(0.30, 0.92),           # distance from center
            rng.random(),                       # per-particle twinkle offset
        )
        for _ in range(n)
    ]


_PARTICLES: list[tuple[float, float, float]] = _generate_particles(_N_PARTICLES)

# Breathing-pulse frequency per state (cycles per second = Hz).
# The core radius oscillates at this rate to keep the orb feeling alive
# even when no amplitude signal is present.
_BREATH_HZ: dict[ConversationalState, float] = {
    ConversationalState.LISTENING: 1.0,
    ConversationalState.THINKING:  1.5,
    ConversationalState.SPEAKING:  2.5,
}


def _react_amplitude(smoothed: float) -> float:
    """Shape EMA-smoothed [0,1] amplitude for visible orb variation."""
    if smoothed <= 0.0:
        return 0.0
    return min(1.0, smoothed ** 0.82 * 1.15)

# --------------------------------------------------------------------------
# Render helpers
# --------------------------------------------------------------------------


def _ring_effective_speeds(
    phase: float, base_speed: float
) -> tuple[float, float, float]:
    """Return per-ring effective rotation speeds using a multi-frequency
    superposition.  Three sine waves at incommensurable frequencies with
    staggered phase offsets keep the motion aperiodic across any reasonable
    run time.  Each ring gets a slightly different formula so they never move
    identically.

    Ring 0: standard superposition at phase.
    Ring 1: slower variation — phase scaled by 0.85.
    Ring 2: faster variation — phase scaled by 1.15, different offsets.
    """
    ph0 = phase
    ph1 = phase * 0.85
    ph2 = phase * 1.15

    mod0 = (
        0.12 * math.sin(ph0 * 0.31)
        + 0.08 * math.sin(ph0 * 0.79 + 1.7)
        + 0.05 * math.sin(ph0 * 1.43 + 3.1)
    )
    mod1 = (
        0.12 * math.sin(ph1 * 0.31)
        + 0.08 * math.sin(ph1 * 0.79 + 1.7)
        + 0.05 * math.sin(ph1 * 1.43 + 3.1)
    )
    mod2 = (
        0.12 * math.sin(ph2 * 0.31)
        + 0.08 * math.sin(ph2 * 0.79 + 2.3)
        + 0.05 * math.sin(ph2 * 1.43 + 0.6)
    )

    # Scale modulation by the lerped base_speed so rings stay still in
    # LISTENING/THINKING (where base_speed → 0).
    scale = base_speed / 0.4  # 1.0 at the SPEAKING target of 0.4
    s0 = base_speed + scale * mod0
    s1 = base_speed + scale * mod1
    s2 = base_speed + scale * mod2
    return (max(0.0, s0), max(0.0, s1), max(0.0, s2))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _qcolor_alpha(base: QColor, alpha_f: float) -> QColor:
    """Return a copy of *base* with alpha set to *alpha_f* (0.0–1.0)."""
    c = QColor(base)
    c.setAlphaF(max(0.0, min(1.0, alpha_f)))
    return c


# --------------------------------------------------------------------------
# AmplitudeLatch
# --------------------------------------------------------------------------


class AmplitudeLatch:
    """Single-float shared between the TTS audio thread (writer) and the
    Qt main thread (reader). No lock — Python attribute writes on a
    primitive float are atomic under the GIL, and we only care about
    the latest value per Qt frame, so a missed write is harmless.

    Exposed as a __call__ so it can be passed directly as PiperTTS's
    `on_amplitude` callback without an extra wrapper."""

    __slots__ = ("_value",)

    def __init__(self) -> None:
        self._value: float = 0.0

    def __call__(self, amplitude: float) -> None:
        # TTS thread write. Clamp here so the Qt-side reader never sees
        # garbage from a misbehaving voice.
        if amplitude < 0.0:
            amplitude = 0.0
        elif amplitude > 1.0:
            amplitude = 1.0
        self._value = amplitude

    def latest(self) -> float:
        # Qt-thread read.
        return self._value

    def reset(self) -> None:
        self._value = 0.0


# --------------------------------------------------------------------------
# OverlayOrb
# --------------------------------------------------------------------------


class OverlayOrb(QWidget):
    """Bottom-center always-on-top frameless orb reflecting the
    StateMachine's ConversationalState. Constructed AFTER QApplication."""

    def __init__(
        self,
        *,
        sm: StateMachine,
        bus: EventBus,
        amplitude_latch: AmplitudeLatch,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sm = sm
        self._bus = bus
        self._amplitude_latch = amplitude_latch
        self._alive: bool = True

        # Animation phase — advances monotonically, wraps at 2π.
        self._phase: float = 0.0
        self._displayed_amplitude: float = 0.0
        self._state: ConversationalState = sm.conversational_state

        # Lerp-able render state. Initialised near-zero/IDLE; ramps up
        # toward the active state's _StateTarget on each _tick().
        self._energy: float = 0.0
        self._render_rotation_speed: float = 0.0
        self._particle_speed: float = 0.5
        self._shell_radius: float = 0.6
        self._ring_spread: float = 1.0
        self._filament_opacity: float = 0.0
        self._core_scale: float = 0.8
        self._bloom: float = 0.5
        # Ring rotation phases, evenly spaced so all three are visible.
        self._ring_phases: list[float] = [
            0.0,
            2.0 * math.pi / 3.0,
            4.0 * math.pi / 3.0,
        ]

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        # WA_TranslucentBackground lets paintEvent's per-pixel alpha reach
        # the compositor. Box artifact is a known Windows DWM limitation —
        # deferred to v1.1.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Clicks pass through to whatever window is beneath.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.setFixedSize(_WIDGET_DIAMETER, _WIDGET_DIAMETER)
        self._position_on_primary_screen()

        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

        self._unsubscribe = bus.subscribe(
            ConversationalStateChanged, self._on_state_event
        )

        # React to screen layout changes. Skipped under the offscreen Qt
        # platform (test environment): binding to offscreen screen signals
        # corrupts the heap on the next widget construction (Windows
        # 0xc0000374).
        self._screen_signals_bound = (
            os.environ.get("QT_QPA_PLATFORM") != "offscreen"
        )
        if self._screen_signals_bound:
            self._connect_screen_signals()
            guiapp = QGuiApplication.instance()
            screen_added = getattr(guiapp, "screenAdded", None)
            screen_removed = getattr(guiapp, "screenRemoved", None)
            if screen_added is not None:
                screen_added.connect(self._on_screen_changed)
            if screen_removed is not None:
                screen_removed.connect(self._on_screen_changed)

        self._apply_state(self._state)

    # -- bus subscriber (audio thread) ------------------------------------

    def _on_state_event(self, event: ConversationalStateChanged) -> None:
        """Runs on the audio loop's thread. Marshal to Qt via the same
        QMetaObject.invokeMethod / str-metatype pattern tray.py uses."""
        if not self._alive:
            return
        QMetaObject.invokeMethod(
            self,
            "_on_state_name",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, event.new.name),
        )

    @Slot(str)
    def _on_state_name(self, name: str) -> None:
        """Runs on the Qt main thread. Guarded against late-arriving
        queued calls post-close()."""
        if not self._alive:
            return
        try:
            state = ConversationalState[name]
        except KeyError:
            log.warning("overlay received unknown state name: %r", name)
            return
        self._apply_state(state)

    # -- state transitions ------------------------------------------------

    def _apply_state(self, state: ConversationalState) -> None:
        self._state = state
        if state is ConversationalState.IDLE:
            self._timer.stop()
            self._amplitude_latch.reset()
            self._displayed_amplitude = 0.0
            self.hide()
        else:
            if not self._timer.isActive():
                self._timer.start()
            self.show()
            self.raise_()

    # -- per-frame tick ---------------------------------------------------

    @Slot()
    def _tick(self) -> None:
        """30 fps animation driver. Advances phase, lerps render state,
        advances ring phases, triggers repaint."""
        if not self._alive:
            return

        # Phase wraps at 2π so sin/cos stay in stable float range.
        self._phase += 2.0 * math.pi * (_FRAME_INTERVAL_MS / 1000.0)

        target = _STATE_TARGETS.get(self._state)
        if target is not None:
            t_energy = target.energy

            # Breathing pulse: core_scale oscillates at a state-specific Hz
            # so the orb feels alive even without amplitude input.
            breath_hz = _BREATH_HZ.get(self._state, 1.0)
            t_core_scale = target.core_scale * (
                1.0 + 0.25 * math.sin(self._phase * breath_hz)
            )
            t_bloom = target.bloom
            t_shell = target.shell_radius
            t_filament = target.filament_opacity
            t_particle = target.particle_speed

            if self._state is ConversationalState.SPEAKING:
                amp = self._amplitude_latch.latest()
                self._displayed_amplitude = (
                    _EMA_KEEP * self._displayed_amplitude + _EMA_NEW * amp
                )
                react = _react_amplitude(self._displayed_amplitude)
                t_energy = target.energy + react * _SPEAK_AMP_ENERGY
                t_core_scale = t_core_scale + react * _SPEAK_AMP_CORE
                t_bloom = target.bloom + react * _SPEAK_AMP_BLOOM
                t_shell = target.shell_radius + react * _SPEAK_AMP_SHELL
                t_filament = target.filament_opacity + react * _SPEAK_AMP_FILAMENT
                t_particle = target.particle_speed + react * _SPEAK_AMP_PARTICLES
            else:
                # Decay residual amplitude so SPEAKING→anything transitions
                # start from near-zero rather than mid-bulge.
                self._displayed_amplitude *= _EMA_KEEP

            lf = _LERP_FACTOR
            self._energy               = _lerp(self._energy,               t_energy,                lf)
            self._render_rotation_speed = _lerp(self._render_rotation_speed, target.rotation_speed,  lf)
            self._particle_speed        = _lerp(self._particle_speed,        t_particle,              lf)
            self._shell_radius          = _lerp(self._shell_radius,          t_shell,                 lf)
            self._ring_spread           = _lerp(self._ring_spread,           target.ring_spread,      lf)
            self._filament_opacity      = _lerp(self._filament_opacity,      t_filament,              lf)
            self._core_scale            = _lerp(self._core_scale,            t_core_scale,            lf)
            self._bloom                 = _lerp(self._bloom,                 t_bloom,                 lf)

        # Advance ring rotation phases with a multi-frequency superposition.
        # Three sine waves at incommensurable frequencies with staggered phase
        # offsets prevent wave alignment at zero, producing aperiodic motion
        # that feels organic rather than mechanical.
        dt = _FRAME_INTERVAL_MS / 1000.0
        if self._render_rotation_speed > 0.001:
            ph = self._phase
            speeds = _ring_effective_speeds(ph, self._render_rotation_speed)
        else:
            speeds = (0.0, 0.0, 0.0)
        for i in range(3):
            self._ring_phases[i] = (
                self._ring_phases[i] + speeds[i] * dt
            ) % (2.0 * math.pi)

        self.update()

    # -- rendering --------------------------------------------------------

    def paintEvent(self, _event: QPaintEvent) -> None:
        if not self._alive or self._state is ConversationalState.IDLE:
            return
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            w = self.width()
            h = self.height()
            center = QPointF(w / 2.0, h / 2.0)
            widget_radius = min(w, h) / 2.0
            self._paint_halo(p, center, widget_radius)
            self._paint_rings(p, center, widget_radius)
            self._paint_particles(p, center, widget_radius)
            self._paint_core(p, center, widget_radius)
        finally:
            p.end()

    def _paint_halo(
        self, p: QPainter, center: QPointF, widget_radius: float
    ) -> None:
        """Layer 1 — large bloom glow that extends nearly to the widget edge."""
        halo_radius = self._shell_radius * widget_radius
        bloom = self._bloom
        grad = QRadialGradient(center, halo_radius)
        grad.setColorAt(0.0, _qcolor_alpha(_PRIMARY,   0.30 * bloom))
        grad.setColorAt(0.4, _qcolor_alpha(_SECONDARY, 0.25 * bloom))
        grad.setColorAt(1.0, _qcolor_alpha(_SECONDARY, 0.0))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawEllipse(center, halo_radius, halo_radius)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

    def _paint_rings(
        self, p: QPainter, center: QPointF, widget_radius: float
    ) -> None:
        """Layer 2 — three faint solid orbital ellipses, background traces."""
        if self._filament_opacity < 0.01:
            return
        # Rings are at half the opacity of before — they should feel like
        # background structure, not the dominant visual element.
        # Minimum 0.5 alpha so rings always read as clearly cyan even when
        # filament_opacity is still ramping up from the lerp start.
        ring_alpha = max(0.5, self._filament_opacity) * 0.55
        for i, (r_frac, base_y) in enumerate(zip(_RING_RADII, _RING_Y_SCALES)):
            ring_r = r_frac * widget_radius * self._ring_spread
            phase = self._ring_phases[i]
            # Tilt precession: Y-scale wobbles around its base value so the
            # orbital plane appears to tip and rock in 3D space.
            wobble = _RING_TILT_AMPS[i] * math.sin(
                self._phase * _RING_TILT_FREQS[i] + _RING_TILT_OFFSETS[i]
            )
            y_scale = max(0.05, base_y + wobble)
            pen = QPen(_qcolor_alpha(_PRIMARY, ring_alpha))
            pen.setWidthF(1.5)
            pen.setStyle(Qt.PenStyle.SolidLine)
            p.save()
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.translate(center.x(), center.y())
            p.rotate(math.degrees(phase))
            p.drawEllipse(
                QRectF(
                    -ring_r,
                    -ring_r * y_scale,
                    ring_r * 2.0,
                    ring_r * y_scale * 2.0,
                )
            )
            p.restore()

    def _paint_particles(
        self, p: QPainter, center: QPointF, widget_radius: float
    ) -> None:
        """Layer 3 — 36 twinkling dots at fixed polar positions."""
        if self._filament_opacity < 0.01:
            return
        p.setPen(Qt.PenStyle.NoPen)
        particle_r = 2.5  # slightly larger for visibility
        for angle, r_frac, a_seed in _PARTICLES:
            twinkle = 0.5 + 0.5 * math.sin(
                self._phase * (1.0 + a_seed * 2.0) * self._particle_speed
                + a_seed * 12.7
            )
            alpha_f = twinkle * self._filament_opacity * 0.75
            if alpha_f < 0.02:
                continue
            r = r_frac * widget_radius
            px = center.x() + r * math.cos(angle)
            py = center.y() + r * math.sin(angle)
            pt = QPointF(px, py)
            grad = QRadialGradient(pt, particle_r)
            grad.setColorAt(0.0, _qcolor_alpha(_PRIMARY, alpha_f))
            grad.setColorAt(1.0, _qcolor_alpha(_PRIMARY, 0.0))
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            p.setBrush(grad)
            p.drawEllipse(pt, particle_r, particle_r)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

    def _paint_core(
        self, p: QPainter, center: QPointF, widget_radius: float
    ) -> None:
        """Layer 4 — dominant glowing core, near-white centre fading to cyan."""
        energy_boost = 1.0 + 0.40 * self._energy
        # Base multiplier 0.35 gives ~42 px at widget_radius=120 (LISTENING);
        # minimum guard ensures it's always visible even from zero init.
        core_r = max(
            0.25 * widget_radius,
            self._core_scale * widget_radius * 0.35 * energy_boost,
        )
        grad = QRadialGradient(center, core_r)
        # White centre → cyan → transparent gives "glowing energy source" depth.
        grad.setColorAt(0.0,  _qcolor_alpha(QColor(0xFF, 0xFF, 0xFF), 1.0))
        grad.setColorAt(0.2,  _qcolor_alpha(_PRIMARY, 1.0))
        grad.setColorAt(0.5,  _qcolor_alpha(_PRIMARY, 0.7))
        grad.setColorAt(1.0,  _qcolor_alpha(_SECONDARY, 0.0))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawEllipse(center, core_r, core_r)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

    # -- screen positioning -----------------------------------------------

    def _connect_screen_signals(self) -> None:
        """Bind to the primary screen's geometryChanged. Re-bound on
        screen add/remove because the primary itself can change."""
        new_screen = self.screen() or QGuiApplication.primaryScreen()
        old_screen = getattr(self, "_connected_screen", None)
        if old_screen is not None and old_screen is not new_screen:
            for sig_name in ("geometryChanged", "availableGeometryChanged"):
                sig = getattr(old_screen, sig_name, None)
                if sig is None:
                    continue
                try:
                    sig.disconnect(self._on_geometry_changed)
                except (RuntimeError, TypeError):
                    pass
        if new_screen is None:
            self._connected_screen = None
            return
        if new_screen is not old_screen:
            new_screen.geometryChanged.connect(self._on_geometry_changed)
            new_screen.availableGeometryChanged.connect(
                self._on_geometry_changed
            )
        self._connected_screen = new_screen

    def _on_screen_changed(self, *_args: object) -> None:
        if not self._alive:
            return
        self._connect_screen_signals()
        self._position_on_primary_screen()

    def _on_geometry_changed(self, *_args: object) -> None:
        if not self._alive:
            return
        self._position_on_primary_screen()

    def _position_on_primary_screen(self) -> None:
        """Center horizontally, sit _BOTTOM_MARGIN_PX above the work-area
        bottom edge (the taskbar boundary on Windows)."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - _WIDGET_DIAMETER) // 2
        y = geo.y() + geo.height() - _WIDGET_DIAMETER - _BOTTOM_MARGIN_PX
        self.move(x, y)

    # -- shutdown ---------------------------------------------------------

    def close(self) -> bool:
        """Unsubscribe, stop the timer, disconnect Qt signals, hide the
        widget. Idempotent and safe under concurrent late events from
        the bus."""
        if not self._alive:
            return super().close()
        self._alive = False
        try:
            self._unsubscribe()
        except Exception:
            log.exception("overlay unsubscribe failed")
        try:
            self._timer.stop()
        except Exception:
            log.exception("overlay timer stop failed")
        if self._screen_signals_bound:
            self._disconnect_screen_signals()
        self.hide()
        return super().close()

    def _disconnect_screen_signals(self) -> None:
        guiapp = QGuiApplication.instance()
        for sig_name in ("screenAdded", "screenRemoved"):
            sig = getattr(guiapp, sig_name, None)
            if sig is None:
                continue
            try:
                sig.disconnect(self._on_screen_changed)
            except (RuntimeError, TypeError):
                pass
        screen = getattr(self, "_connected_screen", None)
        if screen is None:
            return
        for sig_name in ("geometryChanged", "availableGeometryChanged"):
            sig = getattr(screen, sig_name, None)
            if sig is None:
                continue
            try:
                sig.disconnect(self._on_geometry_changed)
            except (RuntimeError, TypeError):
                pass
        self._connected_screen = None


def make_amplitude_callback() -> tuple[AmplitudeLatch, Callable[[float], None]]:
    """Composition-root helper: returns (latch, callback). Pass the
    callback into PiperTTS(on_amplitude=...); pass the latch into
    OverlayOrb(amplitude_latch=...). Two refs so the composition root
    can demonstrate the wiring is intentional — neither side knows
    about the other."""
    latch = AmplitudeLatch()
    return latch, latch
