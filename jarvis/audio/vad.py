"""Silero-VAD wrapper, ONNX-only (no torch at runtime).

Implements VoiceActivityDetector (audio/protocols.py) and Loadable
(core/lifecycle.py).

Why ONNX-only and not silero_vad.load_silero_vad(onnx=True)
-----------------------------------------------------------
The silero-vad pip package's Python wrapper (`OnnxWrapper`) wraps audio
in `torch.Tensor` at the inference call site, so importing `silero_vad`
pulls torch into the runtime image even when the inference backend is
ONNX. Torch wheels are hundreds of MB and would dominate the PyInstaller
bundle in Phase 7 even though we never call any torch op.

This module instead loads silero-vad's bundled `silero_vad.onnx` file
directly via onnxruntime, bypassing the torch-wrapped Python API. We
locate the file using importlib.util.find_spec("silero_vad") which does
NOT execute the package's __init__.py (and therefore does not import
torch).

In production (post-Phase-7) the installer can either:
  (a) bundle silero_vad/data/silero_vad.onnx alongside our package and
      pass its path to the constructor, or
  (b) keep silero-vad as an install-time data-only dep, ship its ONNX
      file, and configure PyInstaller to exclude torch entirely.
Either way the runtime image avoids torch.

Threshold mapping
-----------------
silero-vad outputs per-window speech probability in [0.0, 1.0]. Default
speech-start threshold is 0.5: above is voice, below is silence. 0.5 is
silero-vad's documented recommended default for typical clean speech;
lower (e.g. 0.3) reacts faster but is more prone to false positives on
noise; higher (e.g. 0.7) is more conservative.

We expose `speech_threshold` directly on the constructor rather than
mapping a [0,1] sensitivity field because VAD sensitivity is not in the
SPEC config schema today. If it lands later, the wrapper can be wired
the same way wake_word does it (sensitivity = 1 - threshold).

Window math
-----------
Pipeline frames are 480 samples (30 ms at 16 kHz). The silero-VAD ONNX
model strictly requires 512-sample windows (32 ms). We buffer; most
feed() calls produce exactly one inference, the very first feed produces
zero. The 32 ms window resolution is also what we tick the silence-since-
speech counter by, so endpoint timing is granular to ~32 ms.
"""

from __future__ import annotations

import logging
import sysconfig
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.audio.protocols import (
    FRAME_BYTES,
    SAMPLE_RATE,
    SPEECH_START_WINDOWS_DEFAULT,
    VAD_ENDPOINT_MS_DEFAULT,
    AudioFrame,
    VADEvent,
)

log = logging.getLogger(__name__)

# Silero-VAD ONNX expects exactly this many samples per inference.
_VAD_WINDOW_SAMPLES: int = 512
_VAD_WINDOW_BYTES: int = _VAD_WINDOW_SAMPLES * 2  # int16
_VAD_WINDOW_MS: int = _VAD_WINDOW_SAMPLES * 1000 // SAMPLE_RATE  # 32

# Silero LSTM state shape.
_VAD_STATE_SHAPE: tuple[int, int, int] = (2, 1, 128)


class VADModelMissingError(RuntimeError):  # noqa: N818
    """Raised when the silero-vad ONNX file cannot be located."""


class _Internal(Enum):
    SILENCE = "silence"
    SPEECH = "speech"


def _find_silero_onnx() -> Path | None:
    """Locate silero_vad/data/silero_vad.onnx WITHOUT importing the
    silero_vad package (which would trigger torch import).

    Anchored to the active interpreter's site-packages via sysconfig
    rather than importlib.util.find_spec, which walks sys.path and can
    return a stale system-Python install when the venv is unusual.
    sysconfig is bound to the running interpreter, so it always points
    at the venv that actually executed `import` at runtime."""
    purelib = Path(sysconfig.get_paths()["purelib"])
    onnx = purelib / "silero_vad" / "data" / "silero_vad.onnx"
    return onnx if onnx.exists() else None


class SileroVAD:
    name: str = "vad"

    def __init__(
        self,
        *,
        speech_threshold: float = 0.5,
        endpoint_ms: int = VAD_ENDPOINT_MS_DEFAULT,
        speech_start_windows: int = SPEECH_START_WINDOWS_DEFAULT,
        model_path: Path | None = None,
        on_probability: Callable[[float], None] | None = None,
    ) -> None:
        if not 0.0 <= speech_threshold <= 1.0:
            raise ValueError(
                f"speech_threshold must be in [0.0, 1.0], got {speech_threshold}"
            )
        if endpoint_ms <= 0:
            raise ValueError(f"endpoint_ms must be positive, got {endpoint_ms}")
        if speech_start_windows < 1:
            raise ValueError(
                f"speech_start_windows must be >= 1, got {speech_start_windows}"
            )
        self.speech_threshold = speech_threshold
        self.endpoint_ms = endpoint_ms
        self.speech_start_windows = speech_start_windows
        self._model_path = model_path
        # Set on load() — exposed so the composition root can print which
        # ONNX file was actually loaded (regression-detection diagnostic).
        self.model_path: Path | None = None
        # Optional per-window probability hook for diagnostics / tuning.
        # Called from the loop thread (VAD runs in the pipeline frame loop).
        # Wrapped in try/except so a misbehaving callback doesn't break VAD.
        self._on_probability = on_probability
        self._session: Any = None
        self._state: np.ndarray | None = None
        self._sr_input: np.ndarray = np.array(SAMPLE_RATE, dtype=np.int64)
        self._buffer = bytearray()
        self._internal: _Internal = _Internal.SILENCE
        self._silence_ms: int = 0
        # Counter for sustained-speech detection. Reset by silence in the
        # SILENCE state, by reset(), or when SPEECH_STARTED fires.
        self._consecutive_speech_windows: int = 0
        self.is_loaded: bool = False

    # -- Loadable --

    async def load(self) -> None:
        if self.is_loaded:
            return
        # Late import: onnxruntime initializes execution providers at
        # import time on some platforms.
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - hard dep
            raise VADModelMissingError("onnxruntime is not installed") from e

        path = self._model_path or _find_silero_onnx()
        if path is None or not Path(path).exists():
            raise VADModelMissingError(
                "silero-vad ONNX file not found. Expected via the silero-vad "
                "pip package at silero_vad/data/silero_vad.onnx, or via an "
                "explicit model_path constructor argument."
            )
        self.model_path = Path(path)
        log.info("[boot] silero-vad onnx from: %s", self.model_path)
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
                sess_options=opts,
            )
        except Exception as e:
            raise VADModelMissingError(
                f"could not load silero-vad ONNX from {path}: {e}"
            ) from e
        self._state = np.zeros(_VAD_STATE_SHAPE, dtype=np.float32)
        self.reset()
        self.is_loaded = True

    async def unload(self) -> None:
        if not self.is_loaded:
            return
        self._session = None
        self._state = None
        self._buffer.clear()
        self.is_loaded = False

    # -- VoiceActivityDetector --

    async def feed(self, frame: AudioFrame) -> VADEvent | None:
        if self._session is None:
            return None
        if len(frame) != FRAME_BYTES:
            log.error(
                "vad: wrong frame size %d (expected %d)", len(frame), FRAME_BYTES
            )
            return None
        self._buffer.extend(frame)

        events: list[VADEvent] = []
        while len(self._buffer) >= _VAD_WINDOW_BYTES:
            chunk_bytes = bytes(self._buffer[:_VAD_WINDOW_BYTES])
            del self._buffer[:_VAD_WINDOW_BYTES]
            # Convert int16 PCM bytes -> float32 normalized at the stage
            # boundary, per SPEC § Audio Pipeline.
            audio = (
                np.frombuffer(chunk_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            ).reshape(1, _VAD_WINDOW_SAMPLES)
            try:
                prob_arr, new_state = self._session.run(
                    None,
                    {"input": audio, "sr": self._sr_input, "state": self._state},
                )
            except Exception:
                log.exception("silero-vad inference raised")
                continue
            self._state = new_state
            prob = float(np.asarray(prob_arr).squeeze())
            if self._on_probability is not None:
                try:
                    self._on_probability(prob)
                except Exception:
                    log.exception("on_probability callback raised")
            evt = self._step(prob)
            if evt is not None:
                events.append(evt)

        # If both fired in one feed, ENDPOINT is the more significant
        # transition for the pipeline (state-out vs. state-in).
        if VADEvent.ENDPOINT in events:
            return VADEvent.ENDPOINT
        if VADEvent.SPEECH_STARTED in events:
            return VADEvent.SPEECH_STARTED
        return None

    def reset(self) -> None:
        # Clears all per-LISTENING-session state. Called by the pipeline on
        # entry to LISTENING and on barge-in. Critical: a stale
        # _Internal.SPEECH must NOT survive into a new session, otherwise
        # the very first window's silence would start the endpoint timer
        # for a session in which no speech has yet occurred. The
        # consecutive-speech counter must also be cleared so a partial
        # streak from a prior session can't combine with new windows to
        # spuriously fire SPEECH_STARTED.
        if self._state is not None:
            self._state.fill(0)
        self._buffer.clear()
        self._internal = _Internal.SILENCE
        self._silence_ms = 0
        self._consecutive_speech_windows = 0

    # -- internal --

    def _step(self, prob: float) -> VADEvent | None:
        if self._internal is _Internal.SILENCE:
            # Require N consecutive windows above threshold before firing
            # SPEECH_STARTED. A single high-prob blip followed by silence
            # must NOT arm the endpoint timer.
            if prob >= self.speech_threshold:
                self._consecutive_speech_windows += 1
                if self._consecutive_speech_windows >= self.speech_start_windows:
                    self._internal = _Internal.SPEECH
                    self._silence_ms = 0
                    self._consecutive_speech_windows = 0
                    return VADEvent.SPEECH_STARTED
                return None
            # Silence breaks the streak.
            self._consecutive_speech_windows = 0
            return None
        # In SPEECH:
        if prob >= self.speech_threshold:
            self._silence_ms = 0
            return None
        self._silence_ms += _VAD_WINDOW_MS
        if self._silence_ms >= self.endpoint_ms:
            self._internal = _Internal.SILENCE
            self._silence_ms = 0
            return VADEvent.ENDPOINT
        return None
