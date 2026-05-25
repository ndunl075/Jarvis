"""openWakeWord-based wake-word detector for the 'hey_jarvis' model.

Implements WakeWordDetector (audio/protocols.py) and Loadable (core/lifecycle.py).

Model files
-----------
The openWakeWord pretrained ONNX models (hey_jarvis_v0.1.onnx plus the
shared melspectrogram.onnx and embedding_model.onnx) are NOT included in
the openwakeword pip wheel; they are downloaded into the package's
`resources/models/` directory by `openwakeword.utils.download_models()`.

In production (post-Phase-7) the installer pre-places these files into the
same package directory at install time, so `Model(wakeword_models=["hey_jarvis"])`
finds them by name with no network access. The installer's PyInstaller spec
must include them; see BUILD.md Phase 7 Task 1.

For development before the installer exists, run once per venv:

    python -c "import openwakeword; openwakeword.utils.download_models()"

This wrapper does NOT auto-download at runtime. If files are missing, load()
raises WakeWordModelMissingError with the recovery instruction in the
message; the first-launch flow (Phase 7 Task 3) is the only place that
should ever trigger a download.

Sensitivity mapping
-------------------
WakeWordConfig.sensitivity is 0.0..1.0 with the SPEC intuition
"higher = more sensitive". openWakeWord returns per-prediction scores in
0.0..1.0. We map:

    threshold = 1.0 - sensitivity

so sensitivity=1.0 -> threshold=0.0 (fires on any non-zero score, very prone
to false positives); sensitivity=0.0 -> threshold=1.0 (effectively never
fires); sensitivity=0.5 -> threshold=0.5 (matches openWakeWord's commonly
recommended default). The flip exists specifically to preserve the
"higher = more" intuition exposed in the settings UI.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from jarvis.audio.protocols import (
    FRAME_BYTES,
    AudioFrame,
    WakeWordResult,
)

log = logging.getLogger(__name__)


class WakeWordModelMissingError(RuntimeError):  # noqa: N818
    """Raised when openWakeWord model files cannot be located on disk."""


def _resolve_model_path(model_name: str) -> Path:
    """Locate the ONNX file for `model_name` inside the *imported* openwakeword
    package's resources/models directory.

    Anchored to openwakeword.__file__ -- NOT importlib.util.find_spec -- so the
    lookup matches the actually-imported module instead of sys.path-walking
    and possibly hitting a stale system-Python install of the package."""
    import openwakeword  # the actually-imported one
    models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    # Models are versioned on disk (hey_jarvis_v0.1.onnx). Lex-sort picks the
    # highest version when multiple exist; falls back cleanly to a single
    # unversioned file if the layout ever changes.
    matches = sorted(models_dir.glob(f"{model_name}*.onnx"))
    if not matches:
        raise WakeWordModelMissingError(
            f"No ONNX model matching {model_name!r} in {models_dir}. "
            "If this is a development environment, run once: "
            'python -c "import openwakeword; openwakeword.utils.download_models()"'
        )
    return matches[-1]


class OpenWakeWord:
    name: str = "wake_word"

    def __init__(
        self,
        *,
        sensitivity: float = 0.5,
        model_name: str = "hey_jarvis",
    ) -> None:
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError(
                f"sensitivity must be in [0.0, 1.0], got {sensitivity}"
            )
        self.sensitivity = sensitivity
        self._model_name = model_name
        self._model = None
        # Set on load() — exposed so the composition root can print which
        # ONNX file was actually loaded (regression-detection diagnostic).
        self.model_path: Path | None = None
        self.is_loaded: bool = False
        # Raw score from the last feed() call. Updated every frame so
        # debug logging (pipeline.py) can report the score without
        # waiting for a threshold crossing.
        self.last_score: float = 0.0

    @property
    def threshold(self) -> float:
        """Detection-score threshold derived from sensitivity (see header)."""
        return 1.0 - self.sensitivity

    # -- Loadable --

    async def load(self) -> None:
        if self.is_loaded:
            return
        # Late import: openwakeword loads its ONNX runtime at import time;
        # keep that off the cold path of code that never calls load().
        try:
            from openwakeword.model import Model
        except ImportError as e:  # pragma: no cover - openwakeword is a hard dep
            raise WakeWordModelMissingError(
                "openwakeword is not installed"
            ) from e
        self.model_path = _resolve_model_path(self._model_name)
        log.info("[boot] openwakeword models from: %s", self.model_path)
        try:
            self._model = Model(
                wakeword_models=[str(self.model_path)],
                inference_framework="onnx",
            )
        except Exception as e:
            raise WakeWordModelMissingError(
                f"could not load openWakeWord model {self._model_name!r} "
                f"from {self.model_path}: {e}. "
                "If this is a development environment, run once: "
                'python -c "import openwakeword; openwakeword.utils.download_models()"'
            ) from e
        self.is_loaded = True

    async def unload(self) -> None:
        if not self.is_loaded:
            return
        self._model = None
        self.is_loaded = False

    # -- WakeWordDetector --

    async def feed(self, frame: AudioFrame) -> WakeWordResult | None:
        if self._model is None:
            return None
        if len(frame) != FRAME_BYTES:
            log.error(
                "wake_word: wrong frame size %d (expected %d)",
                len(frame), FRAME_BYTES,
            )
            return None
        # Convert int16 PCM bytes -> int16 numpy at the stage boundary, per
        # SPEC § Audio Pipeline ("resampling/conversion at the stage
        # boundary, not the pipeline").
        audio = np.frombuffer(frame, dtype=np.int16)
        try:
            scores = self._model.predict(audio)
        except Exception:
            log.exception("openwakeword.predict raised")
            return None
        # Some openwakeword versions can return (scores, debug); normalize.
        if isinstance(scores, tuple):
            scores = scores[0]
        # openWakeWord keys the result dict by the FULL model name including
        # the version suffix (e.g. "hey_jarvis_v0.1"), not the short name we
        # passed at construction. Looking up self._model_name silently misses
        # and returns 0.0 for every frame — a bug that hid 99% confidence
        # detections. Since we only ever load one wake word per instance, we
        # take the single value from the dict directly. Resilient to future
        # version-suffix changes upstream.
        if not scores:
            return None
        score = float(next(iter(scores.values())))
        self.last_score = score
        if score >= self.threshold:
            return WakeWordResult(confidence=score)
        return None

    def reset(self) -> None:
        if self._model is None:
            return
        # Clears openWakeWord's internal feature buffers so the next
        # detection requires a fresh utterance rather than firing on the
        # tail end of the one we just consumed.
        try:
            self._model.reset()
        except Exception:
            log.exception("openwakeword.reset raised")
