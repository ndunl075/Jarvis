"""faster-whisper STT wrapper.

Implements SpeechToText (audio/protocols.py) and Loadable (core/lifecycle.py).

Threading
---------
faster_whisper.WhisperModel.transcribe() is synchronous and CPU-bound.
A few-second utterance typically takes 200 ms - 2 s of CPU on a modern
desktop with the int8 base.en model. Calling it directly from the asyncio
loop would block the frame loop -- which freezes barge-in detection and
audio capture during exactly the moment the user might want to interrupt.

This wrapper runs every transcribe() call inside asyncio.to_thread so the
loop stays free. The frame loop continues to process VAD frames
concurrently with whisper inference. CRITICAL: do NOT short-circuit this
for "small" inputs. Even a 0.5 s clip can take 200 ms+; that's the same
order as our 30 ms frame cadence, so blocking would visibly chunk audio.

Model construction also goes through asyncio.to_thread, because the
WhisperModel(...) constructor downloads model files from HuggingFace on
first run -- a multi-second-to-multi-minute operation depending on
network speed and model size.

Model files
-----------
faster-whisper downloads quantized models from HuggingFace Hub on first
use, caching them at ~/.cache/huggingface/hub/ by default. The base.en
int8 model is ~150 MB.

In production (post-Phase-7) the recommended approach is to pre-place
the model files into a known directory and pass `download_root` so the
runtime never needs network. Bundling ~150 MB inside the installer is
acceptable; the alternative (download-on-first-launch with a progress UI
in the first-launch wizard) is documented in BUILD.md Phase 7 Task 3 but
not chosen yet -- defer to Phase 7.

For development before the installer exists, the model downloads
transparently on the first transcribe() call.

Empty / silence input
---------------------
Empty bytes return "" without invoking whisper.

Real silence may produce empty text from whisper, OR may produce a
hallucination (whisper has known issues like "Thanks for watching!" on
near-silence content). The wrapper does NOT currently filter; the
pipeline relies on the empty-string path for IDLE return, but spoken
hallucinations would fall through. faster-whisper has a
`vad_filter=True` parameter that runs silero-vad internally to suppress
these; we don't enable it because we already run silero-vad in the
pipeline and double-VAD adds latency. If hallucinations prove painful
in real use, revisit (likely by gating transcribe() on a minimum VAD
voice-confidence threshold from the listen buffer).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class STTLoadError(RuntimeError):  # noqa: N818
    """Raised when the whisper model cannot be loaded."""


def _resolve_model_id(model_size: str, language: str) -> str:
    """Map (model_size, language) to a faster-whisper model identifier.

    English-specific variants exist for tiny/base/small/medium and are
    slightly faster + more accurate for English-only use, so we prefer
    them when language is 'en' and the size has an English variant."""
    if language == "en" and model_size in ("tiny", "base", "small", "medium"):
        return f"{model_size}.en"
    return model_size


class FasterWhisperSTT:
    name: str = "stt"

    def __init__(
        self,
        *,
        model_size: str = "base",
        language: str = "en",
        compute_type: str = "int8",
        download_root: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.compute_type = compute_type
        self._download_root = download_root
        self._device = device
        self._model = None
        self.is_loaded: bool = False

    # -- Loadable --

    async def load(self) -> None:
        if self.is_loaded:
            return
        # Late import: faster-whisper loads ctranslate2 at import time;
        # keep that off any code path that doesn't actually need STT.
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover - hard dep
            raise STTLoadError("faster_whisper is not installed") from e

        model_id = _resolve_model_id(self.model_size, self.language)
        try:
            # Construction can download large model files on first use;
            # run in a thread so the loop stays responsive throughout.
            kwargs: dict = {
                "device": self._device,
                "compute_type": self.compute_type,
            }
            if self._download_root is not None:
                kwargs["download_root"] = str(self._download_root)
                kwargs["local_files_only"] = True
            self._model = await asyncio.to_thread(
                WhisperModel,
                model_id,
                **kwargs,
            )
        except Exception as e:
            raise STTLoadError(
                f"could not load whisper model {model_id!r}: {e}"
            ) from e
        self.is_loaded = True

    async def unload(self) -> None:
        if not self.is_loaded:
            return
        self._model = None
        self.is_loaded = False

    # -- SpeechToText --

    async def transcribe(self, audio: bytes) -> str:
        if self._model is None:
            log.warning("transcribe() called before load(); returning empty")
            return ""
        if not audio:
            return ""
        # Convert int16 PCM bytes -> float32 normalized at the stage
        # boundary, per SPEC § Audio Pipeline.
        audio_np = (
            np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        )
        try:
            text = await asyncio.to_thread(self._sync_transcribe, audio_np)
        except Exception:
            log.exception("whisper transcribe raised")
            return ""
        # faster-whisper sometimes emits leading/trailing whitespace;
        # strip so the pipeline's empty-text check works correctly.
        return text.strip()

    # -- internal --

    def _sync_transcribe(self, audio: np.ndarray) -> str:
        """Run faster-whisper inference synchronously. Always called inside
        asyncio.to_thread; never on the loop thread."""
        assert self._model is not None
        # vad_filter=True runs faster-whisper's internal silero-VAD over the
        # input audio to suppress non-speech regions before decoding. Adds
        # ~50 ms of CPU per transcription but dramatically reduces
        # whisper's known hallucination behavior on short / low-signal
        # clips ("Thanks for watching!", "..."). Worth it for the quality
        # win even though the pipeline already runs VAD upstream -- they
        # operate on different signal shapes (live frames vs. captured
        # utterance).
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        # Materialize the segment generator INSIDE this thread; iterating
        # the generator IS the inference work. Returning an unconsumed
        # generator across the to_thread boundary would defer that work
        # back onto the loop thread, defeating the whole point.
        return "".join(seg.text for seg in segments)
