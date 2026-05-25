"""Tests for jarvis.audio.vad.SileroVAD.

Wrapper-logic tests stub onnxruntime's InferenceSession with a scriptable
fake that returns a sequence of probabilities, so we can exhaustively
verify the speech/silence state machine, endpoint timing, and reset
behavior without loading the real ONNX model.

Two fixture-based tests run against the real silero-vad ONNX (located via
the silero-vad pip package) when WAV fixtures are committed. They auto-
skip when fixtures or the ONNX file are absent.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.audio.protocols import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    VAD_ENDPOINT_MS_DEFAULT,
    VADEvent,
    VoiceActivityDetector,
)
from jarvis.audio.vad import SileroVAD, VADModelMissingError
from jarvis.core.lifecycle import Loadable

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "audio"
SPEECH_FIXTURE = FIXTURES_DIR / "speech.wav"
SILENCE_FIXTURE = FIXTURES_DIR / "silence.wav"


# Each pipeline frame = 30 ms = 480 samples = 960 bytes.
# Silero window = 32 ms = 512 samples = 1024 bytes.
# Buffer math: feed 1 -> 480 samples (no inference), feed 2+ -> at least
# one inference per call. Feed 17 across the start fills enough that
# the 1-feed-1-inference cadence holds afterwards.

SILENT_FRAME = b"\x00" * FRAME_BYTES


# --- protocol smoke ------------------------------------------------------


def test_implements_protocols():
    v = SileroVAD()
    assert isinstance(v, VoiceActivityDetector)
    assert isinstance(v, Loadable)


# --- constructor validation ----------------------------------------------


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        SileroVAD(speech_threshold=1.5)
    with pytest.raises(ValueError):
        SileroVAD(speech_threshold=-0.1)


def test_invalid_endpoint_ms_rejected():
    with pytest.raises(ValueError):
        SileroVAD(endpoint_ms=0)
    with pytest.raises(ValueError):
        SileroVAD(endpoint_ms=-100)


def test_default_endpoint_uses_protocol_constant():
    v = SileroVAD()
    assert v.endpoint_ms == VAD_ENDPOINT_MS_DEFAULT
    assert v.endpoint_ms == 700


# --- mock session helpers -----------------------------------------------


def _scripted_session(probs):
    """Build a mock InferenceSession whose .run() returns the next prob
    from the iterable, replaying the last value once exhausted (so tests
    that overshoot the script keep working)."""
    session = MagicMock()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    iterator = iter(list(probs))
    last = [0.0]

    def run(_output_names, _inputs):
        try:
            last[0] = next(iterator)
        except StopIteration:
            pass
        return (
            np.array([[last[0]]], dtype=np.float32),
            state,
        )

    session.run.side_effect = run
    return session


@pytest.fixture
def patch_ort():
    """Patch onnxruntime.InferenceSession; tests assign per-test sessions
    to its return_value."""
    with patch("onnxruntime.InferenceSession") as cls:
        yield cls


async def _loaded(
    patch_ort, probs, *,
    speech_threshold=0.5,
    endpoint_ms=VAD_ENDPOINT_MS_DEFAULT,
    speech_start_windows=1,  # default 1 here so single-window scripts
                             # in existing tests still drive SPEECH_STARTED
                             # on the first matching window
):
    patch_ort.return_value = _scripted_session(probs)
    v = SileroVAD(
        speech_threshold=speech_threshold,
        endpoint_ms=endpoint_ms,
        speech_start_windows=speech_start_windows,
        model_path=Path(__file__),  # any existing path; load() doesn't read it (mock)
    )
    await v.load()
    return v


# --- load / unload ------------------------------------------------------


async def test_load_with_explicit_model_path_succeeds(patch_ort):
    patch_ort.return_value = _scripted_session([0.0])
    v = SileroVAD(model_path=Path(__file__))
    await v.load()
    assert v.is_loaded
    patch_ort.assert_called_once()


async def test_load_idempotent(patch_ort):
    patch_ort.return_value = _scripted_session([0.0])
    v = SileroVAD(model_path=Path(__file__))
    await v.load()
    await v.load()
    assert patch_ort.call_count == 1


async def test_unload_releases_session(patch_ort):
    patch_ort.return_value = _scripted_session([0.0])
    v = SileroVAD(model_path=Path(__file__))
    await v.load()
    await v.unload()
    assert not v.is_loaded
    assert v._session is None  # type: ignore[attr-defined]


async def test_load_missing_file_raises():
    v = SileroVAD(model_path=Path("/nope/does/not/exist.onnx"))
    with pytest.raises(VADModelMissingError):
        await v.load()


async def test_load_session_construction_failure_raises(patch_ort):
    patch_ort.side_effect = OSError("corrupt model")
    v = SileroVAD(model_path=Path(__file__))
    with pytest.raises(VADModelMissingError, match="could not load"):
        await v.load()


# --- state machine: SPEECH_STARTED on rising edge ----------------------


async def test_speech_started_fires_on_first_window_above_threshold(patch_ort):
    # First inference fires after enough buffered audio (>= 512 samples).
    # We script: window 1 = 0.0 (silence), window 2 = 0.9 (speech).
    v = await _loaded(patch_ort, probs=[0.0, 0.9])
    # Feed 1: 480 samples buffered, no inference.
    assert await v.feed(SILENT_FRAME) is None
    # Feed 2: 960 samples buffered -> 1 inference (window 1: silence). No event.
    assert await v.feed(SILENT_FRAME) is None
    # Subsequent feeds drain at ~1 inference each. Find SPEECH_STARTED.
    event = None
    for _ in range(5):
        ev = await v.feed(SILENT_FRAME)
        if ev is not None:
            event = ev
            break
    assert event is VADEvent.SPEECH_STARTED


async def test_no_speech_started_when_below_threshold(patch_ort):
    v = await _loaded(patch_ort, probs=[0.4, 0.4, 0.4, 0.4, 0.4],
                      speech_threshold=0.5)
    for _ in range(6):
        assert await v.feed(SILENT_FRAME) is None


async def test_threshold_boundary_inclusive(patch_ort):
    # prob == threshold should fire.
    v = await _loaded(patch_ort, probs=[0.5], speech_threshold=0.5)
    seen = None
    for _ in range(3):
        ev = await v.feed(SILENT_FRAME)
        if ev is not None:
            seen = ev
            break
    assert seen is VADEvent.SPEECH_STARTED


# --- sustained-speech requirement (Bug B) ------------------------------


def test_default_speech_start_windows_uses_protocol_constant():
    from jarvis.audio.protocols import SPEECH_START_WINDOWS_DEFAULT
    v = SileroVAD()
    assert v.speech_start_windows == SPEECH_START_WINDOWS_DEFAULT
    assert v.speech_start_windows == 3


def test_invalid_speech_start_windows_rejected():
    with pytest.raises(ValueError):
        SileroVAD(speech_start_windows=0)
    with pytest.raises(ValueError):
        SileroVAD(speech_start_windows=-1)


async def test_single_high_window_does_not_fire_with_default(patch_ort):
    """Default speech_start_windows=3: a single high blip MUST NOT fire."""
    v = await _loaded(
        patch_ort,
        probs=[0.9, 0.0, 0.0, 0.0, 0.0],
        speech_start_windows=3,
    )
    events = []
    for _ in range(6):
        events.append(await v.feed(SILENT_FRAME))
    assert all(e is None for e in events), (
        "single-window blip wrongly produced an event"
    )


async def test_three_consecutive_high_windows_fire_speech_started(patch_ort):
    v = await _loaded(
        patch_ort,
        probs=[0.9, 0.9, 0.9, 0.0, 0.0],
        speech_start_windows=3,
    )
    events = []
    for _ in range(6):
        events.append(await v.feed(SILENT_FRAME))
    started = [e for e in events if e is VADEvent.SPEECH_STARTED]
    assert len(started) == 1


async def test_silence_breaks_speech_streak(patch_ort):
    """Two highs, then silence, then two more highs: the streak resets,
    so SPEECH_STARTED never fires (need 3 consecutive)."""
    v = await _loaded(
        patch_ort,
        probs=[0.9, 0.9, 0.0, 0.9, 0.9, 0.0, 0.0],
        speech_start_windows=3,
    )
    events = []
    for _ in range(8):
        events.append(await v.feed(SILENT_FRAME))
    assert all(e is None for e in events)


# --- state machine: ENDPOINT after trailing silence --------------------


async def test_endpoint_fires_after_endpoint_ms_of_silence(patch_ort):
    # Speech in window 1, then long silence. With endpoint_ms=64ms and
    # window=32ms, 2 silent windows after speech should trigger ENDPOINT.
    probs = [0.9] + [0.0] * 30
    v = await _loaded(patch_ort, probs=probs, endpoint_ms=64)

    # Drive feeds and collect events.
    events: list[VADEvent | None] = []
    for _ in range(40):
        events.append(await v.feed(SILENT_FRAME))
    started = [i for i, e in enumerate(events) if e is VADEvent.SPEECH_STARTED]
    ended = [i for i, e in enumerate(events) if e is VADEvent.ENDPOINT]
    assert len(started) == 1
    assert len(ended) == 1
    assert ended[0] > started[0]


async def test_continuing_speech_resets_silence_count(patch_ort):
    # Pattern: speech, brief silence, speech, brief silence, then long
    # silence that finally crosses endpoint_ms.
    # endpoint_ms=96 (3 windows of silence required to fire).
    probs = [0.9, 0.0, 0.9, 0.0, 0.9] + [0.0] * 30
    v = await _loaded(patch_ort, probs=probs, endpoint_ms=96)

    events = []
    for _ in range(40):
        events.append(await v.feed(SILENT_FRAME))
    started = [i for i, e in enumerate(events) if e is VADEvent.SPEECH_STARTED]
    ended = [i for i, e in enumerate(events) if e is VADEvent.ENDPOINT]
    # Only one SPEECH_STARTED (the rising edge); only one ENDPOINT (after
    # the sustained trailing silence).
    assert len(started) == 1
    assert len(ended) == 1


async def test_endpoint_does_not_fire_before_threshold_reached(patch_ort):
    # Silence-only: no speech ever, so no ENDPOINT either.
    v = await _loaded(patch_ort, probs=[0.0] * 50, endpoint_ms=32)
    events = []
    for _ in range(40):
        events.append(await v.feed(SILENT_FRAME))
    assert all(e is None for e in events)


async def test_state_returns_to_silence_after_endpoint(patch_ort):
    # After ENDPOINT, the next speech must fire a NEW SPEECH_STARTED.
    probs = [0.9, 0.0, 0.0, 0.9, 0.9, 0.9]
    v = await _loaded(patch_ort, probs=probs, endpoint_ms=32)

    events = []
    for _ in range(15):
        events.append(await v.feed(SILENT_FRAME))
    started = [e for e in events if e is VADEvent.SPEECH_STARTED]
    ended = [e for e in events if e is VADEvent.ENDPOINT]
    assert len(started) == 2
    assert len(ended) == 1


# --- reset clears all state --------------------------------------------


async def test_reset_clears_in_speech_flag(patch_ort):
    """The critical correctness invariant: reset between LISTENING sessions
    must NOT leave the wrapper in _Internal.SPEECH. Otherwise the very
    first silent window of the next session would start the endpoint
    counter for a session in which no speech has occurred."""
    probs = [0.9, 0.0, 0.0, 0.0, 0.0]
    v = await _loaded(patch_ort, probs=probs, endpoint_ms=10_000)
    # Drive into SPEECH state.
    for _ in range(3):
        await v.feed(SILENT_FRAME)
    assert v._internal.value == "speech"  # type: ignore[attr-defined]
    v.reset()
    assert v._internal.value == "silence"  # type: ignore[attr-defined]
    assert v._silence_ms == 0  # type: ignore[attr-defined]
    assert len(v._buffer) == 0  # type: ignore[attr-defined]


async def test_reset_zeros_lstm_state(patch_ort):
    v = await _loaded(patch_ort, probs=[0.9, 0.9])
    for _ in range(3):
        await v.feed(SILENT_FRAME)
    v.reset()
    assert np.all(v._state == 0)  # type: ignore[attr-defined]


async def test_reset_after_speech_does_not_orphan_endpoint_timer(patch_ort):
    """Tripwire: if reset() forgot to clear _internal, the next session's
    silence frames would tick the endpoint counter and eventually fire
    ENDPOINT without any SPEECH_STARTED ever happening."""
    probs = [0.9] + [0.0] * 50
    v = await _loaded(patch_ort, probs=probs, endpoint_ms=32)

    # Session 1: drive into speech.
    for _ in range(3):
        await v.feed(SILENT_FRAME)
    v.reset()

    # Session 2: feed only silence (script produces 0.0). If reset failed,
    # ENDPOINT would fire after ~32 ms of silence even though no speech
    # occurred.
    for _ in range(20):
        ev = await v.feed(SILENT_FRAME)
        assert ev is None, "reset failed to clear in-speech flag"


def test_reset_before_load_is_noop():
    SileroVAD().reset()


# --- frame conversion / size ------------------------------------------


async def test_feed_converts_int16_bytes_to_float32_normalized(patch_ort):
    captured: list[np.ndarray] = []

    def run(_output_names, inputs):
        captured.append(inputs["input"].copy())
        return (np.array([[0.0]], dtype=np.float32),
                np.zeros((2, 1, 128), dtype=np.float32))

    session = MagicMock()
    session.run.side_effect = run
    patch_ort.return_value = session

    v = SileroVAD(model_path=Path(__file__))
    await v.load()
    pattern = np.array([16384] * FRAME_SAMPLES, dtype=np.int16)
    frame = pattern.tobytes()
    # Two feeds to trigger one inference.
    await v.feed(frame)
    await v.feed(frame)
    assert len(captured) >= 1
    audio = captured[0]
    assert audio.dtype == np.float32
    assert audio.shape == (1, 512)
    # 16384 / 32768 = 0.5
    assert audio[0, 0] == pytest.approx(0.5, abs=1e-4)


async def test_feed_wrong_size_returns_none_and_logs(
    patch_ort, caplog: pytest.LogCaptureFixture
):
    v = await _loaded(patch_ort, probs=[0.0])
    with caplog.at_level(logging.ERROR, logger="jarvis.audio.vad"):
        result = await v.feed(b"\x00" * 100)
    assert result is None
    assert any("wrong frame size" in r.message for r in caplog.records)


async def test_feed_isolates_inference_exception(
    patch_ort, caplog: pytest.LogCaptureFixture
):
    session = MagicMock()
    session.run.side_effect = RuntimeError("model crashed")
    patch_ort.return_value = session
    v = SileroVAD(model_path=Path(__file__))
    await v.load()
    with caplog.at_level(logging.ERROR, logger="jarvis.audio.vad"):
        for _ in range(3):
            assert await v.feed(SILENT_FRAME) is None
    assert any("silero-vad inference raised" in r.message for r in caplog.records)


async def test_feed_before_load_returns_none():
    v = SileroVAD()
    assert await v.feed(SILENT_FRAME) is None


# --- fixture-based real-model tests -----------------------------------


def _read_wav_int16_mono_16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, (
            f"fixture {path.name} must be {SAMPLE_RATE} Hz "
            f"(got {w.getframerate()})"
        )
        assert w.getnchannels() == 1, f"fixture {path.name} must be mono"
        assert w.getsampwidth() == 2, f"fixture {path.name} must be int16"
        return w.readframes(w.getnframes())


async def _try_real_vad(**kwargs) -> SileroVAD | None:
    v = SileroVAD(**kwargs)
    try:
        await v.load()
    except VADModelMissingError:
        return None
    return v


async def _walk_clip(v: SileroVAD, clip_bytes: bytes) -> list[VADEvent | None]:
    events = []
    for offset in range(0, len(clip_bytes) - FRAME_BYTES, FRAME_BYTES):
        events.append(await v.feed(clip_bytes[offset:offset + FRAME_BYTES]))
    return events


@pytest.mark.skipif(
    not SPEECH_FIXTURE.exists(),
    reason=f"commit {SPEECH_FIXTURE} (16 kHz mono int16 WAV of speech) to enable",
)
async def test_real_vad_detects_speech_in_clip():
    v = await _try_real_vad()
    if v is None:
        pytest.skip("silero-vad ONNX file not present")
    try:
        events = await _walk_clip(v, _read_wav_int16_mono_16k(SPEECH_FIXTURE))
        assert any(e is VADEvent.SPEECH_STARTED for e in events), (
            "speech fixture did not fire SPEECH_STARTED"
        )
    finally:
        await v.unload()


@pytest.mark.skipif(
    not SILENCE_FIXTURE.exists(),
    reason=f"commit {SILENCE_FIXTURE} (16 kHz mono int16 WAV of silence) to enable",
)
async def test_real_vad_no_speech_started_on_silence_clip():
    v = await _try_real_vad()
    if v is None:
        pytest.skip("silero-vad ONNX file not present")
    try:
        events = await _walk_clip(v, _read_wav_int16_mono_16k(SILENCE_FIXTURE))
        assert all(e is not VADEvent.SPEECH_STARTED for e in events), (
            "silence fixture spuriously fired SPEECH_STARTED"
        )
    finally:
        await v.unload()


async def test_synthesized_silence_does_not_trigger_real_vad():
    """500 ms of zero-valued silence through the real model must never
    fire SPEECH_STARTED. Auto-skips if ONNX file unavailable."""
    v = await _try_real_vad()
    if v is None:
        pytest.skip("silero-vad ONNX file not present")
    try:
        silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.int16).tobytes()
        events = await _walk_clip(v, silence)
        assert all(e is None for e in events)
    finally:
        await v.unload()
