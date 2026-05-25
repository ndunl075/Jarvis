"""Tests for jarvis.audio.wake_word.OpenWakeWord.

Most tests stub openwakeword.model.Model so they verify the wrapper logic
(sensitivity mapping, byte->numpy conversion, lifecycle, exception
isolation) without loading real ONNX models.

Two fixture-based tests run against the real openWakeWord model when WAV
fixtures are present in tests/fixtures/audio/. They auto-skip when fixtures
or model files are missing -- no CI dependency on audio assets.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.audio.protocols import FRAME_BYTES, SAMPLE_RATE, WakeWordDetector, WakeWordResult
from jarvis.audio.wake_word import OpenWakeWord, WakeWordModelMissingError
from jarvis.core.lifecycle import Loadable

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "audio"
HEY_JARVIS_FIXTURE = FIXTURES_DIR / "hey_jarvis.wav"
BACKGROUND_FIXTURE = FIXTURES_DIR / "background_speech.wav"


# --- protocol smoke ------------------------------------------------------


def test_implements_protocols():
    ww = OpenWakeWord()
    assert isinstance(ww, WakeWordDetector)
    assert isinstance(ww, Loadable)


def test_invalid_sensitivity_rejected():
    with pytest.raises(ValueError):
        OpenWakeWord(sensitivity=1.5)
    with pytest.raises(ValueError):
        OpenWakeWord(sensitivity=-0.1)


# --- sensitivity mapping -------------------------------------------------


@pytest.mark.parametrize(
    "sensitivity,expected_threshold",
    [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 0.0)],
)
def test_sensitivity_to_threshold_mapping(
    sensitivity: float, expected_threshold: float
):
    ww = OpenWakeWord(sensitivity=sensitivity)
    assert ww.threshold == pytest.approx(expected_threshold)


# --- mock-based wrapper logic -------------------------------------------


@pytest.fixture
def mock_model_class():
    """Patch the openwakeword.model.Model class AND the model-path resolver.
    The resolver is patched so tests don't depend on an actual ONNX file
    being present in the dev venv (the resolver greps the imported package's
    resources/models dir). Returns the patched Model class."""
    fake_path = Path("/fake/openwakeword/resources/models/hey_jarvis_v0.1.onnx")
    with patch("openwakeword.model.Model") as cls, \
         patch("jarvis.audio.wake_word._resolve_model_path", return_value=fake_path):
        cls.return_value = MagicMock()
        yield cls


def _instance(cls):
    return cls.return_value


async def test_load_constructs_model_with_hey_jarvis_and_onnx(mock_model_class):
    ww = OpenWakeWord()
    await ww.load()
    assert ww.is_loaded
    mock_model_class.assert_called_once()
    _, kwargs = mock_model_class.call_args
    # Full path is passed (not bare name) so openwakeword can't sys.path-walk
    # to a different install of itself than the one we imported.
    assert len(kwargs["wakeword_models"]) == 1
    assert kwargs["wakeword_models"][0].endswith("hey_jarvis_v0.1.onnx")
    assert kwargs["inference_framework"] == "onnx"
    assert ww.model_path is not None
    assert ww.model_path.name == "hey_jarvis_v0.1.onnx"


async def test_load_idempotent(mock_model_class):
    ww = OpenWakeWord()
    await ww.load()
    await ww.load()
    assert mock_model_class.call_count == 1


async def test_unload_releases_model(mock_model_class):
    ww = OpenWakeWord()
    await ww.load()
    await ww.unload()
    assert not ww.is_loaded
    assert ww._model is None  # type: ignore[attr-defined]


async def test_unload_idempotent(mock_model_class):
    ww = OpenWakeWord()
    await ww.load()
    await ww.unload()
    await ww.unload()  # no-op


async def test_load_failure_raises_with_recovery_hint():
    fake_path = Path("/fake/openwakeword/resources/models/hey_jarvis_v0.1.onnx")
    with patch(
        "openwakeword.model.Model",
        side_effect=FileNotFoundError("hey_jarvis_v0.1.onnx not found"),
    ), patch("jarvis.audio.wake_word._resolve_model_path", return_value=fake_path):
        ww = OpenWakeWord()
        with pytest.raises(WakeWordModelMissingError) as exc:
            await ww.load()
        assert "download_models" in str(exc.value)
        assert not ww.is_loaded


async def test_feed_below_threshold_returns_none(mock_model_class):
    ww = OpenWakeWord(sensitivity=0.5)  # threshold=0.5
    _instance(mock_model_class).predict.return_value = {"hey_jarvis": 0.3}
    await ww.load()
    assert await ww.feed(b"\x00" * FRAME_BYTES) is None


async def test_feed_at_threshold_returns_result(mock_model_class):
    ww = OpenWakeWord(sensitivity=0.5)
    _instance(mock_model_class).predict.return_value = {"hey_jarvis": 0.5}
    await ww.load()
    result = await ww.feed(b"\x00" * FRAME_BYTES)
    assert result == WakeWordResult(confidence=0.5)


async def test_feed_above_threshold_returns_result(mock_model_class):
    ww = OpenWakeWord(sensitivity=0.5)
    _instance(mock_model_class).predict.return_value = {"hey_jarvis": 0.95}
    await ww.load()
    result = await ww.feed(b"\x00" * FRAME_BYTES)
    assert result is not None
    assert result.confidence == pytest.approx(0.95)


async def test_feed_uses_first_dict_value_regardless_of_key_name(mock_model_class):
    """openWakeWord keys its predict() return dict by the FULL model name
    including a version suffix (e.g. 'hey_jarvis_v0.1'), which is NOT the
    short name we passed at construction. The wrapper must take the score
    by position, not by key, or every detection silently reads 0.0 — the
    exact bug that hid 99%+ confidence detections in production."""
    ww = OpenWakeWord(sensitivity=0.5)  # threshold=0.5
    # Future-proof against another upstream rename: arbitrary key, real value.
    _instance(mock_model_class).predict.return_value = {"hey_jarvis_v2": 0.95}
    await ww.load()
    result = await ww.feed(b"\x00" * FRAME_BYTES)
    assert result is not None
    assert result.confidence == pytest.approx(0.95)


async def test_feed_higher_sensitivity_lowers_threshold(mock_model_class):
    # sensitivity=0.9 -> threshold=0.1; even a low score should fire.
    ww = OpenWakeWord(sensitivity=0.9)
    _instance(mock_model_class).predict.return_value = {"hey_jarvis": 0.15}
    await ww.load()
    result = await ww.feed(b"\x00" * FRAME_BYTES)
    assert result is not None


async def test_feed_lower_sensitivity_raises_threshold(mock_model_class):
    # sensitivity=0.1 -> threshold=0.9; mid-range score must not fire.
    ww = OpenWakeWord(sensitivity=0.1)
    _instance(mock_model_class).predict.return_value = {"hey_jarvis": 0.5}
    await ww.load()
    assert await ww.feed(b"\x00" * FRAME_BYTES) is None


async def test_feed_converts_int16_bytes_to_numpy(mock_model_class):
    ww = OpenWakeWord()
    _instance(mock_model_class).predict.return_value = {"hey_jarvis": 0.0}
    await ww.load()
    pattern = np.array([100, -200, 300] * 160, dtype=np.int16)
    frame = pattern.tobytes()
    assert len(frame) == FRAME_BYTES
    await ww.feed(frame)
    args, _ = _instance(mock_model_class).predict.call_args
    audio = args[0]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.int16
    assert audio.shape == (480,)
    np.testing.assert_array_equal(audio, pattern)


async def test_feed_wrong_size_returns_none_and_logs(
    mock_model_class, caplog: pytest.LogCaptureFixture
):
    ww = OpenWakeWord()
    await ww.load()
    with caplog.at_level(logging.ERROR, logger="jarvis.audio.wake_word"):
        result = await ww.feed(b"\x00" * 100)
    assert result is None
    assert any("wrong frame size" in r.message for r in caplog.records)
    _instance(mock_model_class).predict.assert_not_called()


async def test_feed_isolates_predict_exception(
    mock_model_class, caplog: pytest.LogCaptureFixture
):
    ww = OpenWakeWord()
    _instance(mock_model_class).predict.side_effect = RuntimeError("model crashed")
    await ww.load()
    with caplog.at_level(logging.ERROR, logger="jarvis.audio.wake_word"):
        result = await ww.feed(b"\x00" * FRAME_BYTES)
    assert result is None
    assert any("openwakeword.predict raised" in r.message for r in caplog.records)


async def test_feed_before_load_returns_none():
    ww = OpenWakeWord()
    assert await ww.feed(b"\x00" * FRAME_BYTES) is None


async def test_reset_calls_model_reset(mock_model_class):
    ww = OpenWakeWord()
    await ww.load()
    ww.reset()
    _instance(mock_model_class).reset.assert_called_once()


def test_reset_before_load_is_noop():
    ww = OpenWakeWord()
    ww.reset()  # must not raise


async def test_reset_isolates_exception(
    mock_model_class, caplog: pytest.LogCaptureFixture
):
    ww = OpenWakeWord()
    _instance(mock_model_class).reset.side_effect = RuntimeError("boom")
    await ww.load()
    with caplog.at_level(logging.ERROR, logger="jarvis.audio.wake_word"):
        ww.reset()
    assert any("openwakeword.reset raised" in r.message for r in caplog.records)


# --- fixture-based real-model tests --------------------------------------
# These run only if (a) the WAV fixtures are committed and (b) the
# openWakeWord model files are present in the venv. No real audio in CI
# unless the operator chose to populate both.


def _read_wav_int16_mono_16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, (
            f"fixture {path.name} must be {SAMPLE_RATE} Hz (got {w.getframerate()})"
        )
        assert w.getnchannels() == 1, f"fixture {path.name} must be mono"
        assert w.getsampwidth() == 2, f"fixture {path.name} must be int16"
        return w.readframes(w.getnframes())


async def _try_load_real_model(sensitivity: float = 0.5) -> OpenWakeWord | None:
    ww = OpenWakeWord(sensitivity=sensitivity)
    try:
        await ww.load()
    except WakeWordModelMissingError:
        return None
    return ww


async def _max_score_over_clip(ww: OpenWakeWord, clip_bytes: bytes) -> float:
    # Drive the wrapper's full pipeline (bytes -> int16 numpy -> predict)
    # by feeding 30 ms frames in order. We need direct access to scores
    # rather than the threshold-gated WakeWordResult, so call the underlying
    # model directly with the same conversion.
    max_score = 0.0
    assert ww._model is not None
    for offset in range(0, len(clip_bytes) - FRAME_BYTES, FRAME_BYTES):
        frame = clip_bytes[offset:offset + FRAME_BYTES]
        audio = np.frombuffer(frame, dtype=np.int16)
        scores = ww._model.predict(audio)
        s = float(scores.get("hey_jarvis", 0.0))
        if s > max_score:
            max_score = s
    return max_score


async def test_silence_does_not_trigger_real_model():
    """500 ms of silence must not trigger the real 'hey_jarvis' model."""
    ww = await _try_load_real_model(sensitivity=0.5)
    if ww is None:
        pytest.skip("openWakeWord model files not present; run download_models()")
    try:
        silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.int16).tobytes()
        for offset in range(0, len(silence) - FRAME_BYTES, FRAME_BYTES):
            frame = silence[offset:offset + FRAME_BYTES]
            assert await ww.feed(frame) is None
    finally:
        await ww.unload()


@pytest.mark.skipif(
    not HEY_JARVIS_FIXTURE.exists(),
    reason=f"commit {HEY_JARVIS_FIXTURE} (mono 16 kHz int16 WAV of 'hey jarvis') to enable",
)
async def test_hey_jarvis_clip_triggers_real_model():
    ww = await _try_load_real_model(sensitivity=0.5)
    if ww is None:
        pytest.skip("openWakeWord model files not present; run download_models()")
    try:
        clip = _read_wav_int16_mono_16k(HEY_JARVIS_FIXTURE)
        max_score = await _max_score_over_clip(ww, clip)
        assert max_score >= ww.threshold, (
            f"hey_jarvis fixture peak score {max_score:.3f} did not reach "
            f"threshold {ww.threshold:.3f}"
        )
    finally:
        await ww.unload()


@pytest.mark.skipif(
    not BACKGROUND_FIXTURE.exists(),
    reason=f"commit {BACKGROUND_FIXTURE} (non-wakeword speech WAV) to enable",
)
async def test_background_speech_does_not_trigger_real_model():
    ww = await _try_load_real_model(sensitivity=0.5)
    if ww is None:
        pytest.skip("openWakeWord model files not present; run download_models()")
    try:
        clip = _read_wav_int16_mono_16k(BACKGROUND_FIXTURE)
        max_score = await _max_score_over_clip(ww, clip)
        assert max_score < ww.threshold, (
            f"background fixture peaked at {max_score:.3f}, "
            f"above threshold {ww.threshold:.3f} (false positive)"
        )
    finally:
        await ww.unload()
