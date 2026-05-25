"""Tests for jarvis.audio.stt.FasterWhisperSTT.

Mock-based tests stub faster_whisper.WhisperModel for fast unit coverage of
the wrapper logic (model id resolution, byte conversion, threading,
whitespace stripping, empty-input shortcut, exception isolation).

One conditional fixture-based test runs against the real whisper model when
the WAV fixture is present; auto-skipped otherwise.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.audio.protocols import SAMPLE_RATE, SpeechToText
from jarvis.audio.stt import FasterWhisperSTT, STTLoadError, _resolve_model_id
from jarvis.core.lifecycle import Loadable

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "audio"
SPEECH_FIXTURE = FIXTURES_DIR / "hello_world.wav"


# --- protocol smoke -----------------------------------------------------


def test_implements_protocols():
    s = FasterWhisperSTT()
    assert isinstance(s, SpeechToText)
    assert isinstance(s, Loadable)


# --- model id resolution -----------------------------------------------


@pytest.mark.parametrize(
    "size,language,expected",
    [
        ("tiny", "en", "tiny.en"),
        ("base", "en", "base.en"),
        ("small", "en", "small.en"),
        ("medium", "en", "medium.en"),
        ("base", "es", "base"),
        ("base", "de", "base"),
        ("large-v3", "en", "large-v3"),  # no .en variant
    ],
)
def test_model_id_resolution(size: str, language: str, expected: str):
    assert _resolve_model_id(size, language) == expected


# --- mock helpers ------------------------------------------------------


@pytest.fixture
def mock_whisper_class():
    with patch("faster_whisper.WhisperModel") as cls:
        cls.return_value = MagicMock()
        yield cls


def _instance(cls):
    return cls.return_value


def _segment(text: str):
    seg = MagicMock()
    seg.text = text
    return seg


def _set_segments(instance, *segments):
    instance.transcribe.return_value = (iter(segments), MagicMock())


# --- load / unload ----------------------------------------------------


async def test_load_constructs_with_resolved_model_id(mock_whisper_class):
    s = FasterWhisperSTT(model_size="base", language="en", compute_type="int8")
    await s.load()
    assert s.is_loaded
    args, kwargs = mock_whisper_class.call_args
    assert args[0] == "base.en"
    assert kwargs["device"] == "cpu"
    assert kwargs["compute_type"] == "int8"


async def test_load_passes_download_root(mock_whisper_class, tmp_path):
    s = FasterWhisperSTT(download_root=tmp_path)
    await s.load()
    _, kwargs = mock_whisper_class.call_args
    assert kwargs["download_root"] == str(tmp_path)


async def test_load_with_no_download_root_passes_none(mock_whisper_class):
    s = FasterWhisperSTT()
    await s.load()
    _, kwargs = mock_whisper_class.call_args
    assert kwargs["download_root"] is None


async def test_load_idempotent(mock_whisper_class):
    s = FasterWhisperSTT()
    await s.load()
    await s.load()
    assert mock_whisper_class.call_count == 1


async def test_unload_releases_model(mock_whisper_class):
    s = FasterWhisperSTT()
    await s.load()
    await s.unload()
    assert not s.is_loaded
    assert s._model is None  # type: ignore[attr-defined]


async def test_load_failure_raises_stt_load_error(mock_whisper_class):
    mock_whisper_class.side_effect = OSError("model not found")
    s = FasterWhisperSTT()
    with pytest.raises(STTLoadError, match="could not load"):
        await s.load()
    assert not s.is_loaded


async def test_load_runs_in_thread_executor(mock_whisper_class):
    """Construction can take seconds (download); must not block the loop."""
    loop_thread = threading.current_thread()
    capture: dict = {}

    def make_model(*args, **kwargs):
        capture["thread"] = threading.current_thread()
        return MagicMock()

    mock_whisper_class.side_effect = make_model
    s = FasterWhisperSTT()
    await s.load()
    assert capture["thread"] is not loop_thread


# --- transcribe shortcuts -------------------------------------------


async def test_transcribe_empty_bytes_short_circuits(mock_whisper_class):
    s = FasterWhisperSTT()
    await s.load()
    result = await s.transcribe(b"")
    assert result == ""
    _instance(mock_whisper_class).transcribe.assert_not_called()


async def test_transcribe_before_load_returns_empty(caplog: pytest.LogCaptureFixture):
    s = FasterWhisperSTT()
    with caplog.at_level(logging.WARNING, logger="jarvis.audio.stt"):
        result = await s.transcribe(b"\x00" * 1000)
    assert result == ""
    assert any("before load" in r.message for r in caplog.records)


# --- transcribe behavior --------------------------------------------


async def test_transcribe_concatenates_segments(mock_whisper_class):
    s = FasterWhisperSTT()
    await s.load()
    _set_segments(_instance(mock_whisper_class), _segment("hello "), _segment("world"))
    result = await s.transcribe(b"\x00" * 1000)
    assert result == "hello world"


async def test_transcribe_strips_whitespace(mock_whisper_class):
    s = FasterWhisperSTT()
    await s.load()
    _set_segments(_instance(mock_whisper_class), _segment("  leading and trailing   "))
    result = await s.transcribe(b"\x00" * 1000)
    assert result == "leading and trailing"


async def test_transcribe_strips_whitespace_only_to_empty(mock_whisper_class):
    """Whisper's whitespace-only output should normalize to empty so the
    pipeline's IDLE-on-empty path fires correctly."""
    s = FasterWhisperSTT()
    await s.load()
    _set_segments(_instance(mock_whisper_class), _segment("   "))
    result = await s.transcribe(b"\x00" * 1000)
    assert result == ""


async def test_transcribe_converts_int16_bytes_to_float32_normalized(
    mock_whisper_class,
):
    s = FasterWhisperSTT()
    await s.load()
    _set_segments(_instance(mock_whisper_class), _segment(""))
    pattern = np.array([16384] * 100, dtype=np.int16)
    audio_bytes = pattern.tobytes()
    await s.transcribe(audio_bytes)
    args, kwargs = _instance(mock_whisper_class).transcribe.call_args
    audio = args[0]
    assert audio.dtype == np.float32
    assert audio.shape == (100,)
    # 16384 / 32768 == 0.5
    assert audio[0] == pytest.approx(0.5, abs=1e-4)
    assert kwargs["language"] == "en"


async def test_transcribe_isolates_inference_exception(
    mock_whisper_class, caplog: pytest.LogCaptureFixture
):
    s = FasterWhisperSTT()
    await s.load()
    _instance(mock_whisper_class).transcribe.side_effect = RuntimeError(
        "model crashed"
    )
    with caplog.at_level(logging.ERROR, logger="jarvis.audio.stt"):
        result = await s.transcribe(b"\x00" * 1000)
    assert result == ""
    assert any("whisper transcribe raised" in r.message for r in caplog.records)


async def test_transcribe_passes_vad_filter_true(mock_whisper_class):
    """Bug E: the wrapper must pass vad_filter=True to faster-whisper to
    suppress hallucinations on short / low-signal clips."""
    s = FasterWhisperSTT()
    await s.load()
    _set_segments(_instance(mock_whisper_class), _segment("hi"))
    await s.transcribe(b"\x00" * 1000)
    _, kwargs = _instance(mock_whisper_class).transcribe.call_args
    assert kwargs.get("vad_filter") is True


async def test_transcribe_passes_configured_language(mock_whisper_class):
    s = FasterWhisperSTT(model_size="base", language="es")
    await s.load()
    _set_segments(_instance(mock_whisper_class), _segment(""))
    await s.transcribe(b"\x00" * 1000)
    _, kwargs = _instance(mock_whisper_class).transcribe.call_args
    assert kwargs["language"] == "es"


# --- threading: must not block the loop ----------------------------


async def test_transcribe_runs_in_thread_executor(mock_whisper_class):
    """Critical: transcribe() must run faster-whisper inference in a
    thread executor so the asyncio loop stays free during inference.
    Without this, barge-in detection freezes for the duration of
    transcription."""
    s = FasterWhisperSTT()
    await s.load()

    loop_thread = threading.current_thread()
    capture: dict = {}

    def slow_transcribe(audio, **kw):
        capture["thread"] = threading.current_thread()
        return (iter([_segment("done")]), MagicMock())

    _instance(mock_whisper_class).transcribe.side_effect = slow_transcribe
    await s.transcribe(b"\x00" * 1000)
    assert capture["thread"] is not None
    assert capture["thread"] is not loop_thread, (
        "transcribe() ran on the loop thread; would block frame loop"
    )


async def test_loop_remains_responsive_during_transcribe(mock_whisper_class):
    """End-to-end tripwire: a concurrent task must continue to tick while
    transcribe inference is in flight. If the wrapper accidentally awaits
    on the loop thread, the counter wouldn't advance during the blocking
    period."""
    s = FasterWhisperSTT()
    await s.load()

    def slow_transcribe(audio, **kw):
        time.sleep(0.05)  # blocks the executor thread, not the loop
        return (iter([_segment("done")]), MagicMock())

    _instance(mock_whisper_class).transcribe.side_effect = slow_transcribe

    counter = {"n": 0}

    async def keep_busy():
        for _ in range(20):
            counter["n"] += 1
            await asyncio.sleep(0.005)

    busy_task = asyncio.create_task(keep_busy())
    await s.transcribe(b"\x00" * 1000)
    await busy_task
    # Over the ~50 ms blocking period, the keep_busy task should have ticked
    # multiple times. If transcribe() blocked the loop, counter would still
    # be 0 or near-0 when transcribe() returned.
    assert counter["n"] >= 5, (
        f"loop ticked only {counter['n']} times during transcribe; "
        "loop is likely being blocked"
    )


# --- materialization: generator must be consumed inside the thread ---


async def test_segment_generator_consumed_in_executor_thread(mock_whisper_class):
    """The faster-whisper segment generator IS the inference work. It must
    be consumed inside the thread executor; returning an unconsumed
    generator across the to_thread boundary would defer the heavy lifting
    back onto the loop."""
    s = FasterWhisperSTT()
    await s.load()

    loop_thread = threading.current_thread()
    consume_threads: list = []

    class TrackingSegments:
        def __init__(self):
            self._items = [_segment("hi"), _segment(" there")]
            self._idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            consume_threads.append(threading.current_thread())
            if self._idx >= len(self._items):
                raise StopIteration
            item = self._items[self._idx]
            self._idx += 1
            return item

    _instance(mock_whisper_class).transcribe.return_value = (
        TrackingSegments(), MagicMock()
    )
    result = await s.transcribe(b"\x00" * 1000)
    assert result == "hi there"
    assert consume_threads, "generator was never iterated"
    assert all(t is not loop_thread for t in consume_threads), (
        "segment generator was iterated on the loop thread"
    )


# --- fixture-based real-model test --------------------------------


def _read_wav_int16_mono_16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        return w.readframes(w.getnframes())


@pytest.mark.skipif(
    not SPEECH_FIXTURE.exists(),
    reason=f"commit {SPEECH_FIXTURE} (16 kHz mono int16 'hello world') to enable",
)
async def test_real_whisper_transcribes_clip():
    s = FasterWhisperSTT()
    try:
        await s.load()
    except STTLoadError:
        pytest.skip("whisper model unavailable; first-run download or no network")
    try:
        audio = _read_wav_int16_mono_16k(SPEECH_FIXTURE)
        text = await s.transcribe(audio)
        assert text, "expected non-empty transcription"
        # Whisper may capitalize/punctuate; lowercase comparison.
        lowered = text.lower()
        assert "hello" in lowered or "world" in lowered, (
            f"unexpected transcription: {text!r}"
        )
    finally:
        await s.unload()
