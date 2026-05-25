"""Tests for jarvis.audio.tts.PiperTTS.

Mock-based tests fully stub Piper and sounddevice. They cover:
  - Lifecycle (load/unload, voice file resolution, idempotency, errors)
  - speak() flow (writes to stream, applies volume, runs in executor)
  - cancel() actually aborts the stream (not stop()) and breaks _sync_speak
  - Sentence-segmenting speak_stream (boundary, max-wait, final flush)
  - Amplitude envelope at ~30 Hz with the configured callback
  - Empty / pre-load / exception isolation paths

One conditional fixture-based test runs against the real Piper voice if
en_GB-alan-medium files are present in tests/fixtures/audio/voices/.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.audio.protocols import TextToSpeech
from jarvis.audio.tts import PiperTTS, TTSLoadError
from jarvis.core.lifecycle import Loadable

VOICES_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "audio" / "voices"
REAL_VOICE_NAME = "en_GB-alan-medium"


@pytest.fixture
def time_compress_off(monkeypatch):
    """Opt-in fixture for tests that assert exact byte-content equality
    on TTS output. The shipped default (TIME_COMPRESS_FACTOR=1.30)
    shrinks audio to ~10/13 of its original length, which breaks
    byte-exact tests but is correct for the production playback path.
    Tests that care about playback timing/wiring (queue blocks, callback
    flow) opt in; tests that verify the compression itself don't."""
    monkeypatch.setattr("jarvis.audio.tts.TIME_COMPRESS_FACTOR", None)


# --- protocol smoke ----------------------------------------------------


def test_implements_protocols():
    t = PiperTTS()
    assert isinstance(t, TextToSpeech)
    assert isinstance(t, Loadable)


def test_invalid_volume_rejected():
    with pytest.raises(ValueError):
        PiperTTS(volume=1.5)
    with pytest.raises(ValueError):
        PiperTTS(volume=-0.1)


def test_invalid_speed_rejected():
    with pytest.raises(ValueError):
        PiperTTS(speed=0.0)
    with pytest.raises(ValueError):
        PiperTTS(speed=-1.0)


def test_effective_length_scale_combines_baseline_and_speed():
    """SPEECH_LENGTH_SCALE is the conversational-pace baseline; the per-
    instance `speed` multiplier divides it. speed=1.0 -> baseline; speed=1.2
    -> ~0.71 (snappier). Speed bumps must therefore *decrease* length_scale,
    not increase it — getting the direction wrong made Piper sound drugged
    when we last shipped the inverse mapping by mistake."""
    from jarvis.audio.tts import SPEECH_LENGTH_SCALE
    assert PiperTTS(speed=1.0)._effective_length_scale() == pytest.approx(
        SPEECH_LENGTH_SCALE
    )
    assert PiperTTS(speed=1.2)._effective_length_scale() == pytest.approx(
        SPEECH_LENGTH_SCALE / 1.2
    )
    # speed=2.0 -> 0.425 — half the duration of baseline.
    assert PiperTTS(speed=2.0)._effective_length_scale() == pytest.approx(
        SPEECH_LENGTH_SCALE / 2.0
    )


async def test_synthesize_receives_length_scale_via_syn_config():
    """End-to-end propagation: the value returned by _effective_length_scale
    must reach Piper's voice.synthesize() as SynthesisConfig.length_scale.
    Without this the constant is dead code and Piper synthesizes at its
    upstream default pace."""
    from jarvis.audio.tts import SPEECH_LENGTH_SCALE
    tts, voice, _ = _loaded_tts(voice_chunks=[b"\x00\x01" * 100])
    tts.speed = 1.0  # explicit; matches default
    await tts.speak("hi")
    args, kwargs = voice.synthesize.call_args
    syn_config = kwargs.get("syn_config")
    assert syn_config is not None
    assert syn_config.length_scale == pytest.approx(SPEECH_LENGTH_SCALE)


async def test_synthesize_length_scale_honours_speed_multiplier():
    from jarvis.audio.tts import SPEECH_LENGTH_SCALE
    tts, voice, _ = _loaded_tts(voice_chunks=[b"\x00\x01" * 100])
    tts.speed = 1.5
    await tts.speak("hi")
    _, kwargs = voice.synthesize.call_args
    syn_config = kwargs.get("syn_config")
    assert syn_config is not None
    assert syn_config.length_scale == pytest.approx(SPEECH_LENGTH_SCALE / 1.5)


# --- helpers -----------------------------------------------------------


def _make_fake_stream() -> MagicMock:
    s = MagicMock()
    s.active = False

    def start():
        s.active = True

    def stop():
        s.active = False

    def abort():
        s.active = False

    s.start.side_effect = start
    s.stop.side_effect = stop
    s.abort.side_effect = abort
    return s


def _make_fake_voice(sample_rate: int = 22050) -> MagicMock:
    v = MagicMock()
    v.config.sample_rate = sample_rate
    return v


def _audio_chunk(byte_payload: bytes) -> MagicMock:
    c = MagicMock()
    c.audio_int16_bytes = byte_payload
    return c


def _loaded_tts(
    *,
    voice_chunks: list[bytes] | None = None,
    on_amplitude=None,
    sample_rate: int = 22050,
    volume: float = 1.0,
) -> tuple[PiperTTS, MagicMock, MagicMock]:
    """Return a PiperTTS with mocks pre-installed (skipping the real load
    path). Returns (tts, voice, stream).

    Because we bypass _open_stream, _callback_wired stays False — that's
    deliberate: PiperTTS's drain wait short-circuits when the callback is
    not wired (no live PortAudio thread to drain the queue), so
    `await tts.speak(...)` returns once everything is queued. Tests then
    drive the callback manually via _drain_via_callback to inspect bytes."""
    voice = _make_fake_voice(sample_rate=sample_rate)
    stream = _make_fake_stream()
    if voice_chunks is None:
        voice_chunks = [b"\x00\x01" * 100]
    voice.synthesize.return_value = iter([_audio_chunk(c) for c in voice_chunks])
    tts = PiperTTS(
        voices_dir=Path("/fake"),
        on_amplitude=on_amplitude,
        volume=volume,
    )
    tts._voice = voice
    tts._stream = stream
    tts._sample_rate = sample_rate
    tts._output_rate = sample_rate
    tts.is_loaded = True
    return tts, voice, stream


def _drain_via_callback(tts: PiperTTS, frames: int = 2048) -> bytes:
    """Drive the PortAudio callback from the test thread until the queue and
    pending residual are exhausted. Returns the concatenated playback bytes
    with trailing silence trimmed.

    This is the test-side equivalent of PortAudio's audio thread pulling
    callback after callback. Real production playback works the same way;
    we just drive it manually because there's no real device behind the
    MagicMock stream."""
    out = bytearray()
    # Safety bound: at 22050 Hz / int16 / frames=2048 each iter consumes
    # ~93 ms; 2000 iters = ~3 minutes of audio, far beyond any real test.
    for _ in range(2000):
        buf = bytearray(frames * 2)
        tts._on_audio_callback(buf, frames, None, None)
        out.extend(buf)
        if tts._audio_queue.empty() and not tts._pending:
            break
    while out and out[-1] == 0:
        out.pop()
    return bytes(out)


def _install_callback_drainer(tts: PiperTTS) -> threading.Event:
    """Spawn a daemon thread that drives _on_audio_callback continuously
    while the test runs. Use only with tests that exercise the real
    _open_stream path (where _callback_wired is True), so _sync_speak's
    drain wait can complete. Returns the stop Event so the test can shut
    the drainer down (the daemon flag also kills it on test exit)."""
    stop = threading.Event()
    frames = 2048

    def loop() -> None:
        while not stop.is_set():
            buf = bytearray(frames * 2)
            try:
                tts._on_audio_callback(buf, frames, None, None)
            except Exception:
                return
            stream = tts._stream
            if stream is not None:
                try:
                    stream.write(bytes(buf))
                except Exception:
                    pass
            time.sleep(0.001)

    threading.Thread(target=loop, daemon=True).start()
    return stop


# --- load / unload ----------------------------------------------------


async def test_load_resolves_voice_files_and_constructs_stream(tmp_path):
    onnx = tmp_path / f"{REAL_VOICE_NAME}.onnx"
    config = tmp_path / f"{REAL_VOICE_NAME}.onnx.json"
    onnx.write_bytes(b"fake")
    config.write_text("{}")

    fake_voice = _make_fake_voice(sample_rate=22050)
    fake_stream = _make_fake_stream()
    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = fake_voice
        sd_mock.RawOutputStream.return_value = fake_stream

        tts = PiperTTS(voice_name=REAL_VOICE_NAME, voices_dir=tmp_path)
        await tts.load()
        assert tts.is_loaded
        voice_cls.load.assert_called_once_with(str(onnx))
        sd_mock.RawOutputStream.assert_called_once()
        _, kwargs = sd_mock.RawOutputStream.call_args
        assert kwargs["samplerate"] == 22050
        assert kwargs["channels"] == 1
        assert kwargs["dtype"] == "int16"
        # blocksize is matched to OUTPUT_BUFFER_MS (see _open_stream
        # docstring). Asserting > 0 instead of an exact value keeps the
        # test forward-compatible to tuning the buffer constant.
        assert kwargs["blocksize"] > 0
        assert kwargs["latency"] == "high"


async def test_load_missing_voice_files_raises(tmp_path):
    tts = PiperTTS(voice_name=REAL_VOICE_NAME, voices_dir=tmp_path)
    with pytest.raises(TTSLoadError, match="not found"):
        await tts.load()


async def test_load_missing_config_json_raises(tmp_path):
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    # No .onnx.json
    tts = PiperTTS(voice_name=REAL_VOICE_NAME, voices_dir=tmp_path)
    with pytest.raises(TTSLoadError, match="not found"):
        await tts.load()


async def test_load_voice_load_failure_raises(tmp_path):
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")
    with patch("piper.PiperVoice") as voice_cls:
        voice_cls.load.side_effect = OSError("corrupt model")
        tts = PiperTTS(voice_name=REAL_VOICE_NAME, voices_dir=tmp_path)
        with pytest.raises(TTSLoadError, match="could not load"):
            await tts.load()
    assert not tts.is_loaded


async def test_load_idempotent(tmp_path):
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")
    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.return_value = _make_fake_stream()
        tts = PiperTTS(voice_name=REAL_VOICE_NAME, voices_dir=tmp_path)
        await tts.load()
        await tts.load()
        assert voice_cls.load.call_count == 1


async def test_unload_closes_stream_and_releases_voice():
    tts, _, stream = _loaded_tts()
    await tts.unload()
    assert not tts.is_loaded
    stream.close.assert_called_once()
    assert tts._voice is None  # type: ignore[attr-defined]
    assert tts._stream is None  # type: ignore[attr-defined]


# --- output device routing -------------------------------------------


def _output_device(name: str, max_output_channels: int = 2) -> dict:
    return {"name": name, "max_output_channels": max_output_channels}


async def _load_with_output(output_device, query_devices_return, tmp_path):
    """Helper: load a PiperTTS with a given output_device, returning the
    sd_mock so the test can inspect the RawOutputStream call."""
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")
    fake_voice = _make_fake_voice(sample_rate=22050)
    fake_stream = _make_fake_stream()
    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = fake_voice
        sd_mock.RawOutputStream.return_value = fake_stream
        sd_mock.query_devices.return_value = query_devices_return
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device=output_device,
        )
        await tts.load()
        return tts, sd_mock


async def test_output_device_none_passes_none_to_stream(tmp_path):
    _, sd_mock = await _load_with_output(None, [_output_device("Speakers")], tmp_path)
    _, kwargs = sd_mock.RawOutputStream.call_args
    assert kwargs["device"] is None


async def test_output_device_int_passes_through(tmp_path):
    _, sd_mock = await _load_with_output(3, [_output_device("Speakers")], tmp_path)
    _, kwargs = sd_mock.RawOutputStream.call_args
    assert kwargs["device"] == 3


async def test_output_device_name_substring_matched_case_insensitive(tmp_path):
    devices = [
        _output_device("Built-in Output"),
        _output_device("Headphones (USB)"),
        _output_device("HDMI Audio"),
    ]
    _, sd_mock = await _load_with_output("HEADPHONES", devices, tmp_path)
    _, kwargs = sd_mock.RawOutputStream.call_args
    assert kwargs["device"] == 1


async def test_output_device_skips_input_only_devices(tmp_path):
    # An input-only device whose name happens to match must NOT be picked
    # for output routing.
    devices = [
        _output_device("USB Microphone", max_output_channels=0),
        _output_device("Default Speakers"),
    ]
    _, sd_mock = await _load_with_output("microphone", devices, tmp_path)
    # No match -> falls back to None (system default).
    _, kwargs = sd_mock.RawOutputStream.call_args
    assert kwargs["device"] is None


async def test_output_device_prefers_wasapi_over_mme(tmp_path):
    """When both MME and WASAPI variants match the name substring, WASAPI is
    picked to avoid the MME stream-restart race after barge-in abort()."""
    devices = [
        {"name": "Pebble Pro (MME)", "max_output_channels": 2, "hostapi": 0},
        {"name": "Pebble Pro (Windows WASAPI)", "max_output_channels": 2, "hostapi": 1},
    ]
    hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")
    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.return_value = _make_fake_stream()
        sd_mock.query_devices.return_value = devices
        sd_mock.query_hostapis.return_value = hostapis
        tts = PiperTTS(voice_name=REAL_VOICE_NAME, voices_dir=tmp_path, output_device="Pebble Pro")
        await tts.load()
    _, kwargs = sd_mock.RawOutputStream.call_args
    assert kwargs["device"] == 1  # WASAPI variant at index 1


async def test_output_device_not_found_falls_back_to_default_with_warning(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    devices = [_output_device("Speakers"), _output_device("HDMI")]
    with caplog.at_level(logging.WARNING, logger="jarvis.audio.tts"):
        _, sd_mock = await _load_with_output("ghost device", devices, tmp_path)
    _, kwargs = sd_mock.RawOutputStream.call_args
    assert kwargs["device"] is None
    assert any("not found" in r.message for r in caplog.records)


async def test_output_device_name_match_logs_candidates(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """Matching candidates and selected device are logged at INFO level."""
    devices = [_output_device("USB Headphones")]
    with caplog.at_level(logging.INFO, logger="jarvis.audio.tts"):
        await _load_with_output("headphones", devices, tmp_path)
    assert any("output device candidates" in r.message for r in caplog.records)


# --- "(2-" phantom device de-prioritization --------------------------


async def test_prefers_real_device_over_2prefix_phantom(tmp_path):
    """When both a real device and a Windows '(2- …)' phantom enumeration
    match the configured substring, the real one (no '(2- ' prefix) is
    selected even though it has a higher device index."""
    devices = [
        # Index 0: phantom — Windows re-enumeration, often fails to open
        {"name": "(2- Creative Pebble Pro)", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
        # Index 1: real device — should be preferred
        {"name": "Speakers (Creative Pebble Pro)", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
    ]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.return_value = _make_fake_stream()
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="Creative Pebble Pro",
        )
        await tts.load()

    _, kwargs = sd_mock.RawOutputStream.call_args
    assert kwargs["device"] == 1  # real device, not phantom at idx 0


async def test_phantom_only_still_used_when_no_real_match(tmp_path):
    """If the only matching device IS the phantom, it's still used — we
    de-prioritize it but don't discard it entirely."""
    devices = [
        {"name": "(2- Creative Pebble Pro)", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
    ]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.return_value = _make_fake_stream()
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="Creative Pebble Pro",
        )
        await tts.load()

    assert tts.is_loaded
    _, kwargs = sd_mock.RawOutputStream.call_args
    assert kwargs["device"] == 0


# --- rewire_output (hot device swap) ---------------------------------


async def test_rewire_output_reopens_stream_on_new_device(tmp_path):
    """rewire_output() closes the current stream and opens a fresh one on
    the specified device without reloading the voice model."""
    devices = [
        {"name": "Pebble Pro", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
        {"name": "HDMI Out", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
    ]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    stream_a = _make_fake_stream()
    stream_b = _make_fake_stream()
    open_calls: list[dict] = []
    streams = iter([stream_a, stream_b])

    def raw_output_stream(**kwargs):
        open_calls.append(dict(kwargs))
        return next(streams)

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.side_effect = raw_output_stream
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="Pebble Pro",
        )
        await tts.load()
        assert open_calls[0]["device"] == 0
        voice_before = tts._voice

        await tts.rewire_output("HDMI Out")

    assert len(open_calls) == 2
    assert open_calls[1]["device"] == 1
    # Voice model was NOT reloaded
    assert tts._voice is voice_before
    assert tts._output_device == "HDMI Out"
    assert tts.is_loaded


async def test_rewire_output_before_load_stores_device():
    """Calling rewire_output() before load() just stores the device for
    the next load() call — no stream operations, no error."""
    tts = PiperTTS(output_device="Pebble Pro")
    await tts.rewire_output("HDMI Out")
    assert tts._output_device == "HDMI Out"
    assert not tts.is_loaded


# --- output sample-rate / native-rate handling -----------------------


async def test_attempts_native_rate_first_with_wasapi_shared_mode(tmp_path):
    """The first open attempt uses the device's reported native sample rate
    (48 kHz here) with WASAPI shared mode. Native rate always succeeds when
    the device is reachable; this avoids driver failures seen when 22050 Hz
    was tried first on devices that only accept their native rate."""
    devices = [
        {"name": "USB DAC", "max_output_channels": 2, "hostapi": 1,
         "default_samplerate": 48000.0},
    ]
    hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")
    fake_stream = _make_fake_stream()

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    sentinel_settings = object()
    open_calls: list[dict] = []

    def raw_output_stream(**kwargs):
        open_calls.append(kwargs)
        return fake_stream

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.side_effect = raw_output_stream
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = hostapis
        sd_mock.WasapiSettings.return_value = sentinel_settings
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="USB DAC",
        )
        await tts.load()
    assert len(open_calls) == 1, "second attempt should not happen on success"
    first = open_calls[0]
    assert first["samplerate"] == 48000   # native rate first
    assert first["device"] == 0
    assert first.get("extra_settings") is sentinel_settings
    sd_mock.WasapiSettings.assert_called_once_with(exclusive=False)


async def test_falls_back_to_piper_rate_when_native_open_fails(tmp_path, time_compress_off):
    """If the driver refuses its reported native rate, the wrapper falls back
    to Piper's synthesis rate (22050 Hz). Both attempts go through the same
    device index — only the rate (and extra_settings) differ."""
    devices = [
        {"name": "USB DAC", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
    ]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")
    chunk = (np.full(2205, 16000, dtype=np.int16)).tobytes()  # 0.1 s @ 22050
    fake_stream = _make_fake_stream()
    fake_voice = _make_fake_voice(sample_rate=22050)
    fake_voice.synthesize.return_value = iter([_audio_chunk(chunk)])

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    open_calls: list[dict] = []

    def raw_output_stream(**kwargs):
        open_calls.append(kwargs)
        if kwargs["samplerate"] == 48000:
            raise RuntimeError("Invalid sample rate [PaErrorCode -9997]")
        return fake_stream

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = fake_voice
        sd_mock.RawOutputStream.side_effect = raw_output_stream
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="USB DAC",
        )
        await tts.load()
        drainer_stop = _install_callback_drainer(tts)
        try:
            await tts.speak("hi")
        finally:
            drainer_stop.set()
    # native (48000) tried first and rejected; piper rate (22050) succeeds
    assert [c["samplerate"] for c in open_calls] == [48000, 22050]
    assert tts._output_rate == 22050


async def test_native_rate_match_skips_resample(tmp_path, time_compress_off):
    """When the output device's native rate equals the Piper rate, the bytes
    written to the stream are exactly the synthesized bytes (no resample)."""
    devices = [
        {"name": "Native22k", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 22050.0},
    ]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")
    chunk = b"\x10\x20" * 500
    fake_stream = _make_fake_stream()
    fake_voice = _make_fake_voice(sample_rate=22050)
    fake_voice.synthesize.return_value = iter([_audio_chunk(chunk)])

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = fake_voice
        sd_mock.RawOutputStream.return_value = fake_stream
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="Native22k",
        )
        await tts.load()
        drainer_stop = _install_callback_drainer(tts)
        try:
            await tts.speak("hi")
        finally:
            drainer_stop.set()
    all_written = b"".join(c.args[0] for c in fake_stream.write.call_args_list)
    # Strip silence padding from both sides: the drainer thread starts the
    # moment _install_callback_drainer returns and may pull several silent
    # callbacks before speak() actually queues audio. The chunk content
    # contains no internal zero bytes so .strip(b"\x00") is safe.
    assert all_written.strip(b"\x00") == chunk


async def test_wasapi_open_failure_falls_back_to_mme(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """If WASAPI refuses every rate we try (22050 then native), the MME
    variant of the same device name is tried as fallback. Mirrors
    AudioInputSource's WASAPI->MME fallback."""
    devices = [
        {"name": "Pebble Pro (MME)", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 44100.0},
        {"name": "Pebble Pro (WASAPI)", "max_output_channels": 2, "hostapi": 1,
         "default_samplerate": 48000.0},
    ]
    hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")
    fake_stream = _make_fake_stream()
    fake_voice = _make_fake_voice(sample_rate=22050)

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    open_calls: list[dict] = []

    def raw_output_stream(**kwargs):
        open_calls.append(kwargs)
        if kwargs["device"] == 1:  # WASAPI variant — refuse every rate
            raise RuntimeError("Invalid sample rate [PaErrorCode -9997]")
        return fake_stream

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = fake_voice
        sd_mock.RawOutputStream.side_effect = raw_output_stream
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = hostapis
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="Pebble Pro",
        )
        with caplog.at_level(logging.WARNING, logger="jarvis.audio.tts"):
            await tts.load()
    # WASAPI device (idx 1) is tried at native rate (48000) then Piper rate
    # (22050) and refuses both. MME variant (idx 0) accepts the very first
    # attempt at its native rate (44100) — no Piper-rate fallback needed.
    devices_tried = [c["device"] for c in open_calls]
    assert devices_tried[:2] == [1, 1]
    assert devices_tried[-1] == 0
    assert any("MME fallback" in r.message for r in caplog.records)


async def test_boot_announces_resample_when_native_rate_differs(
    tmp_path, capsys: pytest.CaptureFixture[str]
):
    """Boot diagnostic goes to stdout (print), not the logger. When the
    device's native rate (48000 Hz) differs from Piper's synthesis rate
    (22050 Hz), native rate is opened first and the resample is announced."""
    devices = [
        {"name": "USB DAC", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
    ]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.return_value = _make_fake_stream()
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="USB DAC",
        )
        await tts.load()
    out = capsys.readouterr().out
    assert "[boot] output opened at 48000Hz, resampling from 22050Hz" in out


async def test_boot_announces_no_resample_when_native_equals_piper_rate(
    tmp_path, capsys: pytest.CaptureFixture[str]
):
    """When the device's native rate equals Piper's synthesis rate (both
    22050 Hz), only one open attempt is made and the boot line reports no
    resample."""
    devices = [
        {"name": "USB DAC", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 22050.0},
    ]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.return_value = _make_fake_stream()
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="USB DAC",
        )
        await tts.load()
    out = capsys.readouterr().out
    assert "[boot] output opened at 22050Hz (no resample)" in out


# --- speak ------------------------------------------------------------


def test_time_compress_factor_default_is_off():
    """Default TIME_COMPRESS_FACTOR is None. scipy.signal.resample_poly
    speeds up AND raises pitch — at any audible compression the pitch
    shift was worse than slow speech. Speed is tuned via Piper's
    length_scale (SPEECH_LENGTH_SCALE) instead. Tripwire if someone
    flips this back on without a pitch-neutral implementation."""
    from jarvis.audio.tts import TIME_COMPRESS_FACTOR
    assert TIME_COMPRESS_FACTOR is None


async def test_time_compress_helper_still_works_when_enabled(monkeypatch):
    """The helper stays in the tree (the constant is the only gate)
    so a future WSOLA-replacement keeps the wiring point. Verifies the
    shrink ratio when explicitly enabled."""
    monkeypatch.setattr("jarvis.audio.tts.TIME_COMPRESS_FACTOR", 1.30)
    from jarvis.audio.tts import PiperTTS
    tts = PiperTTS()
    samples = np.zeros(1300, dtype=np.int16)
    samples[::10] = 16000  # non-zero so resample has signal to chew on
    out_bytes = tts._maybe_time_compress(samples.tobytes())
    out_samples = np.frombuffer(out_bytes, dtype=np.int16)
    expected = int(1300 * 10 / 13)  # 1000
    assert abs(len(out_samples) - expected) <= 2


async def test_first_synth_duration_diagnostic_emits_once(capsys):
    """The diagnostic compares actual synthesised audio duration against
    the expected default-scale baseline so a Piper version that ignores
    SynthesisConfig.length_scale is visible at runtime. Fires once per
    instance to avoid spamming."""
    # 22050 Hz int16: 44100 bytes/sec.
    chunk = b"\x00\x00" * 22050  # exactly 1 second of audio
    tts, _, _ = _loaded_tts(voice_chunks=[chunk])
    await tts.speak("hello world from the first synth")
    out = capsys.readouterr().out
    assert "[tts-debug] first-synth" in out
    assert "ratio_vs_default=" in out
    # Second speak() must NOT re-emit the diagnostic.
    capsys.readouterr()  # drain
    await tts.speak("the second utterance, please.")
    assert "[tts-debug]" not in capsys.readouterr().out


async def test_speak_starts_stream_and_audio_reaches_callback(time_compress_off):
    """Per-chunk byte content reaches the PortAudio callback via the queue,
    in order. The accumulator merges small chunks into ~OUTPUT_BUFFER_MS
    blocks before they hit the queue; concatenating callback-pulled bytes
    must equal the concatenated synthesized chunks."""
    audio_chunks = [b"\x10" * 200, b"\x20" * 200, b"\x30" * 200]
    tts, _, stream = _loaded_tts(voice_chunks=audio_chunks)
    await tts.speak("hello world")
    stream.start.assert_called()
    played = _drain_via_callback(tts)
    assert played == b"".join(audio_chunks)


async def test_small_chunks_accumulate_into_fewer_queue_blocks(time_compress_off):
    """Forwarding each Piper chunk verbatim to OutputStream starves
    PortAudio between writes (audible as crackling). The accumulator is the
    fix: many small synthesizer chunks should produce far fewer (and larger)
    queue blocks for the callback to consume. With OUTPUT_BUFFER_MS=200 at
    22050 Hz the target is ~8820 bytes — twenty 100-byte chunks (2000 bytes
    total) all coalesce into ONE final-flush block, but the bytes must
    still arrive in order."""
    from jarvis.audio.protocols import OUTPUT_BUFFER_MS
    chunks = [bytes([i & 0xFF]) * 100 for i in range(20)]
    tts, _, _ = _loaded_tts(voice_chunks=chunks)
    await tts.speak("hello")
    assert tts._audio_queue.qsize() == 1
    assert tts._audio_queue.qsize() < len(chunks)
    played = _drain_via_callback(tts)
    assert played == b"".join(chunks)
    # And the constant is the one driving the behaviour (defensive: stays
    # in [50, 1000] ms — anything outside is almost certainly a typo).
    assert 50 <= OUTPUT_BUFFER_MS <= 1000


async def test_large_input_flushes_at_target_boundary(time_compress_off):
    """When accumulated audio exceeds the OUTPUT_BUFFER_MS target, a queue
    block is published immediately rather than waiting for end-of-speak.
    Verifies the in-loop flush path (not just the final-flush path)."""
    from jarvis.audio.protocols import OUTPUT_BUFFER_MS
    sample_rate = 22050
    target_bytes = (OUTPUT_BUFFER_MS * sample_rate * 2) // 1000  # 8820
    chunks = [b"\xab" * (target_bytes + 100) for _ in range(3)]
    tts, _, _ = _loaded_tts(voice_chunks=chunks, sample_rate=sample_rate)
    await tts.speak("hi")
    # Three over-target chunks -> at least 3 queue blocks (final-flush may
    # add one more if there's a remainder past the boundary).
    assert tts._audio_queue.qsize() >= 3


async def test_speak_empty_text_short_circuits():
    tts, voice, stream = _loaded_tts()
    await tts.speak("")
    voice.synthesize.assert_not_called()
    stream.start.assert_not_called()


async def test_speak_whitespace_only_short_circuits():
    tts, voice, _ = _loaded_tts()
    await tts.speak("   \n  ")
    voice.synthesize.assert_not_called()


async def test_speak_before_load_logs_warning(caplog: pytest.LogCaptureFixture):
    tts = PiperTTS()
    with caplog.at_level(logging.WARNING, logger="jarvis.audio.tts"):
        await tts.speak("hi")
    assert any("before load" in r.message for r in caplog.records)


async def test_speak_runs_in_thread_executor():
    tts, voice, stream = _loaded_tts()
    loop_thread = threading.current_thread()
    captured = {}

    # **kwargs absorbs syn_config (the length_scale-bearing SynthesisConfig
    # the wrapper now passes when self.speed and SPEECH_LENGTH_SCALE are in
    # effect). Without it the mock raises TypeError before the test asserts.
    def synth(text, **kwargs):
        captured["thread"] = threading.current_thread()
        return iter([_audio_chunk(b"\x00" * 100)])

    voice.synthesize.side_effect = synth
    await tts.speak("hi")
    assert captured["thread"] is not loop_thread


async def test_speak_isolates_synthesis_exception(caplog: pytest.LogCaptureFixture):
    tts, voice, _ = _loaded_tts()
    voice.synthesize.side_effect = RuntimeError("synth crash")
    with caplog.at_level(logging.ERROR, logger="jarvis.audio.tts"):
        await tts.speak("hello")
    assert any("speak failed" in r.message for r in caplog.records)


async def test_speak_restarts_stopped_stream():
    """After cancel() leaves the stream in stopped state, the next speak()
    must restart it; otherwise nothing plays."""
    tts, _, stream = _loaded_tts()
    stream.active = False  # simulate post-abort state
    await tts.speak("hi")
    stream.start.assert_called()
    assert stream.active


async def test_volume_scales_audio_samples(time_compress_off):
    # 0.5 volume should halve sample magnitudes.
    samples = np.array([1000, -2000, 3000, -4000] * 100, dtype=np.int16)
    tts, _, _ = _loaded_tts(voice_chunks=[samples.tobytes()], volume=0.5)
    await tts.speak("hi")
    played = _drain_via_callback(tts)
    out = np.frombuffer(played, dtype=np.int16)
    np.testing.assert_array_equal(out, (samples.astype(np.float32) * 0.5).astype(np.int16))


async def test_volume_one_passes_audio_through_unchanged(time_compress_off):
    payload = b"\xab\xcd" * 100
    tts, _, _ = _loaded_tts(voice_chunks=[payload], volume=1.0)
    await tts.speak("hi")
    assert _drain_via_callback(tts) == payload


# --- cancel -----------------------------------------------------------


async def test_cancel_aborts_then_closes_stream():
    """SPEC-critical: cancel() must abort() (drops buffered samples immediately,
    not stop()), then close() the stream so the next speak() opens a fresh one
    rather than reusing the aborted stream (MME PortAudioError 33 fix)."""
    tts, _, stream = _loaded_tts()
    await tts.cancel()
    stream.abort.assert_called_once()
    stream.close.assert_called_once()
    stream.stop.assert_not_called()
    assert tts._stream is None  # type: ignore[attr-defined]


async def test_cancel_breaks_speak_synthesis_loop():
    """An in-flight speak that produces many chunks must stop writing
    further chunks once cancel() is called."""
    # Many chunks; cancel mid-stream and verify partial write count.
    chunks = [bytes([i]) * 100 for i in range(20)]
    tts, voice, stream = _loaded_tts()

    chunks_yielded = []
    cancel_after = 3
    speak_done = asyncio.Event()

    def slow_synth(text, **kwargs):
        for i, c in enumerate(chunks):
            chunks_yielded.append(i)
            time.sleep(0.005)
            yield _audio_chunk(c)

    voice.synthesize.side_effect = slow_synth

    async def run_speak():
        await tts.speak("hello")
        speak_done.set()

    speak_task = asyncio.create_task(run_speak())
    # Wait for a few chunks to be processed before cancelling.
    while len(chunks_yielded) < cancel_after:
        await asyncio.sleep(0.001)
    await tts.cancel()
    await asyncio.wait_for(speak_done.wait(), timeout=2.0)
    await speak_task

    # Stream should have been aborted, and not all 20 chunks written.
    assert stream.abort.called
    assert stream.write.call_count < len(chunks)


async def test_cancel_safe_to_call_when_nothing_playing():
    tts, _, _ = _loaded_tts()
    await tts.cancel()  # must not raise


async def test_cancel_safe_before_load():
    tts = PiperTTS()
    await tts.cancel()


async def test_speak_after_cancel_works():
    """Sequential cycle: speak -> cancel -> speak. The second speak must
    succeed: cancel() closes the stream, speak() opens a fresh one."""
    tts, voice, _first_stream = _loaded_tts(voice_chunks=[b"\xff" * 100])
    new_stream = _make_fake_stream()
    voice.synthesize.return_value = iter([_audio_chunk(b"\xee" * 100)])
    with patch("jarvis.audio.tts.sd") as sd_mock:
        sd_mock.RawOutputStream.return_value = new_stream
        await tts.cancel()
        # The reopen path goes through _open_stream which wires the real
        # callback — install a drainer so speak()'s drain wait completes.
        drainer_stop = _install_callback_drainer(tts)
        try:
            await tts.speak("hello")
        finally:
            drainer_stop.set()
    new_stream.start.assert_called()
    # Drainer mirrors callback bytes to stream.write; at minimum one write
    # should have happened during synthesis.
    assert new_stream.write.called


async def test_cancel_closes_stream_speak_reopens_it():
    """cancel() closes the output stream; the next speak() must reopen it.
    Asserts both the close and the fresh-open happen correctly (the defensive
    fix for MME PortAudioError 33 after barge-in cancellation)."""
    tts, voice, first_stream = _loaded_tts(voice_chunks=[b"\xff" * 100])
    new_stream = _make_fake_stream()
    voice.synthesize.return_value = iter([_audio_chunk(b"\xee" * 100)])

    with patch("jarvis.audio.tts.sd") as sd_mock:
        sd_mock.RawOutputStream.return_value = new_stream

        await tts.cancel()
        first_stream.abort.assert_called_once()
        first_stream.close.assert_called_once()
        assert tts._stream is None  # type: ignore[attr-defined]

        drainer_stop = _install_callback_drainer(tts)
        try:
            await tts.speak("hello again")
        finally:
            drainer_stop.set()
        sd_mock.RawOutputStream.assert_called_once()
        new_stream.start.assert_called()
        assert new_stream.write.called


# --- speak_stream ----------------------------------------------------


async def _stream_from(*chunks: str, delay: float = 0.0):
    async def gen():
        for c in chunks:
            if delay:
                await asyncio.sleep(delay)
            yield c
    return gen()


async def test_speak_stream_emits_on_sentence_boundary():
    tts, _, _ = _loaded_tts()
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    tts.speak = fake_speak  # type: ignore[method-assign]

    await tts.speak_stream(await _stream_from("Hello world. ", "How are you?"))
    assert spoken == ["Hello world.", "How are you?"]


async def test_speak_stream_buffers_until_sentence_completes():
    """Mid-sentence chunk arrives, then the rest. Should emit only when
    the sentence is complete."""
    tts, _, _ = _loaded_tts()
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    tts.speak = fake_speak  # type: ignore[method-assign]

    await tts.speak_stream(
        await _stream_from("Hello ", "world", ", how", " are you?"),
    )
    assert spoken == ["Hello world, how are you?"]


async def test_speak_stream_max_wait_flushes_incomplete_sentence():
    """If the producer pauses mid-sentence past max_wait, flush whatever's
    buffered so the user isn't left in silence."""
    tts, _, _ = _loaded_tts()
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    tts.speak = fake_speak  # type: ignore[method-assign]

    async def gen():
        yield "First sentence. "
        yield "Trailing fragment without a period"
        await asyncio.sleep(0.15)  # exceeds max_wait below

    await tts.speak_stream(gen(), max_wait_seconds=0.05)
    assert spoken == ["First sentence.", "Trailing fragment without a period"]


async def test_speak_stream_flushes_remaining_on_iter_end():
    tts, _, _ = _loaded_tts()
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    tts.speak = fake_speak  # type: ignore[method-assign]

    await tts.speak_stream(await _stream_from("No terminator here"))
    assert spoken == ["No terminator here"]


async def test_speak_stream_handles_multiple_sentences_in_one_chunk():
    tts, _, _ = _loaded_tts()
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    tts.speak = fake_speak  # type: ignore[method-assign]

    await tts.speak_stream(
        await _stream_from("Sentence one. Sentence two! And three? ")
    )
    assert spoken == ["Sentence one.", "Sentence two!", "And three?"]


async def test_speak_stream_empty_stream_does_nothing():
    tts, _, _ = _loaded_tts()
    spoken: list[str] = []
    tts.speak = lambda t: spoken.append(t) or asyncio.sleep(0)  # type: ignore[method-assign,assignment]

    async def empty():
        return
        yield  # unreachable; makes this an async generator

    await tts.speak_stream(empty())
    assert spoken == []


# --- amplitude envelope ---------------------------------------------


def test_amplitude_from_samples_expands_dynamic_range():
    from jarvis.audio.tts import amplitude_from_samples

    quiet = np.full(735, 2000, dtype=np.int16)
    loud = np.full(735, 16000, dtype=np.int16)
    a_quiet = amplitude_from_samples(quiet)
    a_loud = amplitude_from_samples(loud)
    linear_ratio = (2000 / 32768.0) / (16000 / 32768.0)
    assert (a_loud - a_quiet) > linear_ratio * 0.5
    assert a_loud > a_quiet


async def test_amplitude_callback_invoked_per_window():
    """At 22050 Hz / 30 Hz envelope, window ~= 735 samples. A single 1470-
    sample (3000-byte) chunk should produce 2 envelope samples."""
    sample_rate = 22050
    samples_per_chunk = (sample_rate // 30) * 2  # exactly 2 windows
    chunk = (np.full(samples_per_chunk, 16000, dtype=np.int16)).tobytes()

    amplitudes: list[float] = []
    tts, _, _ = _loaded_tts(
        voice_chunks=[chunk],
        on_amplitude=lambda a: amplitudes.append(a),
        sample_rate=sample_rate,
    )
    await tts.speak("hi")
    assert len(amplitudes) == 2
    from jarvis.audio.tts import amplitude_from_samples

    expected = amplitude_from_samples(
        np.frombuffer(chunk, dtype=np.int16)[: sample_rate // 30]
    )
    for a in amplitudes:
        assert a == pytest.approx(expected, abs=1e-3)


async def test_amplitude_silence_yields_zero_amplitude():
    sample_rate = 22050
    silent_chunk = (np.zeros(sample_rate // 30, dtype=np.int16)).tobytes()
    amplitudes: list[float] = []
    tts, _, _ = _loaded_tts(
        voice_chunks=[silent_chunk],
        on_amplitude=lambda a: amplitudes.append(a),
        sample_rate=sample_rate,
    )
    await tts.speak("hi")
    assert amplitudes
    assert all(a == 0.0 for a in amplitudes)


async def test_amplitude_callback_isolates_exception(caplog: pytest.LogCaptureFixture):
    def bad(amp):
        raise RuntimeError("nope")

    chunk = (np.full(1000, 16000, dtype=np.int16)).tobytes()
    tts, _, _ = _loaded_tts(voice_chunks=[chunk], on_amplitude=bad)
    with caplog.at_level(logging.ERROR, logger="jarvis.audio.tts"):
        await tts.speak("hi")
    assert any("on_amplitude callback raised" in r.message for r in caplog.records)


async def test_no_amplitude_callback_no_calls():
    """Sanity: with no on_amplitude, _emit is a no-op and no error fires."""
    chunk = (np.full(1000, 16000, dtype=np.int16)).tobytes()
    tts, _, _ = _loaded_tts(voice_chunks=[chunk], on_amplitude=None)
    await tts.speak("hi")  # must not raise


# --- fixture-based real-voice test ---------------------------------


@pytest.mark.skipif(
    not (VOICES_FIXTURE_DIR / f"{REAL_VOICE_NAME}.onnx").exists()
    or not (VOICES_FIXTURE_DIR / f"{REAL_VOICE_NAME}.onnx.json").exists(),
    reason=(
        f"commit {REAL_VOICE_NAME}.onnx and .onnx.json under {VOICES_FIXTURE_DIR} "
        "to enable real-voice synthesis test"
    ),
)
async def test_real_piper_synthesizes_audio():
    """Smoke test against the real Piper voice. Captures audio bytes
    instead of playing them so the test is silent in CI."""
    written: list[bytes] = []
    fake_stream = _make_fake_stream()
    fake_stream.write.side_effect = lambda b: written.append(bytes(b))

    with patch("jarvis.audio.tts.sd") as sd_mock:
        sd_mock.RawOutputStream.return_value = fake_stream
        tts = PiperTTS(voice_name=REAL_VOICE_NAME, voices_dir=VOICES_FIXTURE_DIR)
        try:
            await tts.load()
        except TTSLoadError:
            pytest.skip("piper voice file present but load failed")
        try:
            await tts.speak("hello sir")
            total_bytes = sum(len(b) for b in written)
            assert total_bytes > 0, "real synthesis produced no audio bytes"
        finally:
            await tts.unload()


# --- device fallback (Phase 2) ----------------------------------------


async def test_output_fallback_to_secondary_device_on_primary_failure(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """Primary device fails → secondary succeeds → warning logged, NonFatalError published."""
    from unittest.mock import MagicMock
    from jarvis.core.events import EventBus, NonFatalError

    devices = [
        {"name": "BT Headset", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
        {"name": "Pebble Pro", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
    ]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")
    fake_stream = _make_fake_stream()

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    def raw_output_stream(**kwargs):
        if kwargs["device"] == 0:
            raise OSError("device busy")
        return fake_stream

    bus = MagicMock(spec=EventBus)
    published: list = []
    bus.publish.side_effect = published.append

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.side_effect = raw_output_stream
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="BT Headset",
            bus=bus,
        )
        with caplog.at_level(logging.WARNING, logger="jarvis.audio.tts"):
            await tts.load()

    assert tts.is_loaded
    # Warning logged
    assert any("fallback" in r.message for r in caplog.records)
    # NonFatalError published on bus
    assert len(published) == 1
    evt = published[0]
    assert isinstance(evt, NonFatalError)
    assert evt.module == "tts"
    assert evt.issue == "device_fallback"
    assert "BT Headset" in evt.expected
    assert "Pebble Pro" in evt.actual


async def test_output_all_devices_fail_raises_tts_load_error(tmp_path):
    """When every available output device fails, TTSLoadError is raised."""
    devices = [
        {"name": "Dev A", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
        {"name": "Dev B", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
    ]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.side_effect = OSError("all broken")
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="Dev A",
        )
        with pytest.raises(TTSLoadError):
            await tts.load()
    assert not tts.is_loaded


async def test_output_primary_succeeds_no_fallback_triggered(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """Primary device opens on first try — no fallback code path, no warning."""
    from unittest.mock import MagicMock
    from jarvis.core.events import EventBus

    devices = [
        {"name": "Pebble Pro", "max_output_channels": 2, "hostapi": 0,
         "default_samplerate": 48000.0},
    ]
    (tmp_path / f"{REAL_VOICE_NAME}.onnx").write_bytes(b"")
    (tmp_path / f"{REAL_VOICE_NAME}.onnx.json").write_text("{}")
    fake_stream = _make_fake_stream()

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return devices
        return devices[idx]

    bus = MagicMock(spec=EventBus)

    with patch("piper.PiperVoice") as voice_cls, \
         patch("jarvis.audio.tts.sd") as sd_mock:
        voice_cls.load.return_value = _make_fake_voice()
        sd_mock.RawOutputStream.return_value = fake_stream
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        tts = PiperTTS(
            voice_name=REAL_VOICE_NAME,
            voices_dir=tmp_path,
            output_device="Pebble Pro",
            bus=bus,
        )
        with caplog.at_level(logging.WARNING, logger="jarvis.audio.tts"):
            await tts.load()

    assert tts.is_loaded
    assert not any("fallback" in r.message for r in caplog.records)
    bus.publish.assert_not_called()
