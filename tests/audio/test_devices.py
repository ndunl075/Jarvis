"""Tests for jarvis.audio.devices.AudioInputSource. sounddevice is fully
stubbed; no real audio hardware is touched."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from jarvis.audio.devices import AudioInputSource, DeviceOpenError
from jarvis.audio.protocols import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AudioSource,
)
from jarvis.core.lifecycle import Loadable


def _device(name: str, max_input_channels: int = 2, hostapi: int = 0) -> dict:
    return {"name": name, "max_input_channels": max_input_channels, "hostapi": hostapi}


@pytest.fixture
def patched_sd():
    with patch("jarvis.audio.devices.sd") as sd_mock:
        sd_mock.RawInputStream = MagicMock()
        yield sd_mock


def _stream_kwargs(patched_sd):
    return patched_sd.RawInputStream.call_args.kwargs


# --- protocol smoke ------------------------------------------------------


def test_implements_audio_source_and_loadable():
    src = AudioInputSource()
    assert isinstance(src, AudioSource)
    assert isinstance(src, Loadable)


# --- device selection ----------------------------------------------------


async def test_picks_respeaker_when_present(patched_sd):
    patched_sd.query_devices.return_value = [
        _device("Default Mic"),
        _device("ReSpeaker 4 Mic Array"),
        _device("Speakers", max_input_channels=0),
    ]
    src = AudioInputSource(prefer_respeaker=True)
    await src.load()
    assert _stream_kwargs(patched_sd)["device"] == 1


async def test_picks_seeed_case_insensitive(patched_sd):
    patched_sd.query_devices.return_value = [
        _device("Default Mic"),
        _device("SEEED-VoiceCard"),
    ]
    src = AudioInputSource(prefer_respeaker=True)
    await src.load()
    assert _stream_kwargs(patched_sd)["device"] == 1


async def test_skips_output_only_devices(patched_sd):
    # An output-only device named "ReSpeaker Speakers" must not match.
    patched_sd.query_devices.return_value = [
        _device("ReSpeaker Speakers", max_input_channels=0),
        _device("Default Mic"),
    ]
    src = AudioInputSource(prefer_respeaker=True)
    await src.load()
    assert _stream_kwargs(patched_sd)["device"] is None


async def test_falls_back_to_default_when_no_respeaker(patched_sd):
    patched_sd.query_devices.return_value = [
        _device("Default Mic"),
        _device("USB Mic"),
    ]
    src = AudioInputSource(prefer_respeaker=True)
    await src.load()
    assert _stream_kwargs(patched_sd)["device"] is None


async def test_picks_configured_device_by_name_substring(patched_sd):
    patched_sd.query_devices.return_value = [
        _device("Default Mic"),
        _device("Studio Condenser USB"),
        _device("ReSpeaker 4 Mic Array"),
    ]
    src = AudioInputSource(preferred_device="studio")
    await src.load()
    assert _stream_kwargs(patched_sd)["device"] == 1


async def test_picks_wasapi_variant_over_mme(patched_sd):
    """When both MME and WASAPI input devices match the substring, WASAPI is
    preferred to avoid the stream-restart PortAudioError 33 after barge-in."""
    patched_sd.query_hostapis.return_value = [
        {"name": "MME"},
        {"name": "Windows WASAPI"},
    ]
    patched_sd.query_devices.return_value = [
        _device("HyperX SoloCast", hostapi=0),           # MME variant
        _device("HyperX SoloCast", hostapi=1),           # WASAPI variant
    ]
    src = AudioInputSource(preferred_device="hyperx")
    await src.load()
    assert _stream_kwargs(patched_sd)["device"] == 1  # WASAPI variant


async def test_configured_device_not_found_falls_back_to_default(
    patched_sd, caplog
):
    patched_sd.query_devices.return_value = [
        _device("Default Mic"),
        _device("ReSpeaker 4 Mic Array"),
    ]
    src = AudioInputSource(preferred_device="ghost mic", prefer_respeaker=True)
    with caplog.at_level(logging.WARNING, logger="jarvis.audio.devices"):
        await src.load()
    # Explicit mis-set must NOT silently fall through to ReSpeaker auto-detect.
    assert _stream_kwargs(patched_sd)["device"] is None
    assert any("not found" in r.message for r in caplog.records)


async def test_prefer_respeaker_disabled_uses_default(patched_sd):
    patched_sd.query_devices.return_value = [
        _device("Default Mic"),
        _device("ReSpeaker 4 Mic Array"),
    ]
    src = AudioInputSource(prefer_respeaker=False)
    await src.load()
    assert _stream_kwargs(patched_sd)["device"] is None


# --- stream configuration ------------------------------------------------


async def test_stream_opened_with_spec_format(patched_sd):
    patched_sd.query_devices.return_value = [_device("Default Mic")]
    src = AudioInputSource(prefer_respeaker=False)
    await src.load()
    kwargs = _stream_kwargs(patched_sd)
    assert kwargs["samplerate"] == SAMPLE_RATE
    assert kwargs["channels"] == 1
    assert kwargs["dtype"] == "int16"
    assert kwargs["blocksize"] == FRAME_SAMPLES


async def test_open_failure_raises_clear_error(patched_sd):
    patched_sd.query_devices.return_value = [_device("Default Mic")]
    patched_sd.RawInputStream.side_effect = OSError("device unavailable")
    src = AudioInputSource(prefer_respeaker=False)
    with pytest.raises(DeviceOpenError) as exc_info:
        await src.load()
    assert "16000" in str(exc_info.value)
    assert not src.is_loaded


# --- Loadable lifecycle --------------------------------------------------


async def test_load_unload_idempotent(patched_sd):
    patched_sd.query_devices.return_value = [_device("Default Mic")]
    stream = MagicMock()
    patched_sd.RawInputStream.return_value = stream
    src = AudioInputSource(prefer_respeaker=False)

    await src.load()
    await src.load()  # second load is a no-op
    assert patched_sd.RawInputStream.call_count == 1
    assert src.is_loaded
    stream.start.assert_called_once()

    await src.unload()
    await src.unload()  # second unload is a no-op
    assert stream.stop.call_count == 1
    assert stream.close.call_count == 1
    assert not src.is_loaded


async def test_unload_releases_callback_reference(patched_sd):
    patched_sd.query_devices.return_value = [_device("Default Mic")]
    patched_sd.RawInputStream.return_value = MagicMock()
    src = AudioInputSource(prefer_respeaker=False)
    await src.load()
    await src.start(lambda f: None)
    await src.unload()
    assert src._on_frame is None  # type: ignore[attr-defined]


# --- AudioSource lifecycle ----------------------------------------------


async def test_source_start_before_load_raises(patched_sd):
    src = AudioInputSource()
    with pytest.raises(RuntimeError, match="before load"):
        await src.start(lambda f: None)


async def test_callback_routes_frames_to_consumer(patched_sd):
    patched_sd.query_devices.return_value = [_device("Default Mic")]
    patched_sd.RawInputStream.return_value = MagicMock()
    src = AudioInputSource(prefer_respeaker=False)
    await src.load()

    received: list[bytes] = []
    await src.start(lambda f: received.append(f))

    sd_callback = _stream_kwargs(patched_sd)["callback"]
    sd_callback(b"\xab" * FRAME_BYTES, FRAME_SAMPLES, None, None)
    assert received == [b"\xab" * FRAME_BYTES]


async def test_source_stop_detaches_callback_without_closing_stream(patched_sd):
    patched_sd.query_devices.return_value = [_device("Default Mic")]
    stream = MagicMock()
    patched_sd.RawInputStream.return_value = stream
    src = AudioInputSource(prefer_respeaker=False)
    await src.load()

    received: list[bytes] = []
    await src.start(lambda f: received.append(f))
    sd_callback = _stream_kwargs(patched_sd)["callback"]

    await src.stop()
    sd_callback(b"\xcd" * FRAME_BYTES, FRAME_SAMPLES, None, None)
    assert received == []  # detached
    stream.stop.assert_not_called()  # stream still open
    stream.close.assert_not_called()


async def test_callback_drops_wrong_size_frames(patched_sd, caplog):
    patched_sd.query_devices.return_value = [_device("Default Mic")]
    patched_sd.RawInputStream.return_value = MagicMock()
    src = AudioInputSource(prefer_respeaker=False)
    await src.load()
    received: list[bytes] = []
    await src.start(lambda f: received.append(f))
    sd_callback = _stream_kwargs(patched_sd)["callback"]
    with caplog.at_level(logging.ERROR, logger="jarvis.audio.devices"):
        sd_callback(b"\x00" * 50, 25, None, None)
    assert received == []
    assert any("wrong frame size" in r.message for r in caplog.records)


async def test_callback_isolates_consumer_exception(patched_sd, caplog):
    patched_sd.query_devices.return_value = [_device("Default Mic")]
    patched_sd.RawInputStream.return_value = MagicMock()
    src = AudioInputSource(prefer_respeaker=False)
    await src.load()

    def bad(_):
        raise ValueError("nope")

    await src.start(bad)
    sd_callback = _stream_kwargs(patched_sd)["callback"]
    with caplog.at_level(logging.ERROR, logger="jarvis.audio.devices"):
        sd_callback(b"\x00" * FRAME_BYTES, FRAME_SAMPLES, None, None)
    assert any("on_frame callback raised" in r.message for r in caplog.records)


async def test_callback_with_no_consumer_attached_is_noop(patched_sd):
    patched_sd.query_devices.return_value = [_device("Default Mic")]
    patched_sd.RawInputStream.return_value = MagicMock()
    src = AudioInputSource(prefer_respeaker=False)
    await src.load()
    sd_callback = _stream_kwargs(patched_sd)["callback"]
    # Must not raise; just discards the frame.
    sd_callback(b"\x00" * FRAME_BYTES, FRAME_SAMPLES, None, None)


# --- device fallback (Phase 2) -------------------------------------------


async def test_input_fallback_to_secondary_device_on_primary_failure(caplog):
    """Primary device fails → secondary succeeds → warning logged, NonFatalError published."""
    from unittest.mock import MagicMock

    from jarvis.core.events import EventBus, NonFatalError

    call_count = [0]
    fallback_stream = MagicMock()

    def raw_input_stream(**kwargs):
        call_count[0] += 1
        if kwargs["device"] == 0:
            raise OSError("device busy")
        fallback_stream.start = MagicMock()
        return fallback_stream

    bus = MagicMock(spec=EventBus)
    published: list = []
    bus.publish.side_effect = published.append

    all_devices = [
        _device("TONOR USB", hostapi=0),
        _device("USB Mic 2", hostapi=0),
    ]

    def query_devices(idx=None, *args, **kwargs):
        if idx is None:
            return all_devices
        return all_devices[idx]

    with patch("jarvis.audio.devices.sd") as sd_mock:
        sd_mock.RawInputStream.side_effect = raw_input_stream
        sd_mock.query_devices.side_effect = query_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        src = AudioInputSource(preferred_device="TONOR", prefer_respeaker=False, bus=bus)
        with caplog.at_level(logging.WARNING, logger="jarvis.audio.devices"):
            await src.load()

    assert src.is_loaded
    assert any("fallback" in r.message for r in caplog.records)
    assert len(published) == 1
    evt = published[0]
    assert isinstance(evt, NonFatalError)
    assert evt.module == "audio_input"
    assert evt.issue == "device_fallback"
    assert "TONOR" in evt.expected
    assert "USB Mic 2" in evt.actual


async def test_input_all_devices_fail_raises_device_open_error():
    """When every available input device fails, DeviceOpenError is raised."""
    all_devices = [
        _device("Dev A", hostapi=0),
        _device("Dev B", hostapi=0),
    ]

    with patch("jarvis.audio.devices.sd") as sd_mock:
        sd_mock.RawInputStream.side_effect = OSError("all broken")
        sd_mock.query_devices.return_value = all_devices
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        src = AudioInputSource(preferred_device="Dev A", prefer_respeaker=False)
        with pytest.raises(DeviceOpenError):
            await src.load()
    assert not src.is_loaded


async def test_input_primary_succeeds_no_fallback_triggered(caplog):
    """Primary device opens on first try — no fallback code path, no warning."""
    from unittest.mock import MagicMock

    from jarvis.core.events import EventBus

    fake_stream = MagicMock()
    fake_stream.start = MagicMock()

    bus = MagicMock(spec=EventBus)

    with patch("jarvis.audio.devices.sd") as sd_mock:
        sd_mock.RawInputStream.return_value = fake_stream
        sd_mock.query_devices.return_value = [_device("TONOR USB", hostapi=0)]
        sd_mock.query_hostapis.return_value = [{"name": "MME"}]
        src = AudioInputSource(preferred_device="TONOR", prefer_respeaker=False, bus=bus)
        with caplog.at_level(logging.WARNING, logger="jarvis.audio.devices"):
            await src.load()

    assert src.is_loaded
    assert not any("fallback" in r.message for r in caplog.records)
    bus.publish.assert_not_called()


async def test_native_48k_device_resamples_to_16k_in_callback(patched_sd):
    """USB mics commonly only support 44.1k/48k natively under WASAPI. Open
    at the device's native rate, then downsample to 16k inside the callback
    so downstream still sees the SPEC frame format (480 samples / 960 bytes)."""
    import numpy as np

    devices = [
        {
            "name": "TONOR TM20",
            "max_input_channels": 1,
            "hostapi": 1,
            "default_samplerate": 48000.0,
        },
    ]
    hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    patched_sd.query_devices.side_effect = (
        lambda *args: devices[args[0]] if args else devices
    )
    patched_sd.query_hostapis.return_value = hostapis

    src = AudioInputSource(preferred_device="TONOR", prefer_respeaker=False)
    await src.load()

    kwargs = _stream_kwargs(patched_sd)
    assert kwargs["samplerate"] == 48000        # opened at native rate
    assert kwargs["blocksize"] == 1440          # 480 * 48000 / 16000
    assert kwargs["device"] == 0                # WASAPI variant

    received: list[bytes] = []
    await src.start(lambda f: received.append(f))
    sd_callback = kwargs["callback"]

    # Feed one native-rate block (1440 int16 samples = 2880 bytes).
    in_samples = (np.full(1440, 12345, dtype=np.int16)).tobytes()
    sd_callback(in_samples, 1440, None, None)

    assert len(received) == 1
    assert len(received[0]) == FRAME_BYTES   # exactly 480 samples / 960 bytes
    out_samples = np.frombuffer(received[0], dtype=np.int16)
    assert len(out_samples) == FRAME_SAMPLES
    # Constant input → constant output (within rounding).
    assert np.all(np.abs(out_samples.astype(np.int32) - 12345) <= 1)


async def test_falls_back_to_mme_when_wasapi_open_fails(patched_sd, caplog):
    """If opening the WASAPI variant raises (e.g. driver refuses native rate),
    the loader retries with the MME variant of the same device before giving up."""
    devices = [
        {"name": "TONOR TM20", "max_input_channels": 1, "hostapi": 0,
         "default_samplerate": 44100.0},   # MME
        {"name": "TONOR TM20", "max_input_channels": 1, "hostapi": 1,
         "default_samplerate": 48000.0},   # WASAPI
    ]
    hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    patched_sd.query_devices.side_effect = (
        lambda *args: devices[args[0]] if args else devices
    )
    patched_sd.query_hostapis.return_value = hostapis

    wasapi_stream = MagicMock()
    wasapi_stream.start.side_effect = OSError("WASAPI rate unsupported")
    mme_stream = MagicMock()
    patched_sd.RawInputStream.side_effect = [wasapi_stream, mme_stream]

    src = AudioInputSource(preferred_device="TONOR", prefer_respeaker=False)
    with caplog.at_level(logging.WARNING, logger="jarvis.audio.devices"):
        await src.load()

    # Two open attempts: WASAPI first (idx=1), MME fallback (idx=0).
    devices_tried = [c.kwargs["device"] for c in patched_sd.RawInputStream.call_args_list]
    assert devices_tried == [1, 0]
    assert any("MME fallback" in r.message for r in caplog.records)


async def test_status_warning_logged(patched_sd, caplog):
    patched_sd.query_devices.return_value = [_device("Default Mic")]
    patched_sd.RawInputStream.return_value = MagicMock()
    src = AudioInputSource(prefer_respeaker=False)
    await src.load()
    sd_callback = _stream_kwargs(patched_sd)["callback"]
    with caplog.at_level(logging.WARNING, logger="jarvis.audio.devices"):
        sd_callback(b"\x00" * FRAME_BYTES, FRAME_SAMPLES, None, "input overflow")
    assert any("sounddevice status" in r.message for r in caplog.records)
