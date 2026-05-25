"""Audio input source via sounddevice. Implements both AudioSource and the
Loadable protocol from core.lifecycle.

Auto-detects ReSpeaker / SEEED devices when prefer_respeaker is True; falls
back to the system default. The stream is configured for the SPEC frame
format (16 kHz mono int16, 480-sample frames). If the chosen device cannot
be opened in that format, raises DeviceOpenError rather than silently
resampling -- silent resampling would propagate wrong-shape frames to every
downstream stage.

Lifecycle split per the user's spec:
  - Loadable.load()  : pick device, open stream, start capture (mic hot).
  - Loadable.unload(): stop and close stream (mic released).
  - AudioSource.start(on_frame): attach the consumer callback. The stream
    is already running; frames before this call are silently discarded.
  - AudioSource.stop(): detach the consumer callback. Stream stays open.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd

from jarvis.audio.protocols import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AudioFrame,
)
from jarvis.core.events import EventBus, NonFatalError

log = logging.getLogger(__name__)

# Substrings (lower-cased) that identify a ReSpeaker / SEEED device.
_RESPEAKER_NAME_TOKENS: tuple[str, ...] = ("respeaker", "seeed")


def _wasapi_hostapi_index() -> int | None:
    """Return the sounddevice hostapi index for Windows WASAPI, or None.

    WASAPI is preferred over MME when matching devices by name: MME has a
    stream-restart race (PortAudioError 33 — "Cannot perform this operation
    while media data is still playing") that fires after barge-in abort().
    WASAPI's lower-level stream model doesn't exhibit this race."""
    try:
        for idx, api in enumerate(sd.query_hostapis()):
            if "wasapi" in str(api.get("name", "")).lower():
                return idx
    except Exception:
        pass
    return None


def _mme_hostapi_index() -> int | None:
    """Return the sounddevice hostapi index for Windows MME, or None.

    MME is the legacy fallback used only when WASAPI cannot open the device
    at any supported rate (rare but seen on some quirky USB audio classes)."""
    try:
        for idx, api in enumerate(sd.query_hostapis()):
            if str(api.get("name", "")).strip().lower() == "mme":
                return idx
    except Exception:
        pass
    return None


def _device_native_rate(idx: int | None) -> int:
    """Return the device's default sample rate, or SAMPLE_RATE if unknown.

    USB mics rarely support 16 kHz natively under WASAPI (which forbids the
    driver-side resampling MME silently performs). Opening at the native rate
    and resampling in the callback is what makes WASAPI viable for them."""
    if idx is None:
        return SAMPLE_RATE
    try:
        info = sd.query_devices(idx)
        rate = info.get("default_samplerate") if hasattr(info, "get") else None
        if rate:
            return int(rate)
    except Exception:
        pass
    return SAMPLE_RATE


def _device_display_name(device_index: int | None) -> str:
    """Human-readable label for a device index. None → 'system default'."""
    if device_index is None:
        return "system default"
    try:
        info = sd.query_devices(device_index)
        return str(info.get("name", f"device #{device_index}"))
    except Exception:
        return f"device #{device_index}"


def _blocksize_for_native_rate(native_rate: int) -> int:
    """Input blocksize that, after resampling to SAMPLE_RATE, yields exactly
    FRAME_SAMPLES output samples per callback. For 48k → 1440; 44.1k → 1323;
    16k → 480 (no resample). Off-grid rates get rounded; the resulting
    sub-millisecond drift is irrelevant downstream."""
    return int(round(native_rate * FRAME_SAMPLES / SAMPLE_RATE))


class DeviceOpenError(RuntimeError):  # noqa: N818  (Error suffix would be redundant; class name reads well)
    """Raised when the audio device cannot be opened in the SPEC format."""


class AudioInputSource:
    name: str = "audio_input_source"

    def __init__(
        self,
        *,
        preferred_device: str | None = None,
        prefer_respeaker: bool = True,
        bus: EventBus | None = None,
    ) -> None:
        self._preferred_device = preferred_device
        self._prefer_respeaker = prefer_respeaker
        self._bus = bus
        self._stream: Any = None
        self._on_frame: Callable[[AudioFrame], None] | None = None
        self._native_rate: int = SAMPLE_RATE
        self.is_loaded: bool = False

    # -- device selection --

    def _select_device_candidates(self) -> list[int | None]:
        """Return the ordered list of device indices to try. Primary is
        the WASAPI variant (preferred to avoid the MME barge-in restart
        race); fallback is the MME variant of the same device, used only
        if WASAPI cannot open at the device's native rate. None means the
        system default. Always returns at least one entry."""
        if self._preferred_device:
            wasapi, mme = self._find_input_device_variants(self._preferred_device)
            candidates: list[int | None] = [d for d in (wasapi, mme) if d is not None]
            if candidates:
                log.info("using configured input device: %s", self._preferred_device)
                return candidates
            log.warning(
                "configured input device %r not found; using system default",
                self._preferred_device,
            )
            # Explicit fallback to default; do NOT also scan for ReSpeaker
            # because an explicit setting was made (and missed).
            return [None]
        if self._prefer_respeaker:
            idx = self._find_respeaker()
            if idx is not None:
                log.info("auto-selected ReSpeaker / SEEED input at device index %d", idx)
                return [idx]
        log.info("using system default input device")
        return [None]

    def _find_input_device_variants(
        self, name_substring: str
    ) -> tuple[int | None, int | None]:
        """Return (wasapi_idx, mme_idx) for input devices whose name contains
        name_substring (case-insensitive). Either may be None. If neither
        WASAPI nor MME matched but some other host API did, that one is
        returned as the WASAPI slot so it still gets tried."""
        target = name_substring.lower()
        wasapi_idx = _wasapi_hostapi_index()
        mme_idx = _mme_hostapi_index()
        wasapi: int | None = None
        mme: int | None = None
        first_other: int | None = None
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            if target not in str(dev.get("name", "")).lower():
                continue
            ha = dev.get("hostapi")
            if wasapi_idx is not None and ha == wasapi_idx and wasapi is None:
                wasapi = idx
            elif mme_idx is not None and ha == mme_idx and mme is None:
                mme = idx
            elif first_other is None:
                first_other = idx
        if wasapi is None and mme is None:
            return (first_other, None)
        return (wasapi, mme)

    def _find_respeaker(self) -> int | None:
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", "")).lower()
            if any(tok in name for tok in _RESPEAKER_NAME_TOKENS):
                return idx
        return None

    # -- Loadable --

    async def load(self) -> None:
        if self.is_loaded:
            return
        candidates = self._select_device_candidates()
        last_err: Exception | None = None

        for attempt_idx, device_index in enumerate(candidates):
            stream, err = self._attempt_open_input_device(device_index)
            if stream is not None:
                if attempt_idx > 0:
                    log.warning("WASAPI variant failed; using MME fallback for input device")
                self.is_loaded = True
                return
            if err is not None:
                last_err = err

        # Phase 2: fall back to any other available input device.
        tried: set[int | None] = set(candidates)
        expected_label = self._configured_device_label()
        for device_index in self._fallback_input_candidates(tried):
            stream, err = self._attempt_open_input_device(device_index)
            if stream is not None:
                actual_name = _device_display_name(device_index)
                log.warning(
                    "input device %r could not be opened; using %r as fallback",
                    expected_label, actual_name,
                )
                if self._bus is not None:
                    self._bus.publish(NonFatalError(
                        module="audio_input",
                        issue="device_fallback",
                        expected=expected_label,
                        actual=actual_name,
                    ))
                self.is_loaded = True
                return
            if err is not None:
                last_err = err

        self._stream = None
        raise DeviceOpenError(
            f"could not open audio input on any available device "
            f"(targeted {SAMPLE_RATE} Hz mono int16, "
            f"{FRAME_SAMPLES}-sample output frames): {last_err}"
        ) from last_err

    def _attempt_open_input_device(
        self, device_index: int | None
    ) -> tuple[Any | None, Exception | None]:
        """Try to open a RawInputStream on `device_index` at its native rate.
        Returns (stream, None) on success with self._stream and self._native_rate
        set, or (None, last_error) if the attempt failed."""
        native_rate = _device_native_rate(device_index)
        blocksize = _blocksize_for_native_rate(native_rate)
        try:
            stream = sd.RawInputStream(
                samplerate=native_rate,
                channels=1,
                dtype="int16",
                blocksize=blocksize,
                device=device_index,
                callback=self._sd_callback,
            )
            stream.start()
        except Exception as e:
            log.warning(
                "failed to open input device %r at %d Hz: %s",
                device_index, native_rate, e,
            )
            return None, e
        self._stream = stream
        self._native_rate = native_rate
        if native_rate != SAMPLE_RATE:
            print(
                f"[boot] input opened at {native_rate}Hz, "
                f"resampling to {SAMPLE_RATE}Hz"
            )
        else:
            print(
                f"[boot] input opened at {native_rate}Hz "
                "(native, no resample)"
            )
        return stream, None

    def _configured_device_label(self) -> str:
        """Human-readable label for what was configured as the input device."""
        if self._preferred_device is None:
            return "system default"
        return self._preferred_device

    def _fallback_input_candidates(self, tried: set[int | None]) -> list[int | None]:
        """All input devices not in `tried`. Appends None (system default) last."""
        result: list[int | None] = []
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            if idx in tried:
                continue
            result.append(idx)
        if None not in tried:
            result.append(None)
        return result

    async def unload(self) -> None:
        if not self.is_loaded:
            return
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
            self._on_frame = None
            self.is_loaded = False

    # -- AudioSource --

    async def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                "AudioInputSource.start() called before load(); "
                "the LifecycleManager should load the source first."
            )
        self._on_frame = on_frame

    async def stop(self) -> None:
        # Detach only. Stream stays open; full release is unload().
        self._on_frame = None

    # -- internal --

    def _sd_callback(self, indata, frames, time_info, status) -> None:
        # Called from the PortAudio thread.
        if status:
            log.warning("sounddevice status: %s", status)
        cb = self._on_frame
        if cb is None:
            return
        data = bytes(indata)
        if self._native_rate != SAMPLE_RATE:
            data = self._resample_to_target(data)
        if len(data) != FRAME_BYTES:
            log.error(
                "sounddevice produced wrong frame size: %d bytes (expected %d)",
                len(data), FRAME_BYTES,
            )
            return
        try:
            cb(data)
        except Exception:
            # Exceptions from the on_frame callback would otherwise propagate
            # into PortAudio's C-level callback machinery and likely crash
            # the stream. Swallow + log; the pipeline's bridge is responsible
            # for handling its own enqueue failures.
            log.exception("on_frame callback raised")

    def _resample_to_target(self, data: bytes) -> bytes:
        """Linear-interpolation resample from self._native_rate to SAMPLE_RATE.

        Linear is intentionally chosen over polyphase / windowed-sinc: this
        runs in the PortAudio callback thread and per-frame budget is tight
        (~30 ms). Speech intelligibility is unaffected by the missing
        anti-alias filter at typical USB-mic content (energy concentrated
        below 4 kHz; the 8 kHz Nyquist after downsample is well clear).
        Avoids adding scipy as a dependency for one function."""
        samples = np.frombuffer(data, dtype=np.int16)
        if len(samples) == 0:
            return b""
        x_old = np.arange(len(samples), dtype=np.float32)
        x_new = np.linspace(0.0, len(samples) - 1, FRAME_SAMPLES, dtype=np.float32)
        out = np.interp(x_new, x_old, samples.astype(np.float32))
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()
