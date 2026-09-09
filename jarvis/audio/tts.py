"""Piper TTS wrapper with streaming-text input, mid-playback cancellation,
and an amplitude-envelope hook for the (future) overlay orb.

Implements TextToSpeech (audio/protocols.py) and Loadable (core/lifecycle.py).

Streaming text input
--------------------
The TextToSpeech protocol exposes speak(text: str). PiperTTS additionally
provides speak_stream(text_chunks: AsyncIterator[str]) for the Phase 3
LLM-token-streaming case. speak_stream buffers chunks and emits to
synthesis on a sentence boundary OR after `max_wait_seconds` (default
0.8 s) of no new text, whichever comes first. Rationale:

  - Sentence boundary detection means natural prosody and perceived
    responsiveness (Jarvis starts talking after the first sentence
    instead of waiting for the full response).
  - The max-wait fallback prevents the final incomplete sentence from
    stalling forever when the producer simply ends without trailing
    punctuation.
  - 0.8 s is the UX compromise: long enough that the LLM can finish a
    short clause; short enough that the listener doesn't experience a
    perceptible pause before the tail of a response.

Cancellation contract
---------------------
cancel() must abort playback within the duration of the current audio
write -- it is the load-bearing barge-in path. If the user starts
speaking and TTS keeps playing for another sentence, the experience
collapses.

Implementation:
  - threading.Event (_cancel_event) signals the synthesis thread.
  - sounddevice OutputStream.abort() is the actual stop -- NOT stop(),
    which waits for the buffer to drain. abort() drops buffered samples
    immediately.
  - The synthesis loop in _sync_speak checks _cancel_event between audio
    chunks AND catches the exception that stream.write() raises after
    abort, returning cleanly either way.
  - cancel() returns once the stream has been aborted and the event is
    set; the thread executor for in-flight speak() will exit on its
    next chunk boundary.

Amplitude envelope hook (for future overlay orb)
-------------------------------------------------
Constructor accepts an optional `on_amplitude: Callable[[float], None]`
callback. When set, _sync_speak computes per-window RMS amplitude
normalized to [0,1] at ~30 Hz (one window per ~33 ms of output audio)
and calls the callback. The callback is invoked from the synthesis
thread; the consumer is responsible for any thread bridging it needs
(Qt signals, asyncio.run_coroutine_threadsafe, etc.).

Why a callback rather than a bus event:
  - High frequency (~30 Hz). Going through bus dispatch adds overhead
    and ordering races for what is essentially per-frame UI data.
  - Avoids adding a new event type to events.py for a UI-only signal.
  - Phase 5's overlay can wrap this in whatever Qt-thread bridge it
    needs without coupling the TTS layer to PySide6.

Threading
---------
Same discipline as STT. Synthesis is CPU-bound (typical 100-300 ms per
sentence on the medium voice). speak() runs synthesis + playback inside
asyncio.to_thread so the loop stays free for barge-in detection. Stream
abort() is also wrapped in to_thread defensively.

Model files
-----------
Piper voices are two files: `<voice>.onnx` and `<voice>.onnx.json`.
Smaller than whisper (~60 MB for a medium voice), so bundling in the
installer is straightforward.

In production (post-Phase-7) the installer pre-places voice files into
a known directory and passes `voices_dir`. For development, place the
files manually or use the piper-tts CLI to download them. The wrapper
does NOT auto-download.
"""

from __future__ import annotations

import asyncio
import logging
import math
import queue
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from scipy import signal as _scipy_signal

from jarvis.audio.protocols import OUTPUT_BUFFER_MS
from jarvis.core.events import EventBus, NonFatalError

log = logging.getLogger(__name__)


# Sentence boundary: any text up to and including a run of [.!?] followed
# by whitespace OR end-of-string. Multiline (DOTALL) so newlines in
# streamed text don't break the match.
_SENTENCE_BOUNDARY = re.compile(r"(.+?)([.!?]+)(\s+|$)", re.DOTALL)

# Default max-wait for incomplete trailing text in speak_stream.
# 0.4 s keeps end-of-response latency low while still giving a slow
# producer (LLM finishing a final clause) room to land a sentence
# terminator before we flush a fragment.
SPEAK_STREAM_MAX_WAIT_SECONDS: float = 0.4

# Piper length_scale: 1.0 = default, lower = faster. 0.95 is a gentle
# 5% nudge above the upstream baseline; previous attempts at more
# aggressive values plus a scipy time-compression post-process produced
# audibly pitch-shifted output ("fast and chipmunky") which is worse
# than slow speech — slow is intelligible, pitched is not.
#
# Combined with the per-instance `speed` knob as
# `effective = SPEECH_LENGTH_SCALE / speed`. Tune via `speed` rather
# than this constant so the shared baseline stays consistent.
SPEECH_LENGTH_SCALE: float = 0.75

# Rough average duration per character of spoken English at Piper
# length_scale=1.0 — used only by the one-shot diagnostic that compares
# expected vs actual synthesised duration so we can spot a Piper version
# that ignores SynthesisConfig.length_scale.
_SECONDS_PER_CHAR_AT_DEFAULT_SCALE: float = 0.075

# Time-compression knob. DISABLED (None) by default — scipy.signal.
# resample_poly post-processing both speeds up AND raises pitch (no free
# time-stretch without WSOLA/PSOLA). At 1.30 the pitch shift was
# audible enough that we'd rather take the slower, in-tune output.
# Leave None unless a future WSOLA hook lands here.
TIME_COMPRESS_FACTOR: float | None = None


def _wasapi_hostapi_index() -> int | None:
    """Return the sounddevice hostapi index for Windows WASAPI, or None.

    WASAPI is preferred over MME when matching output devices by name: MME has
    a stream-restart race (PortAudioError 33) after abort() that causes the
    next speak() to fail. WASAPI's stream model doesn't exhibit this race."""
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


def _wasapi_shared_settings(device_idx: int | None):
    """Return sd.WasapiSettings(exclusive=False) when the device is on the
    WASAPI host API, else None.

    WASAPI exclusive mode rejects any sample rate the device doesn't natively
    expose — that's what made Piper's 22050 Hz fail on a 48 kHz-default DAC.
    Forcing shared mode lets the OS mixer accept arbitrary rates and (for
    22050 Hz specifically) avoids the audible spectral-imaging distortion
    that linear-interpolation upsampling to 48 kHz would otherwise produce.
    Returns None on non-Windows or when sounddevice doesn't expose the type."""
    try:
        wasapi_idx = _wasapi_hostapi_index()
        if wasapi_idx is None:
            return None
        if device_idx is not None:
            info = sd.query_devices(device_idx)
            ha = info.get("hostapi") if hasattr(info, "get") else None
            if ha is not None and ha != wasapi_idx:
                return None
        return sd.WasapiSettings(exclusive=False)
    except Exception:
        return None


def _device_native_rate(idx: int | None, fallback: int) -> int:
    """Return the device's default sample rate, or `fallback` if unknown.

    USB DACs / speakers rarely advertise 22050 Hz natively under WASAPI (which
    forbids the driver-side resampling MME silently performs). Opening at the
    native rate and resampling Piper's output in the synthesis loop is what
    makes WASAPI viable for them. None means system default — we don't probe
    that and just use the fallback (Piper's voice rate)."""
    if idx is None:
        return fallback
    try:
        info = sd.query_devices(idx)
        rate = info.get("default_samplerate") if hasattr(info, "get") else None
        if rate:
            return int(rate)
    except Exception:
        pass
    return fallback

def _device_display_name(device_index: int | None) -> str:
    """Human-readable label for a device index. None → 'system default'."""
    if device_index is None:
        return "system default"
    try:
        info = sd.query_devices(device_index)
        return str(info.get("name", f"device #{device_index}"))
    except Exception:
        return f"device #{device_index}"


# Amplitude-envelope cadence for the overlay orb.
_AMPLITUDE_HZ: int = 30

# Peak/RMS blend + gain stretch quiet/loud contrast so the orb pulses more
# visibly between syllables and consonants.
_AMP_PEAK_WEIGHT: float = 0.65
_AMP_GAIN: float = 2.2
_AMP_CURVE: float = 0.72


def amplitude_from_samples(window: np.ndarray) -> float:
    """Map an int16 PCM window to [0, 1] with expanded dynamic range."""
    if len(window) == 0:
        return 0.0
    f = window.astype(np.float32)
    rms = float(np.sqrt(np.mean(f ** 2)))
    peak = float(np.max(np.abs(f)))
    rms_n = min(1.0, rms / 32768.0)
    peak_n = min(1.0, peak / 32768.0)
    blended = (1.0 - _AMP_PEAK_WEIGHT) * rms_n + _AMP_PEAK_WEIGHT * peak_n
    boosted = min(1.0, blended * _AMP_GAIN)
    return min(1.0, boosted ** _AMP_CURVE)


class TTSLoadError(RuntimeError):  # noqa: N818
    """Raised when the Piper voice or output stream cannot be opened."""


class PiperTTS:
    name: str = "tts"

    def __init__(
        self,
        *,
        voice_name: str = "en_GB-alan-medium",
        voices_dir: Path | None = None,
        speed: float = 1.0,
        volume: float = 1.0,
        on_amplitude: Callable[[float], None] | None = None,
        output_device: str | int | None = None,
        bus: EventBus | None = None,
    ) -> None:
        if not 0.0 <= volume <= 1.0:
            raise ValueError(f"volume must be in [0.0, 1.0], got {volume}")
        if speed <= 0.0:
            raise ValueError(f"speed must be positive, got {speed}")
        self.voice_name = voice_name
        self._voices_dir = voices_dir
        self.speed = speed
        self.volume = volume
        self._on_amplitude = on_amplitude
        # output_device: None = system default; int = device index passed
        # straight through; str = case-insensitive substring match against
        # sounddevice's output device list (mirrors devices.py input matching).
        self._output_device: str | int | None = output_device
        self._bus = bus
        self._voice: Any = None
        self._stream: Any = None
        self._sample_rate: int = 22_050  # default; overwritten from voice.config
        self._output_rate: int = 22_050  # native rate of the opened output stream
        self._cancel_event = threading.Event()
        # Callback-mode plumbing. PortAudio drives consumption from its audio
        # thread on the device's clock — _sync_speak pushes resampled bytes
        # into _audio_queue and waits for _drained_event; _on_audio_callback
        # pulls from the queue when PortAudio asks for samples and sets the
        # event when both queue and pending residual are exhausted. This is
        # the topology sd.play() uses internally and the only one that gives
        # gap-free PortAudio playback in Python (write mode trickled bytes
        # too slowly between PortAudio pulls and produced choppy / aliased
        # output that no buffer-size or blocksize tweak fixed).
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._drained_event = threading.Event()
        self._drained_event.set()  # nothing pending at construction
        self._pending: bytes = b""  # callback-thread residual between calls
        # Set True by _open_stream when the PortAudio callback is actually
        # wired. Tests that hand-install a mock stream leave this False so
        # _sync_speak doesn't block forever waiting for a non-existent
        # callback to drain the queue.
        self._callback_wired: bool = False
        # One-shot diagnostic: log expected vs actual synthesis duration
        # on the first speak() so we can confirm Piper is honouring
        # SynthesisConfig.length_scale. If the ratio is ~1.0 regardless
        # of length_scale, the param is being ignored and the scipy
        # time-compression fallback (TIME_COMPRESS_FACTOR below) is the
        # remediation.
        self._first_synth_logged: bool = False
        self.is_loaded: bool = False

    # -- Loadable --

    async def load(self) -> None:
        if self.is_loaded:
            return
        try:
            from piper import PiperVoice
        except ImportError as e:  # pragma: no cover
            raise TTSLoadError("piper-tts is not installed") from e

        voice_path = self._find_voice_path()
        if voice_path is None:
            raise TTSLoadError(
                f"Piper voice {self.voice_name!r} not found. Looked in "
                f"{self._voices_dir or '<default search path>'}. In a "
                "development environment, download the voice (.onnx + "
                ".onnx.json) and either place it in the voices_dir you pass "
                "to the constructor or use the piper-tts CLI."
            )
        try:
            self._voice = await asyncio.to_thread(PiperVoice.load, str(voice_path))
        except Exception as e:
            raise TTSLoadError(
                f"could not load Piper voice from {voice_path}: {e}"
            ) from e

        # Pull the sample rate from the loaded voice (medium voices are
        # typically 22050 Hz; high-quality voices vary).
        try:
            self._sample_rate = int(self._voice.config.sample_rate)
        except AttributeError:
            log.warning(
                "could not read sample_rate from voice config; defaulting to %d",
                self._sample_rate,
            )

        try:
            self._stream = self._open_stream()
        except Exception as e:
            self._voice = None
            raise TTSLoadError(
                f"could not open audio output stream at {self._sample_rate} Hz: {e}"
            ) from e
        # Boot diagnostic so the effective Piper pace is visible at runtime.
        # print() (not log.info) keeps it on stdout without a log-level bump,
        # matching the surrounding [boot] device/rate lines.
        print(f"[boot] piper length_scale={self._effective_length_scale():.3f}")
        self.is_loaded = True

    async def unload(self) -> None:
        if not self.is_loaded:
            return
        # Make sure nothing is playing.
        await self.cancel()
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            log.exception("output stream close failed")
        self._stream = None
        self._voice = None
        self.is_loaded = False

    async def rewire_output(self, device: str | int | None) -> None:
        """Hot-swap the output device without reloading the voice model.

        Cancels any in-progress speech, closes the current stream, updates
        _output_device, and opens a fresh stream on the new device. Raises
        TTSLoadError (propagated from _open_stream) if the new device cannot
        be opened at any supported rate.

        Safe to call before load() — just stores the device for the next
        load() call."""
        if not self.is_loaded:
            self._output_device = device
            return
        await self.cancel()
        if self._stream is not None:
            try:
                await asyncio.to_thread(self._stream.close)
            except Exception:
                log.exception("stream close during output rewire failed")
            self._stream = None
        self._output_device = device
        log.info("hot-reload: rewiring output to %r", device)
        self._stream = self._open_stream()

    # -- TextToSpeech --

    async def speak(self, text: str) -> None:
        if not self.is_loaded or self._voice is None:
            log.warning("speak() called before load(); ignoring")
            return
        if not text or not text.strip():
            return
        if self._stream is None:
            # Stream was closed by cancel(). Settle 50 ms before reopening so
            # the audio driver fully releases its hardware buffers first.
            await asyncio.sleep(0.05)
            try:
                self._stream = self._open_stream()
            except Exception:
                log.exception("failed to reopen audio output stream after cancel")
                return
        self._cancel_event.clear()
        # The stream may have been left stopped by a prior abort(); restart.
        try:
            if not self._stream_active():
                self._stream.start()
        except Exception:
            log.exception("output stream start failed")
            return
        try:
            await asyncio.to_thread(self._sync_speak, text)
        except Exception:
            log.exception("speak failed")

    async def cancel(self) -> None:
        # Set the event first so any in-flight thread sees cancellation
        # before its drain wait returns.
        self._cancel_event.set()
        # Drop everything queued for playback BEFORE aborting the stream:
        # the callback may fire one more time between here and abort(), and
        # we don't want it emitting another block of stale audio. Drain also
        # sets _drained_event so an in-flight _sync_speak's wait exits.
        self._drain_audio_queue()
        self._callback_wired = False
        if self._stream is None:
            return
        # abort() (not stop()) drops buffered samples immediately. stop()
        # waits for the buffer to drain, which would leave Jarvis still
        # talking after the user's barge-in.
        try:
            await asyncio.to_thread(self._stream.abort)
        except Exception:
            log.exception("stream abort failed")
        # Close and nullify so the next speak() opens a fresh stream.
        # Reusing an aborted MME stream causes PortAudioError 33 ("Cannot
        # perform this operation while media data is still playing"); closing
        # it here prevents that race even on non-WASAPI host APIs.
        try:
            await asyncio.to_thread(self._stream.close)
        except Exception:
            log.exception("stream close after abort failed")
        self._stream = None

    # -- streaming-text extension (not on the protocol) --

    async def speak_stream(
        self,
        text_chunks: AsyncIterator[str],
        *,
        max_wait_seconds: float = SPEAK_STREAM_MAX_WAIT_SECONDS,
    ) -> None:
        """Buffer streaming text, emit to synthesis on sentence boundary OR
        after `max_wait_seconds` of no new text. Designed for the Phase 3
        LLM-token-streaming case where the producer yields tokens too
        small to synthesize individually."""
        buffer = ""
        aiter = text_chunks.__aiter__()
        producer_done = False

        while True:
            # Drain any complete sentences from the buffer.
            while True:
                match = _SENTENCE_BOUNDARY.match(buffer)
                if not match:
                    break
                sentence = (match.group(1) + match.group(2)).strip()
                buffer = buffer[match.end():]
                if sentence:
                    await self.speak(sentence)

            if producer_done:
                # Producer ended; flush any remaining incomplete tail.
                tail = buffer.strip()
                if tail:
                    await self.speak(tail)
                return

            # Wait for the next chunk. If buffer non-empty, bound the wait
            # so an incomplete final sentence doesn't stall.
            try:
                if buffer:
                    chunk = await asyncio.wait_for(
                        aiter.__anext__(), timeout=max_wait_seconds
                    )
                else:
                    chunk = await aiter.__anext__()
                buffer += chunk
            except StopAsyncIteration:
                producer_done = True
            except TimeoutError:
                # max_wait_seconds elapsed with non-empty buffer; flush.
                tail = buffer.strip()
                if tail:
                    await self.speak(tail)
                    buffer = ""

    # -- internal --

    def _stream_active(self) -> bool:
        try:
            return bool(self._stream.active)
        except AttributeError:
            return False

    def _find_voice_path(self) -> Path | None:
        """Locate the {voice_name}.onnx file. Requires {voice_name}.onnx.json
        alongside it. Returns None if not found."""
        if self._voices_dir is None:
            return None
        onnx = self._voices_dir / f"{self.voice_name}.onnx"
        config = self._voices_dir / f"{self.voice_name}.onnx.json"
        if onnx.exists() and config.exists():
            return onnx
        return None

    def _open_stream(self) -> Any:
        """Open and return a fresh RawOutputStream.

        Phase 1 — configured device: tries the WASAPI then MME variants of the
        configured output device (or system default when none is set).

        Phase 2 — device fallback: if every Phase-1 attempt fails, iterates all
        other available output devices ordered WASAPI > MME > other. The first
        that accepts the stream is used. A WARNING is logged and a NonFatalError
        is published on the bus so the UI layer can surface the situation.

        Only raises TTSLoadError if EVERY available output device fails.

        blocksize is matched to OUTPUT_BUFFER_MS so each PortAudio callback
        consumes one accumulated write from _sync_speak. Combined with
        latency='high' this prevents the write-mode trickle problem. Callback
        mode (not write mode) is what makes gap-free playback possible in Python."""
        candidates = self._select_output_candidates()
        piper_rate = self._sample_rate
        last_err: Exception | None = None

        for attempt_idx, device_index in enumerate(candidates):
            stream, err = self._attempt_open_stream(device_index, piper_rate)
            if stream is not None:
                if attempt_idx > 0:
                    log.warning("WASAPI variant failed; using MME fallback for output device")
                return stream
            if err is not None:
                last_err = err

        # Phase 2: fall back to any other available output device.
        tried: set[int | None] = set(candidates)
        expected_label = self._configured_device_label()
        for device_index in self._fallback_output_candidates(tried):
            stream, err = self._attempt_open_stream(device_index, piper_rate)
            if stream is not None:
                actual_name = _device_display_name(device_index)
                log.warning(
                    "output device %r could not be opened; using %r as fallback",
                    expected_label, actual_name,
                )
                # Remember the fallback so speak()-side reopens reuse it.
                self._output_device = device_index
                if self._bus is not None:
                    self._bus.publish(NonFatalError(
                        module="tts",
                        issue="device_fallback",
                        expected=expected_label,
                        actual=actual_name,
                    ))
                return stream
            if err is not None:
                last_err = err

        raise TTSLoadError(
            f"could not open audio output on any available device "
            f"(targeted {piper_rate} Hz): {last_err}"
        ) from last_err

    def _attempt_open_stream(
        self, device_index: int | None, piper_rate: int
    ) -> tuple[Any | None, Exception | None]:
        """Try to open a RawOutputStream on `device_index`.

        Rate order: device's reported native rate first, Piper's synthesis
        rate second (only when different). Opening at the native rate always
        succeeds when the device is reachable — the OS mixer accepts it
        regardless of WASAPI mode — and triggers in-process resampling via
        scipy. Trying Piper's rate first caused failures on drivers that only
        accept their native rate in WASAPI shared mode.

        Returns (stream, None) on success with self._output_rate set, or
        (None, last_error) if every rate attempt failed."""
        native_rate = _device_native_rate(device_index, piper_rate)
        wasapi_settings = _wasapi_shared_settings(device_index)
        rate_attempts: list[tuple[int, Any]] = [
            (native_rate, wasapi_settings),
        ]
        if piper_rate != native_rate:
            rate_attempts.append((piper_rate, wasapi_settings))
        last_err: Exception | None = None
        for rate, extra_settings in rate_attempts:
            blocksize = max(1, int(OUTPUT_BUFFER_MS * rate / 1000))
            kwargs: dict[str, Any] = {
                "samplerate": rate,
                "channels": 1,
                "dtype": "int16",
                "device": device_index,
                "blocksize": blocksize,
                "latency": "high",
                "callback": self._on_audio_callback,
            }
            if extra_settings is not None:
                kwargs["extra_settings"] = extra_settings
            try:
                stream = sd.RawOutputStream(**kwargs)
            except Exception as e:
                last_err = e
                log.warning(
                    "failed to open output device %r at %d Hz: %s",
                    device_index, rate, e,
                )
                continue
            self._output_rate = rate
            self._pending = b""
            self._callback_wired = True
            self._drained_event.set()
            if rate != piper_rate:
                print(f"[boot] output opened at {rate}Hz, resampling from {piper_rate}Hz")
            else:
                print(f"[boot] output opened at {rate}Hz (no resample)")
            return stream, None
        return None, last_err

    def _configured_device_label(self) -> str:
        """Human-readable label for what was configured as the output device."""
        if self._output_device is None:
            return "system default"
        if isinstance(self._output_device, int):
            return f"device #{self._output_device}"
        return self._output_device

    def _fallback_output_candidates(self, tried: set[int | None]) -> list[int | None]:
        """All output devices not in `tried`, ordered WASAPI > MME > other.
        Appends None (system default) last if it has not been tried yet."""
        wasapi_idx = _wasapi_hostapi_index()
        mme_idx = _mme_hostapi_index()
        wasapi_devs: list[int] = []
        mme_devs: list[int] = []
        other_devs: list[int] = []
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_output_channels", 0)) <= 0:
                continue
            if idx in tried:
                continue
            ha = dev.get("hostapi")
            if wasapi_idx is not None and ha == wasapi_idx:
                wasapi_devs.append(idx)
            elif mme_idx is not None and ha == mme_idx:
                mme_devs.append(idx)
            else:
                other_devs.append(idx)
        result: list[int | None] = [*wasapi_devs, *mme_devs, *other_devs]
        if None not in tried:
            result.append(None)
        return result

    def _select_output_candidates(self) -> list[int | None]:
        """Return ordered device indices to try in Phase 1.

        Sorting priority (lower = tried first):
          1. Host API: WASAPI (0) > MME (1) > other (2) — avoids MME's
             stream-restart race after barge-in abort().
          2. Phantom flag: real devices (0) before Windows "(2- …)"
             phantom re-enumerations (1) which often fail to open.
          3. Device index: tie-break by enumeration order.

        Always returns at least one entry (falls back to [None] when no
        match is found so _open_stream can try the system default)."""
        if self._output_device is None:
            return [None]
        if isinstance(self._output_device, int):
            return [self._output_device]
        target = self._output_device.lower()
        wasapi_idx = _wasapi_hostapi_index()
        mme_idx = _mme_hostapi_index()

        # (api_pri, is_phantom, dev_idx, dev_name)
        matched: list[tuple[int, int, int, str]] = []
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_output_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", ""))
            if target not in name.lower():
                continue
            ha = dev.get("hostapi")
            if wasapi_idx is not None and ha == wasapi_idx:
                api_pri = 0
            elif mme_idx is not None and ha == mme_idx:
                api_pri = 1
            else:
                api_pri = 2
            phantom = 1 if "(2- " in name else 0
            matched.append((api_pri, phantom, idx, name))

        if not matched:
            log.warning(
                "configured output device %r not found; using system default",
                self._output_device,
            )
            return [None]

        matched.sort(key=lambda m: (m[0], m[1], m[2]))
        cand_desc = ", ".join(f"{m[2]}:{m[3]!r}" for m in matched)
        log.info(
            "[boot] output device candidates for %r: [%s] → using %r (idx %d)",
            self._output_device, cand_desc, matched[0][3], matched[0][2],
        )
        return [m[2] for m in matched]

    def _iterate_audio_chunks(self, text: str) -> Iterable[bytes]:
        """Yield raw int16 audio bytes from Piper. Abstracted as its own
        method so tests can patch it without mocking Piper internals.

        Tries the modern AudioChunk API first (piper-tts 1.x) and falls
        back to the legacy synthesize_stream_raw path. self.speed != 1.0
        is wired through Piper's SynthesisConfig.length_scale (inverse of
        speed); for legacy/mocked voices that lack SynthesisConfig, the
        speed is applied later as a sample-domain time compression in
        _apply_speed (pitch-shifts slightly but is acceptable for ≤1.2×)."""
        voice = self._voice
        if voice is None:
            return
        total_bytes = 0
        if hasattr(voice, "synthesize"):
            syn_config = self._build_syn_config()
            kwargs: dict[str, Any] = {}
            if syn_config is not None:
                kwargs["syn_config"] = syn_config
            for chunk in voice.synthesize(text, **kwargs):
                raw = bytes(chunk.audio_int16_bytes)
                total_bytes += len(raw)
                yield self._maybe_time_compress(raw)
            self._maybe_log_first_synth_duration(text, total_bytes)
            return
        if hasattr(voice, "synthesize_stream_raw"):
            for raw in voice.synthesize_stream_raw(text):
                total_bytes += len(raw)
                yield self._maybe_time_compress(raw)
            self._maybe_log_first_synth_duration(text, total_bytes)
            return
        raise TTSLoadError("Piper voice has no recognized synthesis API")

    def _maybe_log_first_synth_duration(self, text: str, total_bytes: int) -> None:
        """One-shot diagnostic so we can see whether Piper actually
        honoured the SynthesisConfig.length_scale we asked for. Print
        runs once per PiperTTS instance to avoid spamming."""
        if self._first_synth_logged or total_bytes == 0:
            return
        self._first_synth_logged = True
        # int16 mono.
        actual_seconds = (total_bytes / 2) / max(1, self._sample_rate)
        expected_default = max(1, len(text)) * _SECONDS_PER_CHAR_AT_DEFAULT_SCALE
        expected_scaled = expected_default * self._effective_length_scale()
        ratio_vs_default = actual_seconds / expected_default
        print(
            f"[tts-debug] first-synth text_chars={len(text)} "
            f"actual={actual_seconds:.2f}s "
            f"expected@1.0={expected_default:.2f}s "
            f"expected@{self._effective_length_scale():.2f}="
            f"{expected_scaled:.2f}s "
            f"ratio_vs_default={ratio_vs_default:.2f}"
        )

    def _maybe_time_compress(self, audio_bytes: bytes) -> bytes:
        """Time-compress Piper output via scipy.signal.resample_poly when
        TIME_COMPRESS_FACTOR is set.

        We pick (up, down) as the smallest rational approximating
        1/factor (limit_denominator(50)). For factor=1.30 this picks
        up=10, down=13 → output has 10/13 the samples and plays in 10/13
        the time at the same 22050 Hz rate, i.e. 30% faster. Pitch rises
        ~30% with it; acceptable at this ratio for spoken English."""
        factor = TIME_COMPRESS_FACTOR
        if factor is None or factor == 1.0:
            return audio_bytes
        if not audio_bytes:
            return audio_bytes
        ratio = Fraction(1.0 / factor).limit_denominator(50)
        up = max(1, ratio.numerator)
        down = max(1, ratio.denominator)
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        out = _scipy_signal.resample_poly(samples.astype(np.float32), up, down)
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()

    def _effective_length_scale(self) -> float:
        """Piper length_scale we want applied per synthesis call.

        SPEECH_LENGTH_SCALE is the conversational-pace baseline (0.85 by
        default); self.speed is the per-instance multiplier. Higher speed
        = lower length_scale = faster playback. Guards against speed<=0
        which would otherwise produce a divide-by-zero / negative scale."""
        speed = self.speed if self.speed > 0 else 1.0
        return SPEECH_LENGTH_SCALE / speed

    def _build_syn_config(self) -> Any | None:
        """Build a piper.SynthesisConfig honouring the effective length_scale.
        Returns None if Piper's API in this venv doesn't expose
        SynthesisConfig (older piper-tts) — in that case _iterate_audio_chunks
        falls back to plain voice.synthesize(text), accepting the upstream
        default pace rather than failing the speak() call."""
        try:
            from piper.config import SynthesisConfig
        except Exception:
            return None
        return SynthesisConfig(length_scale=self._effective_length_scale())

    def _sync_speak(self, text: str) -> None:
        """Synthesize and enqueue audio for the PortAudio callback to play.
        Always runs inside asyncio.to_thread.

        Push semantics: each Piper chunk is volume-adjusted, resampled to
        the output rate, accumulated into ~OUTPUT_BUFFER_MS blocks, and
        put on _audio_queue. PortAudio's audio thread drains via
        _on_audio_callback at the device's clock, so this method does not
        block on stream.write — that was the write-mode trickle bug.
        After all chunks are enqueued, we wait for _drained_event so
        speak() doesn't return until audio has actually played out (the
        pipeline relies on this to keep the SPEAKING-state invariant).

        Cancel: returns immediately on _cancel_event during synthesis;
        skips the final flush and the drain wait."""
        buf = bytearray()
        target_bytes = max(
            1, (OUTPUT_BUFFER_MS * self._output_rate * 2) // 1000
        )
        for chunk_bytes in self._iterate_audio_chunks(text):
            if self._cancel_event.is_set():
                return
            audio = self._apply_volume(chunk_bytes) if self.volume != 1.0 else chunk_bytes
            out_bytes = self._resample_for_output(audio)
            buf.extend(out_bytes)
            # Amplitude envelope is computed from the pre-resample audio so
            # the window math at self._sample_rate stays consistent regardless
            # of the output device's native rate. Fire per-chunk (not
            # per-flush) so the envelope cadence is independent of the
            # accumulator boundary — the future overlay orb expects ~30 Hz.
            # Note: in callback mode envelope fires slightly ahead of the
            # bytes actually reaching the speakers (queue depth latency);
            # for a UI orb the cadence is what matters, not perfect lipsync.
            self._emit_amplitude_envelope(audio)
            if len(buf) >= target_bytes:
                self._enqueue_audio(bytes(buf))
                buf.clear()
        if buf and not self._cancel_event.is_set():
            self._enqueue_audio(bytes(buf))
        # Wait for PortAudio's callback to finish playing everything we
        # queued. Skipped when no real callback is wired (tests with mocked
        # streams) — they never drain, so blocking would hang forever.
        if not self._callback_wired:
            return
        # Bounded wait: cancel_event is the early-exit; the 30 s upper bound
        # only fires if the callback never runs at all (driver hung), which
        # is preferable to a literal-forever hang.
        deadline = time.monotonic() + 30.0
        while not self._cancel_event.is_set() and time.monotonic() < deadline:
            if self._drained_event.wait(timeout=0.1):
                if self._audio_queue.empty() and not self._pending:
                    return
                # Race: more was enqueued or callback re-entered between
                # the empty check and our observation. Loop and re-check.
                self._drained_event.clear()

    def _enqueue_audio(self, data: bytes) -> None:
        """Push one accumulated block onto the audio queue and clear the
        drained event so _sync_speak's drain wait doesn't see a stale set
        value from the last sentence."""
        self._drained_event.clear()
        self._audio_queue.put(data)

    def _on_audio_callback(
        self, outdata: Any, frames: int, time_info: Any, status: Any
    ) -> None:
        """PortAudio audio-thread callback. Pulls from _audio_queue into
        outdata; fills any remainder with silence so PortAudio never reports
        underrun. MUST NOT block, MUST NOT raise — both would corrupt
        PortAudio's C-level callback machinery and likely crash the stream."""
        needed = frames * 2  # int16 mono
        try:
            written = 0
            # Cast to unsigned bytes ('B'): the default memoryview format on
            # a CFFI-backed buffer can be signed, which raises OverflowError
            # silently on assignment of any byte >127. 'B' lets the whole
            # 0..255 range pass through verbatim.
            view = memoryview(outdata).cast("B")
            while written < needed:
                if self._pending:
                    take = min(needed - written, len(self._pending))
                    view[written:written + take] = self._pending[:take]
                    self._pending = self._pending[take:]
                    written += take
                    continue
                try:
                    self._pending = self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            if written < needed:
                # Fill the unplayed remainder with silence so PortAudio
                # always gets a full buffer back.
                view[written:needed] = b"\x00" * (needed - written)
                if not self._pending and self._audio_queue.empty():
                    self._drained_event.set()
        except Exception:
            # Last-resort silence fill so PortAudio doesn't get garbage.
            try:
                memoryview(outdata).cast("B")[:needed] = b"\x00" * needed
            except Exception:
                pass

    def _drain_audio_queue(self) -> None:
        """Discard everything queued for playback. Called from cancel()
        before stream.abort() so the callback can't emit one last block
        after the abort point."""
        try:
            while True:
                self._audio_queue.get_nowait()
        except queue.Empty:
            pass
        self._pending = b""
        self._drained_event.set()

    def _resample_for_output(self, audio_bytes: bytes) -> bytes:
        """Polyphase resample from self._sample_rate (Piper voice rate) to
        self._output_rate (output device native rate). No-op when the rates
        match.

        scipy.signal.resample_poly applies an anti-alias FIR before decimation
        and an anti-imaging FIR after interpolation, which is what the previous
        np.interp linear path was missing — upsampling 22050 → 48000 by linear
        interpolation produces audible spectral-imaging distortion (broadband
        hiss riding on the speech). The integer up/down ratio is reduced by
        gcd so the polyphase filter stays small (320/147 for 22050→48000)."""
        if self._output_rate == self._sample_rate:
            return audio_bytes
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(samples) == 0:
            return b""
        g = math.gcd(self._output_rate, self._sample_rate)
        up = self._output_rate // g
        down = self._sample_rate // g
        out = _scipy_signal.resample_poly(samples.astype(np.float32), up, down)
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()

    def _apply_volume(self, audio_bytes: bytes) -> bytes:
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        samples *= self.volume
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    def _emit_amplitude_envelope(self, audio_bytes: bytes) -> None:
        if self._on_amplitude is None:
            return
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            window_size = self._sample_rate // _AMPLITUDE_HZ
            if window_size <= 0 or len(samples) == 0:
                return
            for i in range(0, len(samples), window_size):
                window = samples[i:i + window_size]
                if len(window) == 0:
                    continue
                amplitude = amplitude_from_samples(window)
                try:
                    self._on_amplitude(amplitude)
                except Exception:
                    log.exception("on_amplitude callback raised")
        except Exception:
            log.exception("amplitude envelope computation failed")
