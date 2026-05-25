"""Audio stage protocols, value types, and pipeline-wide constants.

Every concrete stage module (devices.py, wake_word.py, vad.py, stt.py, tts.py)
imports its protocol from here; the pipeline imports the protocols too.
This sibling-file split exists to avoid the cycle that would form if the
protocols lived in pipeline.py.

Frame format (SPEC § Audio Pipeline): 16 kHz mono int16, 30 ms per frame =
480 samples = 960 bytes. Stages that internally need a different dtype
convert at their own boundary; the pipeline does not normalize.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

# --- frame format ---------------------------------------------------------

SAMPLE_RATE: int = 16_000
FRAME_SAMPLES: int = 480       # 30 ms at 16 kHz
FRAME_BYTES: int = FRAME_SAMPLES * 2  # int16 = 2 bytes/sample

# Type alias: a single audio frame is exactly FRAME_BYTES of int16 PCM.
AudioFrame = bytes

# --- pipeline-wide tunables ----------------------------------------------

# Silence-timeout from CS->LISTENING with no speech-started from VAD.
# Default lives here for Phase 2; the eventual config field is
# STTConfig.listening_timeout_seconds (added when pipeline wires to config).
LISTENING_TIMEOUT_DEFAULT: float = 6.0

# Window after TTS playback start during which VAD speech-started signals
# are ignored. Mitigates desktop-speaker -> desk-mic feedback false positives.
# Tunable; if false barge-ins recur in real use, raise this before reaching
# for software AEC.
TTS_BARGE_IN_GRACE_SECONDS: float = 0.2

# Trailing silence after detected speech that triggers VAD's ENDPOINT
# signal. 700 ms balances natural-pause tolerance against perceived
# responsiveness. Originally bumped to 1000 ms after Phase 2 real-use
# noted conversational pauses; rolled back to 700 ms because
# MIN_UTTERANCE_MS already filters out the spurious-endpoint case the
# longer window was guarding against. Tunable per VAD instance.
VAD_ENDPOINT_MS_DEFAULT: int = 700

# Number of consecutive VAD inference windows above threshold required
# before SPEECH_STARTED fires. Suppresses single-window blips from
# arming the endpoint timer or polluting the listen buffer's
# minimum-utterance check. Default 3 (~96 ms at the 32 ms VAD window).
SPEECH_START_WINDOWS_DEFAULT: int = 3

# After WakeWordDetected, the pipeline discards this many ms of incoming
# audio before frames reach VAD or the listen buffer.
#
# Lowered from 200 ms to 100 ms: on CPU with whisper base.en the first word
# of a command was being clipped ("Count" → "councillor", "slowly" dropped).
# 100 ms still discards the trailing tail of "...jarvis" reliably, while
# halving the risk of eating the onset of a short first word.
POST_WAKE_BLACKOUT_MS: int = 100

# Minimum audio chunk size, in milliseconds at the output device's native
# rate, that PiperTTS accumulates before writing to the sounddevice output
# stream.
#
# Why this exists: Piper emits short audio chunks (~30–80 ms at 22050 Hz)
# per synthesis step. Forwarding each chunk verbatim to OutputStream.write()
# starves PortAudio's playback buffer between writes — the host hears
# choppy / crackling output even though every byte is correct. Reference
# test using sd.play() with the entire array sounded perfectly clean on the
# same device + same scipy resampler, isolating the per-chunk write pattern
# as the cause.
#
# 200 ms is well above PortAudio's typical underrun threshold (~20 ms) yet
# small enough that a tail-of-sentence flush stays imperceptible. Tunable;
# raise to 300 ms if any new device still underruns.
OUTPUT_BUFFER_MS: int = 200

# Minimum buffered post-blackout audio length when ENDPOINT fires.
# Below this the pipeline silently returns to IDLE rather than spending
# an STT call on near-silence (also reduces whisper-hallucination
# opportunities on short noise clips).
MIN_UTTERANCE_MS: int = 500

# Per-frame duration derived from the format constants. Used by the
# pipeline's blackout countdown.
FRAME_DURATION_MS: int = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 30


# --- stage outputs --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WakeWordResult:
    confidence: float


class VADEvent(Enum):
    """Discrete signals the VAD wrapper emits per frame.

    SPEECH_STARTED: a previously-silent stretch transitioned to voice.
    ENDPOINT:       voice ended and silence has lasted long enough that
                    we consider the utterance finished (ready to transcribe).
    """

    SPEECH_STARTED = "speech_started"
    ENDPOINT = "endpoint"


# --- stage protocols ------------------------------------------------------


@runtime_checkable
class AudioSource(Protocol):
    """Captures audio and delivers frames via callback. The callback may be
    invoked from any thread (typically PortAudio's). The pipeline wraps it
    with loop.call_soon_threadsafe at registration time."""

    async def start(self, on_frame: Callable[[AudioFrame], None]) -> None: ...
    async def stop(self) -> None: ...


@runtime_checkable
class WakeWordDetector(Protocol):
    """Per-frame wake-word inference. feed() returns a result on detection,
    None otherwise. reset() clears any internal state (called after each
    detection so a re-trigger requires a fresh utterance)."""

    async def feed(self, frame: AudioFrame) -> WakeWordResult | None: ...
    def reset(self) -> None: ...


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Per-frame VAD with internal endpointing logic. feed() returns a VADEvent
    when state changes, None per uneventful frame. reset() clears endpointing
    state at the start of each new listen / barge-in cycle."""

    async def feed(self, frame: AudioFrame) -> VADEvent | None: ...
    def reset(self) -> None: ...


@runtime_checkable
class SpeechToText(Protocol):
    """Request/response transcription of a captured utterance. The argument
    is a contiguous run of int16 PCM at SAMPLE_RATE."""

    async def transcribe(self, audio: bytes) -> str: ...


@runtime_checkable
class TextToSpeech(Protocol):
    """Speak text. Two paths:

    - speak(text): blocks until the playback of `text` completes (or
      cancel() interrupts).
    - speak_stream(text_chunks): consumes an async iterator of text
      chunks and plays them with internal sentence-boundary buffering.
      This is the path the pipeline uses for LLM-streaming responses
      (token-level chunks would otherwise produce per-token synthesis
      stutter).

    cancel() must be safe to call from any state, including when no
    speak/speak_stream is in flight, and must abort current playback
    immediately (not wait for the buffer to drain)."""

    async def speak(self, text: str) -> None: ...
    async def speak_stream(self, text_chunks: AsyncIterator[str]) -> None: ...
    async def cancel(self) -> None: ...


# --- response producer ----------------------------------------------------

# Takes the user's transcription and yields response text chunks. Phase 2
# uses a trivial echo (yields the input once); Phase 3 swaps in the LLM
# router which streams sentence-grouped chunks.
#
# Cancellation contract (REQUIRED):
#   The pipeline calls _response_task.cancel() on barge-in. Implementations
#   MUST propagate asyncio.CancelledError cleanly through any in-flight I/O
#   (httpx streams, subprocess pipes, file handles) and release resources in
#   a finally block. A producer that swallows CancelledError or blocks past
#   it leaves the SM stuck in SPEAKING and prevents the user's barge-in
#   utterance from being captured. The Phase 3 Ollama streamer is the first
#   real test of this contract; the Phase 2 echo placeholder is trivially
#   compliant because it has no I/O to interrupt.
ResponseProducer = Callable[[str], AsyncIterator[str]]
