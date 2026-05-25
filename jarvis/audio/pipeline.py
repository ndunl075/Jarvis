"""Audio pipeline orchestrator. Owns the audio source, routes frames per
ConversationalState, drives SM transitions, and runs the response chain.

See `audio/protocols.py` for the stage interface contracts and constants.
The design rationale (queue topology, barge-in mechanics, listening-timeout
ownership, frame format trust, TTS-feedback grace window) is documented in
the Phase 2 Task 1 design note in the project history.

Interrupt mechanism
-------------------
Wake word is fed frames in ALL states (IDLE, LISTENING, THINKING, SPEAKING).
When the wake word fires during SPEAKING or THINKING, the pipeline treats it
as an explicit interrupt: it cancels the in-flight response chain (TTS abort
+ task cancel), clears conversation history via on_wake, and transitions
directly to LISTENING. This avoids speaker-to-mic feedback false positives
because openWakeWord is trained specifically on "hey jarvis" phonemes — normal
TTS speech ("Paris", "volume", "of course") will not trigger it.

Known limitation — wake-word interrupt on speaker-based setups
--------------------------------------------------------------
Wake-word interrupt works reliably only when TTS output does not reach the
microphone. On a typical desktop setup (speakers + desk mic, no AEC), Jarvis's
own speech bleeds into the mic and masks the user's "hey jarvis": the combined
signal drops the openWakeWord score below the detection threshold even though
the code path is architecturally correct. Live testing confirmed this: the
handler (_handle_wake_detection) is never reached, not because frames are
skipped (they are fed unconditionally) but because the model simply does not
fire.

Workaround: use headphones. With headphones the mic captures only the user's
voice, and interrupt works as designed.

Phase 6 backlog: integrate webrtc-audio-processing (or a comparable AEC
library) before the wake-word stage. AEC subtracts the speaker reference from
the mic signal so the classifier receives a clean "hey jarvis" even while TTS
is playing. Until that lands, wake-word interrupt is a headphone-only feature.

VAD-based barge-in is a separate mechanism and remains disabled by default
(`BARGE_IN_ENABLED = False`): without AEC, desktop speakers feeding back into
the desk mic fire VAD's SPEECH_STARTED mid-response and cause self-interrupts.
Re-enable when AEC is implemented (Phase 6) or on headphone-only deployments —
pass `barge_in_enabled=True` to the constructor. The barge-in code path is
intact and complementary to wake-word interrupt; they are not mutually exclusive.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from jarvis.audio.protocols import (
    FRAME_BYTES,
    FRAME_DURATION_MS,
    LISTENING_TIMEOUT_DEFAULT,
    MIN_UTTERANCE_MS,
    POST_WAKE_BLACKOUT_MS,
    SAMPLE_RATE,
    TTS_BARGE_IN_GRACE_SECONDS,
    AudioFrame,
    AudioSource,
    ResponseProducer,
    SpeechToText,
    TextToSpeech,
    VADEvent,
    VoiceActivityDetector,
    WakeWordDetector,
    WakeWordResult,
)
from jarvis.core.events import (
    EventBus,
    LLMResponseChunk,
    LLMResponseComplete,
    TranscriptionReady,
    WakeWordDetected,
)
from jarvis.core.state_machine import ConversationalState, Mode, StateMachine

log = logging.getLogger(__name__)

# Default Q_frames depth: ~1 second of audio at 33 fps. Sized for jitter
# absorption; sustained drops indicate the loop is stalling and the warn
# log will surface that.
_FRAME_QUEUE_DEFAULT = 32

# See module docstring. Disabled by default: speaker-to-mic feedback fires
# spurious VAD SPEECH_STARTED during TTS playback and the system interrupts
# itself. Wake-word retrigger is the supported interrupt mechanism until AEC
# (or headphone-only deployment) is in place.
BARGE_IN_ENABLED: bool = False


class AudioPipeline:
    def __init__(
        self,
        source: AudioSource,
        wake_word: WakeWordDetector,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        tts: TextToSpeech,
        response_producer: ResponseProducer,
        bus: EventBus,
        sm: StateMachine,
        *,
        listening_timeout_seconds: float = LISTENING_TIMEOUT_DEFAULT,
        frame_queue_maxsize: int = _FRAME_QUEUE_DEFAULT,
        post_wake_blackout_ms: int = POST_WAKE_BLACKOUT_MS,
        min_utterance_ms: int = MIN_UTTERANCE_MS,
        barge_in_enabled: bool = BARGE_IN_ENABLED,
        on_wake: Callable[[], None] | None = None,
        log_wake_during_speaking: bool = False,
    ) -> None:
        self._source = source
        self._wake_word = wake_word
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._response_producer = response_producer
        self._bus = bus
        self._sm = sm
        self._listening_timeout_seconds = listening_timeout_seconds
        self._post_wake_blackout_ms = post_wake_blackout_ms
        self._min_utterance_ms = min_utterance_ms
        self._barge_in_enabled = barge_in_enabled
        # Fires once per wake detection, before LISTENING transition.
        # The composition root wires Conversation.clear here so every
        # wake-word activation starts with empty history — the LLM
        # cannot anchor on prior turns to fire the wrong tool, at the
        # cost of breaking "tell me more" follow-ups. The trade-off is
        # documented in conversation.py.
        self._on_wake = on_wake
        self._log_wake_during_speaking = log_wake_during_speaking
        self._speaking_debug_frame_count: int = 0

        self._q_frames: asyncio.Queue[AudioFrame] = asyncio.Queue(
            maxsize=frame_queue_maxsize
        )
        self._listen_buffer = bytearray()
        self._tts_started_at: float | None = None
        # Decremented per LISTENING-state frame; while > 0, frames are
        # discarded before VAD or buffer. Set on wake_word detection.
        self._blackout_frames_remaining: int = 0

        self._frame_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None
        self._listening_timer: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False

    # -- lifecycle --

    async def start(self) -> None:
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        # Note: we deliberately do not subscribe to ConversationalStateChanged
        # for timer management. The pipeline is the single CS-mutator (per
        # design note) so timer start/cancel is wired directly into the
        # transition handlers below; subscriber-based wiring would race with
        # the synchronous SM state update vs. async bus dispatch.
        await self._source.start(self._on_audio_frame)
        self._frame_task = asyncio.create_task(self._frame_loop())
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            await self._source.stop()
        except Exception:
            log.exception("audio source stop failed")
        await self._cancel_response_chain()
        self._cancel_listening_timer()
        if self._frame_task is not None:
            self._frame_task.cancel()
            try:
                await self._frame_task
            except asyncio.CancelledError:
                pass
            self._frame_task = None

    # -- source-thread -> loop bridge --

    def _on_audio_frame(self, frame: AudioFrame) -> None:
        # Called from the audio source thread (PortAudio in production, the
        # test harness in tests). Hand off to the loop without blocking.
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._enqueue_frame, frame)

    def _enqueue_frame(self, frame: AudioFrame) -> None:
        # Cheap ingress assertion: misconfigured sources surface immediately.
        if len(frame) != FRAME_BYTES:
            log.error(
                "audio source produced wrong frame size: %d bytes (expected %d)",
                len(frame), FRAME_BYTES,
            )
            return
        try:
            self._q_frames.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._q_frames.get_nowait()  # drop oldest
                self._q_frames.put_nowait(frame)
                log.warning("Q_frames full; dropped oldest frame")
            except asyncio.QueueEmpty:
                # Race: someone drained between put_nowait and get_nowait.
                pass

    # -- listening silence timeout --

    def _start_listening_timer(self) -> None:
        self._cancel_listening_timer()
        self._listening_timer = asyncio.create_task(self._listening_timeout_coro())

    def _cancel_listening_timer(self) -> None:
        if self._listening_timer is not None:
            self._listening_timer.cancel()
            self._listening_timer = None

    async def _listening_timeout_coro(self) -> None:
        try:
            await asyncio.sleep(self._listening_timeout_seconds)
        except asyncio.CancelledError:
            return
        # Re-check: another transition may have moved us out of LISTENING
        # in the gap between sleep wake and this line.
        if self._sm.conversational_state is ConversationalState.LISTENING:
            log.info("listening silence timeout; returning to IDLE")
            self._sm.set_conversational_state(ConversationalState.IDLE)

    # -- frame loop --

    async def _frame_loop(self) -> None:
        try:
            while True:
                frame = await self._q_frames.get()
                if self._sm.mode is not Mode.ACTIVE:
                    continue
                # Feed wake word for ALL states (IDLE, LISTENING, THINKING,
                # SPEAKING). During SPEAKING/THINKING this enables the
                # wake-word interrupt path: if the user says "hey jarvis"
                # mid-response, _handle_wake_detection cancels TTS and
                # returns to LISTENING. openWakeWord's narrow phoneme set
                # means TTS output won't self-trigger it.
                wake_result = await self._wake_word.feed(frame)
                if wake_result is not None:
                    await self._handle_wake_detection(wake_result)
                    continue
                cs = self._sm.conversational_state
                if self._log_wake_during_speaking and cs is ConversationalState.SPEAKING:
                    self._speaking_debug_frame_count += 1
                    if self._speaking_debug_frame_count % 10 == 0:
                        log.debug(
                            "[wake-during-speaking] score=%.4f threshold=%.4f",
                            getattr(self._wake_word, "last_score", 0.0),
                            getattr(self._wake_word, "threshold", float("nan")),
                        )
                if cs is ConversationalState.LISTENING:
                    await self._dispatch_listening(frame)
                elif cs is ConversationalState.THINKING:
                    # Run VAD so its internal state stays warm; do not act.
                    await self._vad.feed(frame)
                elif cs is ConversationalState.SPEAKING:
                    await self._dispatch_speaking(frame)
                # IDLE: nothing to do beyond the wake-word feed already done.
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("frame loop crashed")
            raise

    async def _dispatch_listening(self, frame: AudioFrame) -> None:
        # Post-wake blackout: discard the first ~POST_WAKE_BLACKOUT_MS of
        # frames so the trailing tail of "jarvis" doesn't pollute VAD or
        # the listen buffer.
        if self._blackout_frames_remaining > 0:
            self._blackout_frames_remaining -= 1
            return
        self._listen_buffer.extend(frame)
        event = await self._vad.feed(frame)
        if event is VADEvent.SPEECH_STARTED:
            # User is talking; cancel the silence deadline.
            self._cancel_listening_timer()
        elif event is VADEvent.ENDPOINT:
            await self._handle_endpoint()

    async def _dispatch_speaking(self, frame: AudioFrame) -> None:
        # Gate at the top: when barge-in is disabled there's no reason to
        # spend VAD CPU on SPEAKING-state frames at all. The VAD's internal
        # state will be reset on the next wake detection, so skipping doesn't
        # leave it in a confused state.
        if not self._barge_in_enabled:
            return
        event = await self._vad.feed(frame)
        if event is VADEvent.SPEECH_STARTED:
            await self._handle_barge_in()

    # -- transition handlers --

    async def _handle_wake_detection(self, result: WakeWordResult) -> None:
        # NOTE: this branch is reached only when the wake-word model fires
        # during SPEAKING/THINKING. On speaker-based setups without AEC,
        # speaker bleed masks the user's voice and the model never fires —
        # see the "Known limitation" section in the module docstring.
        # With headphones (or once Phase 6 AEC lands) this path works correctly.
        cs = self._sm.conversational_state
        if cs is ConversationalState.SPEAKING or cs is ConversationalState.THINKING:
            log.info("[wake] interrupted %s", cs.name)
            await self._cancel_response_chain()
            # THINKING → LISTENING is an illegal transition; pass through IDLE.
            # SPEAKING → LISTENING is legal so no detour is needed there.
            if self._sm.conversational_state is ConversationalState.THINKING:
                self._sm.set_conversational_state(ConversationalState.IDLE)
        self._bus.publish(WakeWordDetected(confidence=result.confidence))
        self._listen_buffer.clear()
        self._wake_word.reset()
        self._vad.reset()
        # Wipe conversation history (when the composition root wired it).
        # Isolates each wake-word activation: prior tool-anchored context
        # can't bias the LLM into firing the wrong tool on a fresh
        # request. Failure is swallowed-with-log because a busted hook
        # must not stop the listening transition.
        if self._on_wake is not None:
            try:
                self._on_wake()
            except Exception:
                log.exception("on_wake hook raised; ignoring")
        # Arm the post-wake blackout. Ceiling division so the actual
        # blackout is always >= the configured ms.
        if self._post_wake_blackout_ms > 0:
            self._blackout_frames_remaining = (
                -(-self._post_wake_blackout_ms // FRAME_DURATION_MS)
            )
        else:
            self._blackout_frames_remaining = 0
        self._sm.set_conversational_state(ConversationalState.LISTENING)
        self._start_listening_timer()

    async def _handle_endpoint(self) -> None:
        self._cancel_listening_timer()
        audio = bytes(self._listen_buffer)
        self._listen_buffer.clear()
        # Bug-D gate: don't burn an STT call on near-silence. Buffer is
        # post-blackout, so this measures actual captured audio length.
        audio_ms = (len(audio) // 2) * 1000 // SAMPLE_RATE
        if audio_ms < self._min_utterance_ms:
            log.info(
                "utterance too short (%d ms < %d ms); returning to IDLE",
                audio_ms, self._min_utterance_ms,
            )
            self._sm.set_conversational_state(ConversationalState.IDLE)
            return
        self._sm.set_conversational_state(ConversationalState.THINKING)
        # The response chain runs as a separate task so the frame loop keeps
        # processing barge-in candidates while STT/LLM/TTS execute.
        self._response_task = asyncio.create_task(self._run_response_chain(audio))

    async def _handle_barge_in(self) -> None:
        # Re-check CS at decision time. VAD events can land here after the
        # response chain has already finished and the SM moved SPEAKING->IDLE,
        # because _dispatch_speaking sees a stale state value between when the
        # frame was dequeued and when this coroutine actually runs. Without
        # this guard, a delayed SPEECH_STARTED fires barge-in from IDLE,
        # cancelling nothing and pulling the SM into LISTENING — exactly the
        # spurious "barge_in elapsed_ms=1953" trace from the live log.
        if self._sm.conversational_state is not ConversationalState.SPEAKING:
            return
        # Suppress within the grace window after TTS playback start.
        # Mitigates speaker->mic feedback false positives.
        if self._tts_started_at is not None:
            elapsed = time.monotonic() - self._tts_started_at
            if elapsed < TTS_BARGE_IN_GRACE_SECONDS:
                return
            elapsed_ms = int(elapsed * 1000)
        else:
            # Defensive: SPEAKING reached without _tts_started_at set
            # shouldn't happen, but if it does we log it visibly.
            elapsed_ms = -1
        log.info("[barge-in] triggered after %dms of TTS playback", elapsed_ms)
        await self._cancel_response_chain()
        self._listen_buffer.clear()
        self._vad.reset()
        # SPEAKING -> LISTENING: legal, and the user clearly wants to talk.
        self._sm.set_conversational_state(ConversationalState.LISTENING)
        self._start_listening_timer()

    # -- response chain --

    async def _run_response_chain(self, audio: bytes) -> None:
        try:
            text = await self._stt.transcribe(audio)
            duration_ms = (len(audio) // 2) * 1000 // SAMPLE_RATE
            self._bus.publish(TranscriptionReady(text=text, duration_ms=duration_ms))

            if not text.strip():
                # No speech; abort cleanly back to IDLE.
                self._sm.set_conversational_state(ConversationalState.IDLE)
                return

            # Tee the producer into speak_stream. Each chunk yielded by
            # the producer is recorded for LLMResponseChunk emission and
            # full-text assembly, then handed through to PiperTTS.
            # speak_stream which buffers up to a sentence boundary
            # internally -- avoids the per-token synthesis stutter that
            # `for chunk: await tts.speak(chunk)` would produce on
            # streaming LLM responses. Cancellation propagates through
            # the iterator the same way: a cancel on this task raises
            # CancelledError inside speak_stream's __anext__, which
            # cascades into the producer (-> ollama -> httpx context
            # manager close).
            full_text_parts: list[str] = []
            first_chunk_seen = False

            async def teed_chunks():
                nonlocal first_chunk_seen
                async for chunk in self._response_producer(text):
                    if not first_chunk_seen:
                        self._sm.set_conversational_state(
                            ConversationalState.SPEAKING
                        )
                        self._tts_started_at = time.monotonic()
                        first_chunk_seen = True
                    full_text_parts.append(chunk)
                    self._bus.publish(LLMResponseChunk(text=chunk))
                    yield chunk

            await self._tts.speak_stream(teed_chunks())

            if not first_chunk_seen:
                # Producer yielded nothing; nothing to speak.
                self._sm.set_conversational_state(ConversationalState.IDLE)
                return

            self._bus.publish(
                LLMResponseComplete(full_text="".join(full_text_parts))
            )
            self._sm.set_conversational_state(ConversationalState.IDLE)
        except asyncio.CancelledError:
            # Barge-in or shutdown. Re-raise so the outer awaiter sees it.
            raise
        except Exception:
            log.exception("response chain crashed")
            # Don't leave the SM stuck in THINKING/SPEAKING.
            try:
                self._sm.set_conversational_state(ConversationalState.IDLE)
            except Exception:
                log.exception("recovery to IDLE failed")
        finally:
            self._tts_started_at = None

    async def _cancel_response_chain(self) -> None:
        # Order per design note: TTS first (user-perceptible), then drain
        # internal state, then cancel the producer task.
        try:
            await self._tts.cancel()
        except Exception:
            log.exception("tts.cancel failed during cancellation")
        if self._response_task is not None and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("response task raised during cancellation")
        self._response_task = None
