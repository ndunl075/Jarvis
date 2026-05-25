"""Tests for jarvis.audio.pipeline orchestration logic.

All stages are in-memory fakes; no real audio. Stage-specific behavior
(openWakeWord detection, silero VAD, whisper STT, Piper TTS) is covered
by per-module tests against recorded WAV fixtures (Phase 2 Tasks 2-6).
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.audio.pipeline import AudioPipeline
from jarvis.audio.protocols import (
    FRAME_BYTES,
    FRAME_DURATION_MS,
    POST_WAKE_BLACKOUT_MS,
    TTS_BARGE_IN_GRACE_SECONDS,
    AudioFrame,
    AudioSource,
    SpeechToText,
    TextToSpeech,
    VADEvent,
    VoiceActivityDetector,
    WakeWordDetector,
    WakeWordResult,
)
from jarvis.core.events import (
    EventBus,
    LLMResponseComplete,
    TranscriptionReady,
    WakeWordDetected,
)
from jarvis.core.state_machine import ConversationalState, Mode, StateMachine

SILENCE_FRAME: AudioFrame = b"\x00" * FRAME_BYTES


# --- fakes ----------------------------------------------------------------


class FakeSource:
    def __init__(self) -> None:
        self._on_frame = None
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self, on_frame):
        self._on_frame = on_frame
        self.start_calls += 1

    async def stop(self):
        self.stop_calls += 1

    def push(self, frame: AudioFrame = SILENCE_FRAME) -> None:
        if self._on_frame is not None:
            self._on_frame(frame)


class FakeWakeWord:
    def __init__(self, detect_at: int | list[int] | None = None) -> None:
        self.feed_count = 0
        if detect_at is None:
            self._detect_counts: set[int] = set()
        elif isinstance(detect_at, int):
            self._detect_counts = {detect_at}
        else:
            self._detect_counts = set(detect_at)
        self.reset_count = 0

    async def feed(self, frame):
        self.feed_count += 1
        if self.feed_count in self._detect_counts:
            return WakeWordResult(confidence=0.95)
        return None

    def reset(self):
        self.reset_count += 1


class FakeVAD:
    def __init__(self, script: dict[int, VADEvent] | None = None) -> None:
        self.feed_count = 0
        self.script = script or {}
        self.reset_count = 0

    async def feed(self, frame):
        self.feed_count += 1
        return self.script.get(self.feed_count)

    def reset(self):
        self.reset_count += 1


class FakeSTT:
    def __init__(self, transcription: str = "hello world") -> None:
        self.transcription = transcription
        self.calls = 0
        self.last_audio: bytes | None = None

    async def transcribe(self, audio):
        self.calls += 1
        self.last_audio = audio
        return self.transcription


class FakeTTS:
    """A controllable TTS that blocks each speak/speak_stream chunk
    until finish_speaking() or cancel() is called. Test assertions on
    `spoken` see the raw chunks fed in (no sentence segmentation -- the
    real PiperTTS handles that; here we want deterministic per-chunk
    visibility)."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.cancel_count = 0
        self._gates: list[asyncio.Event] = []

    async def speak(self, text):
        self.spoken.append(text)
        gate = asyncio.Event()
        self._gates.append(gate)
        await gate.wait()

    async def speak_stream(self, text_chunks):
        # Pipeline path: iterate the (tee'd) producer and "speak" each
        # chunk in turn, blocking on a gate per chunk so cancellation
        # tests can interrupt mid-stream.
        async for chunk in text_chunks:
            self.spoken.append(chunk)
            gate = asyncio.Event()
            self._gates.append(gate)
            await gate.wait()

    async def cancel(self):
        self.cancel_count += 1
        for gate in self._gates:
            gate.set()
        self._gates.clear()

    def finish_speaking(self) -> None:
        for gate in self._gates:
            gate.set()
        self._gates.clear()


def make_response_producer(*chunks: str):
    async def producer(text: str):
        for c in chunks:
            yield c
    return producer


def make_pipeline(
    *,
    detect_at: int | list[int] | None = None,
    vad_script: dict[int, VADEvent] | None = None,
    transcription: str = "hello world",
    response_chunks: tuple[str, ...] = ("hello back",),
    listening_timeout: float = 6.0,
    frame_queue_maxsize: int = 32,
    # Both default to 0 in the helper so existing tests' VAD scripts
    # exercise the pipeline without blackout / minimum-length gating.
    # Tests that specifically verify those features pass realistic values.
    post_wake_blackout_ms: int = 0,
    min_utterance_ms: int = 0,
    # Barge-in is disabled by default in production (speaker->mic feedback),
    # but most pipeline tests want to exercise the path. Opt-in default True
    # here keeps the existing barge-in tests assertion-shaped; the tripwire
    # test below explicitly exercises False.
    barge_in_enabled: bool = True,
    on_wake=None,
):
    bus = EventBus()
    sm = StateMachine(bus=bus)
    source = FakeSource()
    ww = FakeWakeWord(detect_at=detect_at)
    vad = FakeVAD(script=vad_script)
    stt = FakeSTT(transcription=transcription)
    tts = FakeTTS()
    pipeline = AudioPipeline(
        source=source, wake_word=ww, vad=vad, stt=stt, tts=tts,
        response_producer=make_response_producer(*response_chunks),
        bus=bus, sm=sm,
        listening_timeout_seconds=listening_timeout,
        frame_queue_maxsize=frame_queue_maxsize,
        post_wake_blackout_ms=post_wake_blackout_ms,
        min_utterance_ms=min_utterance_ms,
        barge_in_enabled=barge_in_enabled,
        on_wake=on_wake,
    )
    return pipeline, bus, sm, source, ww, vad, stt, tts


async def yield_loop(times: int = 20) -> None:
    """Hand control back to the loop repeatedly so scheduled tasks run."""
    for _ in range(times):
        await asyncio.sleep(0)


async def wait_for_cs(sm: StateMachine, target: ConversationalState, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while sm.conversational_state is not target:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(
                f"timed out waiting for CS={target.name}, "
                f"current={sm.conversational_state.name}"
            )
        await asyncio.sleep(0.005)
    # SM state is updated synchronously but bus dispatch (subscribers) is
    # async. Yield so any subscriber-observed side effects of reaching
    # `target` (event handlers, follow-on transitions) can run before the
    # caller's assertions.
    await yield_loop()


# --- protocol smoke -------------------------------------------------------


def test_fakes_satisfy_protocols():
    assert isinstance(FakeSource(), AudioSource)
    assert isinstance(FakeWakeWord(), WakeWordDetector)
    assert isinstance(FakeVAD(), VoiceActivityDetector)
    assert isinstance(FakeSTT(), SpeechToText)
    assert isinstance(FakeTTS(), TextToSpeech)


# --- core orchestration ---------------------------------------------------


async def test_idle_to_listening_on_wake_detection():
    pipeline, bus, sm, source, ww, vad, *_ = make_pipeline(detect_at=2)
    seen: list[WakeWordDetected] = []
    bus.subscribe(WakeWordDetected, lambda e: seen.append(e))

    await pipeline.start()
    try:
        source.push()
        source.push()  # detection on this one
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert ww.reset_count == 1
        assert vad.reset_count == 1
        assert len(seen) == 1
        assert seen[0].confidence == 0.95
    finally:
        await pipeline.stop()


async def test_no_wake_word_no_transition():
    pipeline, bus, sm, source, ww, *_ = make_pipeline(detect_at=None)
    await pipeline.start()
    try:
        for _ in range(10):
            source.push()
        await yield_loop()
        assert sm.conversational_state is ConversationalState.IDLE
        assert ww.feed_count == 10
    finally:
        await pipeline.stop()


async def test_listening_endpoint_triggers_stt_and_response_chain():
    # detect on frame 1; VAD endpoint on frame 3 (during LISTENING).
    pipeline, bus, sm, source, _, _, stt, tts = make_pipeline(
        detect_at=1,
        vad_script={2: VADEvent.SPEECH_STARTED, 4: VADEvent.ENDPOINT},
        transcription="what time is it",
        response_chunks=("the time is now",),
    )
    transcripts: list[TranscriptionReady] = []
    completes: list[LLMResponseComplete] = []
    bus.subscribe(TranscriptionReady, lambda e: transcripts.append(e))
    bus.subscribe(LLMResponseComplete, lambda e: completes.append(e))

    await pipeline.start()
    try:
        # Frame 1: wake -> LISTENING, buffer cleared
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        # Frames 2..4 in LISTENING. VAD frame indices restart per stage's
        # own counter; stage's frame 2 = SPEECH_STARTED, frame 4 = ENDPOINT.
        # We need to push enough frames AFTER LISTENING starts.
        for _ in range(4):
            source.push()
        await wait_for_cs(sm, ConversationalState.SPEAKING)
        assert stt.calls == 1
        assert stt.last_audio is not None
        assert len(stt.last_audio) > 0
        assert len(transcripts) == 1
        assert transcripts[0].text == "what time is it"
        assert tts.spoken == ["the time is now"]

        # Let TTS complete naturally.
        tts.finish_speaking()
        await wait_for_cs(sm, ConversationalState.IDLE)
        assert len(completes) == 1
        assert completes[0].full_text == "the time is now"
    finally:
        await pipeline.stop()


async def test_empty_transcription_returns_to_idle():
    pipeline, _, sm, source, *_ = make_pipeline(
        detect_at=1,
        vad_script={2: VADEvent.ENDPOINT},
        transcription="   ",  # whitespace only
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        for _ in range(2):
            source.push()
        await wait_for_cs(sm, ConversationalState.IDLE)
    finally:
        await pipeline.stop()


async def test_response_producer_yielding_nothing_returns_to_idle():
    pipeline, _, sm, source, *_ = make_pipeline(
        detect_at=1,
        vad_script={2: VADEvent.ENDPOINT},
        response_chunks=(),  # producer yields nothing
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        for _ in range(2):
            source.push()
        await wait_for_cs(sm, ConversationalState.IDLE)
    finally:
        await pipeline.stop()


# --- barge-in -------------------------------------------------------------


async def test_barge_in_after_grace_cancels_tts_and_returns_to_listening():
    # Frame indexing for VAD: not fed during IDLE, so:
    #   push 1 (IDLE wake)        -> VAD count=0
    #   push 2 (LIS frame 1)      -> VAD count=1
    #   push 3 (LIS frame 2)      -> VAD count=2  -> ENDPOINT
    #   push 4 (SPEAKING frame 1) -> VAD count=3  -> SPEECH_STARTED (barge-in)
    pipeline, _, sm, source, _, vad, _, tts = make_pipeline(
        detect_at=1,
        vad_script={2: VADEvent.ENDPOINT, 3: VADEvent.SPEECH_STARTED},
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        source.push()
        await wait_for_cs(sm, ConversationalState.SPEAKING)
        assert tts.spoken == ["hello back"]

        # Wait past the grace window before pushing the barge-in frame.
        await asyncio.sleep(TTS_BARGE_IN_GRACE_SECONDS + 0.05)
        source.push()

        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert tts.cancel_count >= 1
    finally:
        await pipeline.stop()


async def test_barge_in_within_grace_is_ignored():
    pipeline, _, sm, source, _, vad, _, tts = make_pipeline(
        detect_at=1,
        vad_script={2: VADEvent.ENDPOINT, 3: VADEvent.SPEECH_STARTED},
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        source.push()
        await wait_for_cs(sm, ConversationalState.SPEAKING)

        # Push barge-in frame immediately, well within grace window.
        source.push()
        await yield_loop()

        assert sm.conversational_state is ConversationalState.SPEAKING
        assert tts.cancel_count == 0
    finally:
        await pipeline.stop()


# --- listening silence timeout -------------------------------------------


async def test_listening_silence_timeout_returns_to_idle():
    pipeline, _, sm, source, *_ = make_pipeline(
        detect_at=1, listening_timeout=0.1
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        # Don't push more frames; wait past the timeout.
        await asyncio.sleep(0.2)
        assert sm.conversational_state is ConversationalState.IDLE
    finally:
        await pipeline.stop()


async def test_listening_timer_does_not_rearm_after_speech_started():
    # Tripwire for: re-arming the listening timer mid-utterance would force
    # IDLE during a long pause between words and orphan the listen buffer.
    # Once SPEECH_STARTED fires, the timer must stay cancelled until we
    # leave LISTENING.
    pipeline, _, sm, source, _, vad, *_ = make_pipeline(
        detect_at=1,
        # SPEECH_STARTED on first LISTENING frame; ENDPOINT much later.
        vad_script={1: VADEvent.SPEECH_STARTED, 2: VADEvent.ENDPOINT},
        listening_timeout=0.05,  # very short
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()  # SPEECH_STARTED, cancels timer
        await yield_loop()
        # Long pause that exceeds the listening_timeout. If the timer were
        # re-armed (by any path), this sleep would push CS to IDLE.
        await asyncio.sleep(0.15)
        assert sm.conversational_state is ConversationalState.LISTENING
        # Now ENDPOINT should still work and transition to THINKING/SPEAKING.
        source.push()
        await wait_for_cs(sm, ConversationalState.SPEAKING)
    finally:
        await pipeline.stop()


async def test_speech_started_cancels_listening_timeout():
    # Push 1 (IDLE wake) -> VAD count=0
    # Push 2 (LIS frame 1) -> VAD count=1 -> SPEECH_STARTED, cancels timer.
    pipeline, _, sm, source, _, vad, *_ = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.SPEECH_STARTED},
        listening_timeout=0.1,
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()  # SPEECH_STARTED, cancels the deadline
        await yield_loop()
        await asyncio.sleep(0.2)  # past what would have been the deadline
        assert sm.conversational_state is ConversationalState.LISTENING
    finally:
        await pipeline.stop()


# --- frame routing gates --------------------------------------------------


async def test_frames_dropped_when_mode_not_active():
    pipeline, _, sm, source, ww, *_ = make_pipeline(detect_at=2)
    await pipeline.start()
    try:
        sm.set_mode(Mode.MUTED)
        await yield_loop()
        for _ in range(5):
            source.push()
        await yield_loop()
        # Wake word never sees any frame because Mode != ACTIVE drops them.
        assert ww.feed_count == 0
        assert sm.conversational_state is ConversationalState.IDLE
    finally:
        await pipeline.stop()


async def test_wrong_size_frames_are_logged_and_dropped(
    caplog: pytest.LogCaptureFixture,
):
    import logging

    pipeline, _, _, source, ww, *_ = make_pipeline()
    await pipeline.start()
    try:
        with caplog.at_level(logging.ERROR, logger="jarvis.audio.pipeline"):
            source.push(b"\x00" * 100)  # wrong size
        await yield_loop()
        assert ww.feed_count == 0
        assert any("wrong frame size" in r.message for r in caplog.records)
    finally:
        await pipeline.stop()


async def test_full_frame_queue_drops_oldest(
    caplog: pytest.LogCaptureFixture,
):
    import logging

    pipeline, _, sm, source, ww, *_ = make_pipeline(
        detect_at=None, frame_queue_maxsize=2
    )
    # Hold the frame loop blocked by leaving Mode != ACTIVE so the loop
    # reads frames but does no stage work; meanwhile fill the queue from
    # the source side faster than the loop can consume.
    # Simpler: don't start() (no consumer). Just hammer the enqueue path.
    pipeline._loop = asyncio.get_running_loop()  # set what start() would
    with caplog.at_level(logging.WARNING, logger="jarvis.audio.pipeline"):
        for _ in range(10):
            source._on_frame = pipeline._on_audio_frame  # wire the bridge
            source.push()
        await yield_loop()
    assert pipeline._q_frames.qsize() == 2
    assert any("dropped oldest" in r.message for r in caplog.records)


# --- start/stop lifecycle ------------------------------------------------


async def test_start_is_idempotent():
    pipeline, *_, source, ww, _vad, _stt, _tts = make_pipeline(detect_at=None)
    await pipeline.start()
    await pipeline.start()  # second call is a no-op
    try:
        assert source.start_calls == 1
    finally:
        await pipeline.stop()


# --- Phase 2 polish: blackout (Bug A) and short-utterance gate (Bug D) -


async def test_post_wake_blackout_default_swallows_wake_tail():
    """POST_WAKE_BLACKOUT_MS = 100: enough to drop the trailing tail of
    "...jarvis" without clipping the first word of a short command.
    Lowered from 200 ms after live testing showed whisper clipping first
    syllables ("Count" → "councillor"). Tripwire if someone drops it to 0
    without rethinking the trade-off."""
    assert POST_WAKE_BLACKOUT_MS == 100

    blackout_frames = -(-POST_WAKE_BLACKOUT_MS // FRAME_DURATION_MS)
    pipeline, _, sm, source, _, vad, *_ = make_pipeline(
        detect_at=1,
        post_wake_blackout_ms=POST_WAKE_BLACKOUT_MS,
    )
    await pipeline.start()
    try:
        source.push()  # wake -> LISTENING
        await wait_for_cs(sm, ConversationalState.LISTENING)
        # Push exactly `blackout_frames` frames; none should reach VAD or
        # the listen buffer.
        for _ in range(blackout_frames):
            source.push()
        await yield_loop()
        assert vad.feed_count == 0
        assert len(pipeline._listen_buffer) == 0  # type: ignore[attr-defined]
        # The very next frame is post-blackout and must reach VAD.
        source.push()
        await yield_loop()
        assert vad.feed_count == 1
    finally:
        await pipeline.stop()


async def test_barge_in_disabled_by_default_in_production():
    """Tripwire: production constant must stay False until AEC ships."""
    from jarvis.audio.pipeline import BARGE_IN_ENABLED
    assert BARGE_IN_ENABLED is False


async def test_barge_in_disabled_skips_vad_during_speaking():
    """When the pipeline is constructed with barge_in_enabled=False, frames
    that arrive in SPEAKING state must NOT be fed to VAD (saves CPU and,
    more importantly, prevents speaker->mic feedback from ever triggering
    SPEECH_STARTED while TTS is playing)."""
    pipeline, _, sm, source, _, vad, _, tts = make_pipeline(
        detect_at=1,
        vad_script={2: VADEvent.ENDPOINT, 3: VADEvent.SPEECH_STARTED},
        barge_in_enabled=False,
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()
        source.push()
        await wait_for_cs(sm, ConversationalState.SPEAKING)
        vad_count_at_speaking_start = vad.feed_count

        # SPEAKING-state frames: scripted to fire SPEECH_STARTED at count=3.
        # With barge-in disabled, the frames must be dropped before VAD,
        # so the count stays put and TTS is never cancelled.
        for _ in range(5):
            source.push()
        await yield_loop()

        assert vad.feed_count == vad_count_at_speaking_start
        assert sm.conversational_state is ConversationalState.SPEAKING
        assert tts.cancel_count == 0
    finally:
        await pipeline.stop()


# --- wake-word activation policy ----------------------------------------


async def test_on_wake_hook_fires_at_wake_detection():
    """The composition root wires Conversation.clear() here so every
    'hey jarvis' starts with an empty history. Verifies the hook
    actually runs at the IDLE -> LISTENING transition."""
    calls: list[int] = []

    def on_wake() -> None:
        calls.append(1)

    pipeline, _, sm, source, *_ = make_pipeline(
        detect_at=1, on_wake=on_wake,
    )
    await pipeline.start()
    try:
        assert calls == []
        source.push()  # wake -> LISTENING
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert calls == [1]
    finally:
        await pipeline.stop()


async def test_on_wake_hook_exception_does_not_block_listening_transition():
    """A busted hook must not stop the user from being heard. Exception
    is logged and swallowed; the state transition still happens."""
    def on_wake() -> None:
        raise RuntimeError("intentional test boom")

    pipeline, _, sm, source, *_ = make_pipeline(
        detect_at=1, on_wake=on_wake,
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert sm.conversational_state is ConversationalState.LISTENING
    finally:
        await pipeline.stop()


async def test_conversation_clear_on_wake_wipes_prior_history():
    """End-to-end: a tool-call interaction is recorded, then a wake
    fires, then a new user turn lands. The messages sent to the LLM on
    the new turn must be [system, user='hello'] — no leftover assistant
    text from the previous interaction. This is the fix for qwen2.5's
    tendency to anchor on recent context and re-fire the same tool."""
    from jarvis.llm.conversation import Conversation
    conv = Conversation(system_prompt_provider=lambda: "sys")
    # Simulate the previous interaction: user turn + LLM-narration
    # assistant turn (which is what gets stored under the post-fix
    # router rules — tool result strings never reach Conversation).
    conv.add_user_turn("what's CPU usage?")
    conv.add_assistant_turn("Let me check, sir.")
    assert len(conv._turns) == 2  # type: ignore[attr-defined]

    # Pipeline wired with conv.clear as on_wake.
    pipeline, _, sm, source, *_ = make_pipeline(
        detect_at=1, on_wake=conv.clear,
    )
    await pipeline.start()
    try:
        source.push()  # wake fires
        await wait_for_cs(sm, ConversationalState.LISTENING)
    finally:
        await pipeline.stop()

    # Wake-word handler cleared history. Next user turn sees a clean slate.
    msgs = conv.add_user_turn("hello")
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]


async def test_post_wake_blackout_when_explicitly_enabled_still_discards():
    """The blackout implementation remains in place for Phase 5. Verify
    that a non-zero override discards frames during the window and lets
    them through after."""
    blackout_ms = 400
    pipeline, _, sm, source, _, vad, *_ = make_pipeline(
        detect_at=1,
        post_wake_blackout_ms=blackout_ms,
    )
    blackout_frames = -(-blackout_ms // FRAME_DURATION_MS)  # ceil

    await pipeline.start()
    try:
        source.push()  # wake -> LISTENING
        await wait_for_cs(sm, ConversationalState.LISTENING)

        for _ in range(blackout_frames):
            source.push()
        await yield_loop()
        assert vad.feed_count == 0
        assert len(pipeline._listen_buffer) == 0  # type: ignore[attr-defined]

        source.push()
        await yield_loop()
        assert vad.feed_count == 1
        assert len(pipeline._listen_buffer) == FRAME_BYTES  # type: ignore[attr-defined]
    finally:
        await pipeline.stop()


async def test_short_utterance_returns_to_idle_without_calling_stt():
    """ENDPOINT fires with only ~60 ms in the buffer (< 500 ms minimum):
    transition to IDLE silently, do NOT spend an STT call."""
    pipeline, _, sm, source, _, _, stt, _ = make_pipeline(
        detect_at=1,
        # Frame 1 of LISTENING -> SPEECH_STARTED; frame 2 -> ENDPOINT.
        # Buffer at endpoint = 2 frames = 60 ms (well below 500 ms).
        vad_script={1: VADEvent.SPEECH_STARTED, 2: VADEvent.ENDPOINT},
        min_utterance_ms=500,
    )
    await pipeline.start()
    try:
        source.push()  # wake
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()  # SPEECH_STARTED
        source.push()  # ENDPOINT -> short -> IDLE
        await wait_for_cs(sm, ConversationalState.IDLE)
        assert stt.calls == 0, "STT was called on a too-short utterance"
    finally:
        await pipeline.stop()


async def test_long_enough_utterance_proceeds_to_stt():
    """Inverse of the gate: with enough buffered audio at ENDPOINT,
    STT runs and we transition to SPEAKING."""
    # Need >= 500 ms = 17 frames at 30 ms. We'll give it 20 frames of
    # SPEECH state then an ENDPOINT.
    pipeline, _, sm, source, _, _, stt, _ = make_pipeline(
        detect_at=1,
        vad_script={1: VADEvent.SPEECH_STARTED, 21: VADEvent.ENDPOINT},
        min_utterance_ms=500,
    )
    await pipeline.start()
    try:
        source.push()
        await wait_for_cs(sm, ConversationalState.LISTENING)
        for _ in range(21):
            source.push()
        await wait_for_cs(sm, ConversationalState.SPEAKING)
        assert stt.calls == 1
    finally:
        await pipeline.stop()


async def test_stop_cancels_response_chain():
    pipeline, _, sm, source, *_ = make_pipeline(
        detect_at=1,
        vad_script={2: VADEvent.ENDPOINT},
    )
    await pipeline.start()
    source.push()
    await wait_for_cs(sm, ConversationalState.LISTENING)
    source.push()
    source.push()
    await wait_for_cs(sm, ConversationalState.SPEAKING)

    await pipeline.stop()
    # After stop, the response task must be cancelled.
    assert pipeline._response_task is None


# --- wake-word interrupt (new barge-in design) ----------------------------


async def test_wake_word_during_speaking_cancels_tts_and_transitions_to_listening():
    """Saying 'hey jarvis' while Jarvis is speaking must abort playback
    immediately and return to LISTENING so the new query can be heard.
    VAD barge-in stays disabled; this path is purely wake-word driven."""
    # ww fires at count=1 (IDLE→LISTENING) and count=3 (SPEAKING interrupt).
    # VAD endpoint on the first LISTENING frame to reach SPEAKING quickly.
    pipeline, _, sm, source, ww, _, _, tts = make_pipeline(
        detect_at=[1, 3],
        vad_script={1: VADEvent.ENDPOINT},
        barge_in_enabled=False,
    )
    await pipeline.start()
    try:
        source.push()  # frame 1: ww count=1 → wake → LISTENING
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()  # frame 2: ww count=2, VAD ENDPOINT → THINKING → response chain
        await wait_for_cs(sm, ConversationalState.SPEAKING)
        assert tts.spoken  # TTS has started

        source.push()  # frame 3: ww count=3 → wake fires during SPEAKING → interrupt
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert tts.cancel_count >= 1, "TTS must be cancelled on wake interrupt"
        assert ww.reset_count == 2  # reset called on both wake detections
    finally:
        await pipeline.stop()


async def test_wake_word_during_thinking_cancels_response_chain_and_transitions_to_listening():
    """Wake word during THINKING (STT/LLM in-flight) triggers interrupt.
    Unlikely in practice but handled as a safety case.

    Frames 2 and 3 are pushed together (no await between them) so both are in
    the queue when the frame loop handles frame 2. _handle_endpoint creates the
    response task (not yet running) and calls get() immediately; frame 3 is
    ready so get() returns without yielding to the event loop. The response task
    never gets CPU time before frame 3 is processed — CS is still THINKING.
    """
    pipeline, _, sm, source, _, _, _, tts = make_pipeline(
        detect_at=[1, 3],
        vad_script={1: VADEvent.ENDPOINT},
        barge_in_enabled=False,
    )
    await pipeline.start()
    try:
        source.push()  # frame 1: wake → LISTENING (ww count=1)
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()  # frame 2: VAD ENDPOINT → THINKING, response task scheduled (ww count=2)
        source.push()  # frame 3: in queue before task runs → wake fires in THINKING (ww count=3)
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert tts.cancel_count >= 1  # safety cancel issued
    finally:
        await pipeline.stop()


async def test_wake_word_during_idle_does_not_call_cancel():
    """Normal IDLE → LISTENING transition must not cancel TTS (nothing playing)."""
    pipeline, _, sm, source, _, _, _, tts = make_pipeline(detect_at=1)
    await pipeline.start()
    try:
        source.push()  # wake from IDLE → LISTENING
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert tts.cancel_count == 0
    finally:
        await pipeline.stop()


async def test_on_wake_hook_fires_on_speaking_interrupt():
    """on_wake (conversation.clear) must be called every time the wake word
    fires, including mid-SPEAKING interrupts, to wipe prior context."""
    calls: list[str] = []

    def on_wake() -> None:
        calls.append("wake")

    pipeline, _, sm, source, _, _, _, _ = make_pipeline(
        detect_at=[1, 3],
        vad_script={1: VADEvent.ENDPOINT},
        barge_in_enabled=False,
        on_wake=on_wake,
    )
    await pipeline.start()
    try:
        source.push()  # frame 1: wake from IDLE
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert calls == ["wake"]

        source.push()  # frame 2: VAD ENDPOINT → response chain
        await wait_for_cs(sm, ConversationalState.SPEAKING)

        source.push()  # frame 3: wake during SPEAKING → interrupt
        await wait_for_cs(sm, ConversationalState.LISTENING)
        assert calls == ["wake", "wake"]  # cleared again on interrupt
    finally:
        await pipeline.stop()


async def test_wake_word_fed_during_speaking_state():
    """Wake word detector must receive frames even while SPEAKING so the
    interrupt path is reachable. Verifies feed_count grows in SPEAKING."""
    pipeline, _, sm, source, ww, _, _, _ = make_pipeline(
        detect_at=1,  # fires only once (IDLE→LISTENING); never during SPEAKING
        vad_script={1: VADEvent.ENDPOINT},
        barge_in_enabled=False,
    )
    await pipeline.start()
    try:
        source.push()  # wake → LISTENING
        await wait_for_cs(sm, ConversationalState.LISTENING)
        source.push()  # VAD ENDPOINT → SPEAKING
        await wait_for_cs(sm, ConversationalState.SPEAKING)
        count_before = ww.feed_count

        for _ in range(5):
            source.push()
        await yield_loop()

        assert ww.feed_count == count_before + 5, (
            "wake word must be fed every frame in SPEAKING state"
        )
    finally:
        await pipeline.stop()
