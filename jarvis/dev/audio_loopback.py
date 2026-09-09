"""End-to-end audio loopback for Phase 2 manual verification.

.. note::
    **Legacy harness.** The real app now launches via ``python -m jarvis``
    (or the ``jarvis`` entry point after ``pip install -e .``), which uses
    ``jarvis/app.py`` as the composition root.  Keep this file for isolated
    audio-stack debugging — it lets you test wake-word → VAD → STT → TTS
    end-to-end without the Qt event loop and tray overhead.

Composes the full audio stack (input source -> wake word -> VAD -> STT ->
echo -> TTS) on top of the real EventBus, StateMachine, and
LifecycleManager. Runnable as:

    python -m jarvis.dev.audio_loopback

This file doubles as documentation for the composition root that
__main__.py will eventually formalize. The pattern is:
  1. Construct bus and state machine.
  2. Construct each stage (audio source + 4 ML modules).
  3. Construct the lifecycle manager with stages in declared order.
  4. Subscribe printers to bus events for visibility.
  5. EXPLICITLY load_all() so any prerequisite failure surfaces clearly.
  6. bind() lifecycle to bus so subsequent Mode transitions
     (mute/sleep/wake) drive load/unload automatically.
  7. Start the pipeline, wait for Ctrl+C, shut down in reverse order.

Done-criterion (BUILD.md Phase 2): say "hey jarvis, hello world" and hear
the transcription spoken back. Barge-in works: interrupting the TTS by
speaking stops it.

Prerequisites
-------------
1. openWakeWord models (one-time, ~5 MB):
       python -c "import openwakeword; openwakeword.utils.download_models()"

2. silero-vad ONNX: ships with the silero-vad pip package; nothing to do.

3. faster-whisper base.en model (~150 MB):
       Downloads automatically on first transcription. Be patient on the
       first launch, or pre-pull with:
           python -c "from faster_whisper import WhisperModel; \\
                      WhisperModel('base.en', device='cpu', compute_type='int8')"

4. Piper voice (en_GB-alan-medium, ~63 MB):
       Download en_GB-alan-medium.onnx AND en_GB-alan-medium.onnx.json
       from https://github.com/rhasspy/piper/releases (or the HF mirror)
       and place both in:
           $JARVIS_PIPER_VOICES_DIR  (env var)
       or, if unset:
           ~/.jarvis/voices/

5. Microphone: ReSpeaker auto-detected if connected; otherwise the system
   default input device. Headphones recommended -- the TTS-feedback grace
   window in the pipeline mitigates speaker-to-mic feedback but isn't a
   replacement for actual acoustic isolation.

First-launch latency
--------------------
Plan for ~10-30 seconds of model loading on the first run after a clean
install (whisper download + ONNX runtime warmup). Subsequent launches
are ~3-5 s.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from jarvis.audio.devices import AudioInputSource
from jarvis.audio.pipeline import AudioPipeline
from jarvis.audio.stt import FasterWhisperSTT
from jarvis.audio.tts import PiperTTS
from jarvis.audio.vad import SileroVAD
from jarvis.audio.wake_word import OpenWakeWord
from jarvis.core.config import load_config
from jarvis.core.events import (
    ConversationalStateChanged,
    EventBus,
    LLMResponseChunk,
    LLMResponseComplete,
    ModeChanged,
    TranscriptionReady,
    WakeWordDetected,
)
from jarvis.core.lifecycle import LifecycleManager
from jarvis.core.state_machine import ConversationalState, StateMachine
from jarvis.llm.conversation import Conversation
from jarvis.llm.intent_router import (
    IntentRouter,
    execute_intent,
)
from jarvis.llm.ollama_client import OllamaClient
from jarvis.tools import ToolRegistry, setup_local_tools

log = logging.getLogger(__name__)

PIPER_VOICE_NAME = "en_GB-alan-medium"

# TEMPORARY: hardcode your mic name substring here until the Phase 5
# settings UI lands. Faster than editing %APPDATA%/Jarvis/config.json
# every time. Case-insensitive substring match against the input device
# list. Takes precedence over config.audio.input_device when set.
# Set to None to use config (or fall through to the system default).
INPUT_DEVICE: str | None = "TONOR"

# TEMPORARY: hardcode your speaker name substring here until the Phase 5
# settings UI lands. Faster than editing %APPDATA%/Jarvis/config.json
# every time. Case-insensitive substring match against the output device
# list. Takes precedence over config.audio.output_device when set.
# Set to None to use config (or fall through to the system default).
OUTPUT_DEVICE: str | None = "Pebble Pro"

# TEMPORARY: speak rate multiplier passed through to Piper. >1 speeds up,
# <1 slows down. 1.15 nudges the medium voice closer to natural conversational
# pace without sounding rushed; raise toward 1.3 for a snappier feel.
# Wired to PiperVoice via SynthesisConfig(length_scale=1/speed) — Piper's
# native rate-control knob, no sample-domain time stretching needed.
SPEAK_SPEED: float = 1.15

# TEMPORARY (diagnostics): override VAD's speech-start threshold for tuning
# the user's mic input level. None uses SileroVAD's default (0.5). Lower
# values (e.g. 0.3) make the detector fire on quieter input; raise if
# false-triggers occur. Once a sweet spot is found, bake it into config or
# tune the default in audio/vad.py.
VAD_THRESHOLD: float | None = None

# TEMPORARY (diagnostics): every Nth VAD inference window prints its raw
# probability + state classification. ~6 Hz at N=5 (windows are ~32 ms).
# Set to 0 to disable.
VAD_LOG_EVERY_N_WINDOWS: int = 0

# --- composition root helpers -------------------------------------------


def _voices_dir() -> Path:
    """Resolve the Piper voices directory. Env var override, else
    ~/.jarvis/voices/."""
    env = os.environ.get("JARVIS_PIPER_VOICES_DIR")
    if env:
        return Path(env)
    return Path.home() / ".jarvis" / "voices"


def _describe_stream_device(stream) -> str:
    """Return "<device_name> [hostapi=<api_name>]" for an open sounddevice
    stream, or a best-effort fallback string. Used at boot to verify that
    the WASAPI-preference logic is actually selecting the WASAPI variant
    rather than silently falling back to MME."""
    import sounddevice as sd
    try:
        dev_arg = stream.device
        # stream.device may be (in_idx, out_idx) for duplex; we only open
        # one-way streams here, so it's a scalar int.
        if isinstance(dev_arg, (list, tuple)):
            dev_arg = next((d for d in dev_arg if d is not None), None)
        if dev_arg is None:
            return "<system default> [hostapi=?]"
        info = sd.query_devices(dev_arg)
        name = info.get("name", "?")
        api = sd.query_hostapis(info.get("hostapi", -1)).get("name", "?")
        return f"{name} [hostapi={api}]"
    except Exception as e:
        return f"<unresolved: {e}>"


def _print_resolved_devices(source: AudioInputSource, tts: PiperTTS) -> None:
    in_stream = getattr(source, "_stream", None)
    out_stream = getattr(tts, "_stream", None)
    in_desc = _describe_stream_device(in_stream) if in_stream else "<not open>"
    out_desc = _describe_stream_device(out_stream) if out_stream else "<not open>"
    print(f"[boot] resolved input:  {in_desc}")
    print(f"[boot] resolved output: {out_desc}")


def _print_resolved_models(wake_word: OpenWakeWord, vad: SileroVAD) -> None:
    """Print the actually-loaded ONNX file paths so a stale system-Python
    install masquerading as the venv copy is visible at boot rather than
    after a confusing runtime error."""
    ww_path = getattr(wake_word, "model_path", None)
    vad_path = getattr(vad, "model_path", None)
    print(f"[boot] openwakeword models from: {ww_path or '<unresolved>'}")
    print(f"[boot] silero-vad onnx from:     {vad_path or '<unresolved>'}")


def _setup_logging(level: int = logging.WARNING) -> None:
    # Quiet by default so the printers below are the primary signal.
    # Module-level WARNING/ERROR still surface for debugging.
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _wire_printers(bus: EventBus) -> None:
    """Subscribe stdout printers to the events worth watching."""

    def on_mode(e: ModeChanged) -> None:
        print(f"[mode] {e.old.name} -> {e.new.name}")

    def on_cs(e: ConversationalStateChanged) -> None:
        print(f"[cs]   {e.old.name} -> {e.new.name}")

    def on_wake(e: WakeWordDetected) -> None:
        print(f"[wake] confidence={e.confidence:.2f}")

    def on_transcription(e: TranscriptionReady) -> None:
        print(f"[stt]  {e.text!r} ({e.duration_ms} ms of audio)")

    def on_chunk(e: LLMResponseChunk) -> None:
        print(f"[resp] {e.text!r}")

    def on_complete(e: LLMResponseComplete) -> None:
        print(f"[done] full response: {e.full_text!r}")

    bus.subscribe(ModeChanged, on_mode)
    bus.subscribe(ConversationalStateChanged, on_cs)
    bus.subscribe(WakeWordDetected, on_wake)
    bus.subscribe(TranscriptionReady, on_transcription)
    bus.subscribe(LLMResponseChunk, on_chunk)
    bus.subscribe(LLMResponseComplete, on_complete)


def _make_vad_probability_printer(threshold: float, sm: StateMachine):
    """Return a callback for SileroVAD.on_probability that prints every Nth
    inference window. Throttled by VAD_LOG_EVERY_N_WINDOWS (0 disables).

    Tags windows differently when CS is SPEAKING ("[vad-speaking]") so
    barge-in candidate evaluations are visually separable from the
    normal LISTENING-side VAD trace."""
    counter = {"n": 0}

    def cb(prob: float) -> None:
        if VAD_LOG_EVERY_N_WINDOWS <= 0:
            return
        counter["n"] += 1
        if counter["n"] % VAD_LOG_EVERY_N_WINDOWS != 0:
            return
        state = "SPEECH" if prob >= threshold else "SILENCE"
        tag = (
            "[vad-speaking]"
            if sm.conversational_state is ConversationalState.SPEAKING
            else "[vad]"
        )
        print(f"{tag} prob={prob:.2f} state={state}")

    return cb


async def _echo_producer(transcription: str) -> AsyncIterator[str]:
    """Phase 2 placeholder ResponseProducer. Yields the transcription
    verbatim so the user hears their own utterance played back. Kept
    around as a fallback / debug aid; Phase 3 uses _make_router_adapter
    instead."""
    yield transcription


def _make_router_adapter(router: IntentRouter, registry: ToolRegistry):
    """Build a ResponseProducer-shaped callable backed by IntentRouter
    and the tool registry.

    Each yielded Intent is converted to spoken text chunks via
    execute_intent (jarvis/llm/intent_router.py) — SpeakIntents become
    their text verbatim; ToolIntents dispatch through the registry and
    the success output (or error) is spoken back. The pipeline's
    existing Callable[[str], AsyncIterator[str]] contract is preserved,
    so no pipeline change is needed for Phase 4 wiring."""

    async def producer(transcription: str) -> AsyncIterator[str]:
        from jarvis.core.request_context import current_user_transcription

        print(f"[router] route({transcription!r})")
        ctx_token = current_user_transcription.set(transcription)
        try:
            async for intent in router.route(transcription):
                kind = type(intent).__name__
                print(f"[router]  -> {kind}")
                async for chunk in execute_intent(intent, registry):
                    yield chunk
        finally:
            current_user_transcription.reset(ctx_token)

    return producer


# --- main ---------------------------------------------------------------


async def main() -> int:
    _setup_logging()

    voices_dir = _voices_dir()
    cfg = load_config()
    # Hardcoded constants override config so a developer can swap devices
    # without editing the JSON file (until Phase 5 settings UI).
    input_device = INPUT_DEVICE if INPUT_DEVICE is not None else cfg.audio.input_device
    output_device = OUTPUT_DEVICE if OUTPUT_DEVICE is not None else cfg.audio.output_device
    print("Jarvis audio loopback (Phase 3)")
    print(f"  Piper voices dir: {voices_dir}")
    print(f"  Input device:     {input_device or '<auto>'}")
    print(f"  Output device:    {output_device or '<system default>'}")
    print(f"  LLM model:        {cfg.llm.model}")
    print()

    # Bus + state machine. SM defaults to (ACTIVE, IDLE); we manage the
    # initial load explicitly below for clean error reporting.
    bus = EventBus(loop=asyncio.get_running_loop())
    sm = StateMachine(bus=bus)

    # Stages. Config drives device selection so the harness reflects
    # whatever the user set in %APPDATA%/Jarvis/config.json.
    vad_threshold = VAD_THRESHOLD if VAD_THRESHOLD is not None else 0.5
    print(f"  VAD threshold:    {vad_threshold}")
    print()

    source = AudioInputSource(
        preferred_device=input_device,
        prefer_respeaker=cfg.audio.prefer_respeaker,
    )
    wake_word = OpenWakeWord(sensitivity=cfg.wake_word.sensitivity)
    vad = SileroVAD(
        speech_threshold=vad_threshold,
        # endpoint_ms and speech_start_windows use the protocol defaults
        # (700 ms, 3 windows) tuned for natural conversational speech.
        on_probability=_make_vad_probability_printer(vad_threshold, sm),
    )
    stt = FasterWhisperSTT(
        model_size=cfg.stt.model_size,
        language=cfg.stt.language,
        compute_type=cfg.stt.compute_type,
    )
    tts = PiperTTS(
        voice_name=cfg.tts.voice or PIPER_VOICE_NAME,
        voices_dir=voices_dir,
        volume=cfg.tts.volume,
        speed=SPEAK_SPEED,
        output_device=output_device,
    )

    # LLM stack. OllamaClient is a Loadable (Phase 1 Option A: load()
    # is a no-op, unload() sends keep_alive=0 to evict from VRAM).
    ollama = OllamaClient(
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        system_prompt=cfg.llm.system_prompt,
        keep_alive_seconds=cfg.llm.keep_alive_seconds,
    )
    conversation = Conversation(
        system_prompt_provider=lambda: cfg.llm.system_prompt,
        max_turns=cfg.llm.max_turns,
        inactivity_timeout_seconds=cfg.llm.inactivity_timeout_seconds,
    )
    # Tool registry: registered first so the IntentRouter sees the full
    # set at construction; the router queries it fresh per route() call,
    # so config edits while running take effect on the next turn.
    registry = ToolRegistry(cfg.tools)
    setup_local_tools(registry)
    tool_names = sorted(t.name for t in registry.list_enabled())
    print(
        f"[boot] registered {len(tool_names)} tools: "
        f"{', '.join(tool_names)}"
    )

    router = IntentRouter(
        llm=ollama,
        conversation=conversation,
        registry=registry,
        max_tool_iterations=cfg.llm.max_tool_iterations,
    )

    # Lifecycle. Order: source first (mic ready), then audio ML models,
    # then ollama (load is no-op anyway; unload runs first in reverse
    # so eviction fires before the audio modules release their RAM).
    # We do NOT bind to the bus yet -- the initial load happens via an
    # explicit call so any prerequisite failure surfaces with a clear
    # traceback rather than being swallowed by an async dispatch task.
    lm = LifecycleManager(
        [source, wake_word, vad, stt, tts, ollama], bus=bus,
    )

    _wire_printers(bus)

    # Pipeline. The router adapter replaces the Phase 2 echo producer;
    # see _make_router_adapter for the Intent->string bridge note.
    pipeline = AudioPipeline(
        source=source,
        wake_word=wake_word,
        vad=vad,
        stt=stt,
        tts=tts,
        response_producer=_make_router_adapter(router, registry),
        bus=bus,
        sm=sm,
        # Fresh-session-per-wake (see conversation.py header). Wiped
        # here at the composition root so Conversation stays event-
        # system-free and unaware of the pipeline.
        on_wake=conversation.clear,
    )

    # Boot: load all modules explicitly. Per Phase 1 lifecycle policy,
    # load_all() is fail-fast with rollback -- on any failure, modules
    # already loaded are best-effort unloaded and the original exception
    # propagates here.
    print("[boot] loading modules (may take 10-30s on first run)...")
    try:
        await lm.load_all()
    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "\nCheck the prerequisites at the top of "
            "jarvis/dev/audio_loopback.py.",
            file=sys.stderr,
        )
        return 1

    # Print which device + hostapi each stream actually resolved to. This
    # is the runtime check that WASAPI-preference logic actually won (vs
    # silently falling back to MME, which trips PortAudioError 33 on
    # post-barge-in restart).
    _print_resolved_devices(source, tts)
    _print_resolved_models(wake_word, vad)

    # Pre-warm the LLM so the user's first real turn doesn't eat the
    # 15-45 s cold-load. Failure here is non-fatal -- the daemon may
    # not be running yet; the user can still test pattern-layer hits
    # and try the LLM later.
    print("[boot] warming LLM...")
    try:
        await ollama.warm()
        print("[boot] LLM warm.")
    except Exception as e:
        print(f"[boot] LLM warmup failed: {e}")

    # Now bind the lifecycle to the bus so future Mode transitions
    # (e.g., manual sleep/wake) drive load/unload automatically.
    lm.bind(bus)

    # Start the pipeline (attach the audio consumer).
    await pipeline.start()
    print()
    print('[ready] say "hey jarvis, hello world".')
    print("        speak during the response to test barge-in.")
    print("        Ctrl+C to quit.")
    print()

    # Cross-platform Ctrl+C: install a signal handler that flips an
    # asyncio.Event. Avoids the cancellation-during-cleanup race that
    # bare KeyboardInterrupt at the asyncio.run boundary creates.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop(*_args: object) -> None:
        loop.call_soon_threadsafe(stop_event.set)

    original_sigint = signal.signal(signal.SIGINT, _request_stop)

    try:
        await stop_event.wait()
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    # Shutdown. Order matters: unbind first so SM Mode changes during
    # cleanup don't double-fire load/unload via the bus subscriber.
    print()
    print("[shutdown] detaching pipeline...")
    lm.unbind()
    await pipeline.stop()
    print("[shutdown] unloading modules...")
    await lm.unload_all()
    print("[shutdown] done.")
    return 0


def cli() -> None:
    try:
        rc = asyncio.run(main())
    except KeyboardInterrupt:
        # Defense in depth: if Ctrl+C arrives before the signal handler
        # is installed (during early setup), fall through cleanly.
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    cli()
