"""Manual verification: SLEEPING actually frees RAM.

Marked `manual` so it is skipped in the default test run. To execute:

    pytest -m manual tests/core/test_sleep_memory_manual.py -s

Operator checklist (run on the dev machine, not CI):
  1. Ensure Ollama is running with the configured model.
  2. Ensure Piper voice and Whisper model files are present.
  3. Run the command above; expect process RSS to drop by >= 150 MB.
  4. Separately confirm Ollama VRAM dropped via `ollama ps` (the
     model row should disappear or show `Until: ...` in the past).

This test boots a real audio stack, performs a SLEEPING transition via
the coordinator, gives the GC and ONNX runtime caches a few seconds to
release, then measures RSS. Numbers below are conservative thresholds;
actual drops are typically larger (~200-300 MB depending on model
sizes)."""

from __future__ import annotations

import asyncio
import gc

import pytest

_RSS_DROP_BYTES = 150_000_000  # whisper (~150 MB) is the floor we promise


@pytest.mark.manual
@pytest.mark.asyncio
async def test_sleep_frees_at_least_150mb_rss():
    psutil = pytest.importorskip("psutil")

    # Late imports so collection works in plain `pytest` runs (manual is
    # skipped, but pytest still imports the test module).
    from jarvis.audio.devices import AudioInputSource
    from jarvis.audio.pipeline import AudioPipeline
    from jarvis.audio.stt import FasterWhisperSTT
    from jarvis.audio.tts import PiperTTS
    from jarvis.audio.vad import SileroVAD
    from jarvis.audio.wake_word import OpenWakeWord
    from jarvis.core.config import load_config
    from jarvis.core.events import EventBus
    from jarvis.core.lifecycle import LifecycleManager
    from jarvis.core.mode_coordinator import ModeCoordinator
    from jarvis.core.state_machine import Mode, StateMachine
    from jarvis.llm.ollama_client import OllamaClient

    cfg = load_config()
    bus = EventBus()
    sm = StateMachine(bus=bus)

    source = AudioInputSource(
        preferred_device=cfg.audio.input_device,
        prefer_respeaker=cfg.audio.prefer_respeaker,
        bus=bus,
    )
    wake_word = OpenWakeWord(sensitivity=cfg.wake_word.sensitivity)
    vad = SileroVAD(speech_threshold=0.5)
    stt = FasterWhisperSTT(
        model_size=cfg.stt.model_size,
        language=cfg.stt.language,
        compute_type=cfg.stt.compute_type,
    )
    tts = PiperTTS(
        voice_name=cfg.tts.voice,
        volume=cfg.tts.volume,
        speed=cfg.tts.speed,
        output_device=cfg.audio.output_device,
        bus=bus,
    )
    ollama = OllamaClient(
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        system_prompt=cfg.llm.system_prompt,
        keep_alive_seconds=cfg.llm.keep_alive_seconds,
    )
    lm = LifecycleManager([source, wake_word, vad, stt, tts, ollama], bus=bus)
    async def _noop_producer(_: str):
        if False:
            yield ""  # pragma: no cover -- type-stamp as async generator

    pipeline = AudioPipeline(
        source=source, wake_word=wake_word, vad=vad, stt=stt, tts=tts,
        response_producer=_noop_producer, bus=bus, sm=sm, on_wake=lambda: None,
    )

    await lm.load_all()
    try:
        await ollama.warm()
    except Exception:
        pass  # Ollama unavailable is acceptable for this RSS check.
    await pipeline.start()

    # Let the runtime settle.
    await asyncio.sleep(2.0)
    gc.collect()
    proc = psutil.Process()
    before = proc.memory_info().rss

    coord = ModeCoordinator(
        sm=sm, lm=lm, pipeline=pipeline, tts=tts,
        sleep_confirmation=False,
    )
    await coord.request(Mode.SLEEPING)

    # ONNX runtime + faster-whisper hold caches for a few seconds after
    # the Python references go away.
    await asyncio.sleep(5.0)
    gc.collect()
    after = proc.memory_info().rss

    print(f"\n[manual] RSS before sleep: {before / 1e6:.1f} MB")
    print(f"[manual] RSS after  sleep: {after / 1e6:.1f} MB")
    print(f"[manual] dropped:          {(before - after) / 1e6:.1f} MB")
    assert before - after >= _RSS_DROP_BYTES, (
        f"expected >= {_RSS_DROP_BYTES / 1e6:.0f} MB drop, got "
        f"{(before - after) / 1e6:.1f} MB"
    )
