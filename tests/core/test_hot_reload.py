"""Tests for ConfigChanged hot-reload wiring via _handle_config_changed.

Each test verifies a specific reload path:
  - Cheap attribute updates (speed, volume, sensitivity, temperature)
  - Module reload paths (tts voice, llm model, stt model)
  - Device-change warning (no crash, warning logged)
  - ResourceMonitor reconfigure

Strategy: use real module instances for attribute-update tests (so we can
verify the actual computed properties), and AsyncMock patches for reload
paths that would touch hardware/network.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.app import _handle_config_changed
from jarvis.audio.tts import SPEECH_LENGTH_SCALE, PiperTTS
from jarvis.audio.wake_word import OpenWakeWord
from jarvis.core.config import JarvisConfig
from jarvis.core.events import ConfigChanged
from jarvis.llm.ollama_client import OllamaClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(old: JarvisConfig, new: JarvisConfig, *fields: str) -> ConfigChanged:
    return ConfigChanged(old=old, new=new, changed_fields=tuple(fields))


def _make_pipeline() -> MagicMock:
    p = MagicMock()
    p.stop = AsyncMock()
    p.start = AsyncMock()
    return p


def _make_monitor() -> MagicMock:
    m = MagicMock()
    m.reconfigure = MagicMock()
    return m


async def _dispatch(event: ConfigChanged, **kwargs) -> None:
    """Run _handle_config_changed with sensible default mocks for unused modules."""
    defaults: dict = {
        "tts": MagicMock(unload=AsyncMock(), load=AsyncMock(),
                         rewire_output=AsyncMock()),
        "stt": MagicMock(unload=AsyncMock(), load=AsyncMock()),
        "wake_word": MagicMock(),
        "ollama": MagicMock(unload=AsyncMock(), warm=AsyncMock()),
        "pipeline": _make_pipeline(),
        "resource_monitor": _make_monitor(),
        "source": MagicMock(unload=AsyncMock(), load=AsyncMock()),
        "mcp_manager": MagicMock(reload_from_config=AsyncMock()),
    }
    defaults.update(kwargs)
    await _handle_config_changed(event, **defaults)


# ---------------------------------------------------------------------------
# TTS cheap updates
# ---------------------------------------------------------------------------


async def test_tts_speed_updates_attribute_and_length_scale():
    tts = PiperTTS(voice_name="test", speed=1.0)
    old = JarvisConfig()
    new = JarvisConfig()
    new.tts.speed = 1.5

    await _dispatch(_make_event(old, new, "tts.speed"), tts=tts)

    assert tts.speed == pytest.approx(1.5)
    assert tts._effective_length_scale() == pytest.approx(SPEECH_LENGTH_SCALE / 1.5)


async def test_tts_volume_updates_attribute():
    tts = PiperTTS(voice_name="test", volume=1.0)
    old = JarvisConfig()
    new = JarvisConfig()
    new.tts.volume = 0.6

    await _dispatch(_make_event(old, new, "tts.volume"), tts=tts)

    assert tts.volume == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# TTS voice reload
# ---------------------------------------------------------------------------


async def test_tts_voice_reload_calls_unload_then_load():
    tts = MagicMock()
    tts.unload = AsyncMock()
    tts.load = AsyncMock()
    order: list[str] = []
    tts.unload.side_effect = lambda: order.append("unload")
    tts.load.side_effect = lambda: order.append("load")

    old = JarvisConfig()
    new = JarvisConfig()
    new.tts.voice = "en_US-amy-medium"

    await _dispatch(_make_event(old, new, "tts.voice"), tts=tts)

    assert order == ["unload", "load"]
    assert tts.voice_name == "en_US-amy-medium"


# ---------------------------------------------------------------------------
# Wake-word sensitivity
# ---------------------------------------------------------------------------


async def test_wake_word_sensitivity_updates_attribute_and_threshold():
    ww = OpenWakeWord(sensitivity=0.5)
    old = JarvisConfig()
    new = JarvisConfig()
    new.wake_word.sensitivity = 0.8

    await _dispatch(_make_event(old, new, "wake_word.sensitivity"), wake_word=ww)

    assert ww.sensitivity == pytest.approx(0.8)
    assert ww.threshold == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# LLM cheap updates
# ---------------------------------------------------------------------------


async def test_ollama_temperature_updates_attribute():
    ollama = OllamaClient(model="test", temperature=0.7)
    old = JarvisConfig()
    new = JarvisConfig()
    new.llm.temperature = 0.3

    await _dispatch(_make_event(old, new, "llm.temperature"), ollama=ollama)

    assert ollama.temperature == pytest.approx(0.3)


async def test_ollama_max_tokens_updates_attribute():
    ollama = OllamaClient(model="test", max_tokens=1024)
    old = JarvisConfig()
    new = JarvisConfig()
    new.llm.max_tokens = 512

    await _dispatch(_make_event(old, new, "llm.max_tokens"), ollama=ollama)

    assert ollama.max_tokens == 512


# ---------------------------------------------------------------------------
# LLM model reload
# ---------------------------------------------------------------------------


async def test_ollama_model_reload_order_and_model_update():
    ollama = OllamaClient(model="model-a")
    order: list[str] = []
    ollama.unload = AsyncMock(side_effect=lambda: order.append("unload"))
    ollama.warm = AsyncMock(side_effect=lambda: order.append("warm"))

    old = JarvisConfig()
    new = JarvisConfig()
    new.llm.model = "model-b"

    await _dispatch(_make_event(old, new, "llm.model"), ollama=ollama)

    assert order == ["unload", "warm"]
    assert ollama.model == "model-b"


async def test_ollama_model_reload_warm_failure_is_non_fatal():
    """If warm() fails after model switch, the handler should not propagate."""
    ollama = OllamaClient(model="model-a")
    ollama.unload = AsyncMock()
    ollama.warm = AsyncMock(side_effect=ConnectionRefusedError("no daemon"))

    old = JarvisConfig()
    new = JarvisConfig()
    new.llm.model = "model-b"

    # Must not raise
    await _dispatch(_make_event(old, new, "llm.model"), ollama=ollama)

    assert ollama.model == "model-b"


# ---------------------------------------------------------------------------
# STT cheap update
# ---------------------------------------------------------------------------


async def test_stt_language_updates_attribute():
    from jarvis.audio.stt import FasterWhisperSTT

    stt = FasterWhisperSTT(model_size="tiny.en", language="en")
    old = JarvisConfig()
    new = JarvisConfig()
    new.stt.language = "fr"

    await _dispatch(_make_event(old, new, "stt.language"), stt=stt)

    assert stt.language == "fr"


# ---------------------------------------------------------------------------
# STT model reload
# ---------------------------------------------------------------------------


async def test_stt_model_reload_stops_pipeline_before_unload():
    from jarvis.audio.stt import FasterWhisperSTT

    stt = FasterWhisperSTT(model_size="tiny.en")
    order: list[str] = []
    stt.unload = AsyncMock(side_effect=lambda: order.append("stt.unload"))
    stt.load = AsyncMock(side_effect=lambda: order.append("stt.load"))

    pipeline = _make_pipeline()
    pipeline.stop.side_effect = lambda: order.append("pipeline.stop")
    pipeline.start.side_effect = lambda: order.append("pipeline.start")

    old = JarvisConfig()
    new = JarvisConfig()
    new.stt.model_size = "base.en"

    await _dispatch(
        _make_event(old, new, "stt.model_size"),
        stt=stt,
        pipeline=pipeline,
    )

    assert order == ["pipeline.stop", "stt.unload", "stt.load", "pipeline.start"]
    assert stt.model_size == "base.en"


async def test_stt_compute_type_change_triggers_reload():
    from jarvis.audio.stt import FasterWhisperSTT

    stt = FasterWhisperSTT(model_size="tiny.en", compute_type="int8")
    stt.unload = AsyncMock()
    stt.load = AsyncMock()
    pipeline = _make_pipeline()

    old = JarvisConfig()
    new = JarvisConfig()
    new.stt.compute_type = "float16"

    await _dispatch(
        _make_event(old, new, "stt.compute_type"),
        stt=stt,
        pipeline=pipeline,
    )

    pipeline.stop.assert_awaited_once()
    stt.unload.assert_awaited_once()
    assert stt.compute_type == "float16"
    stt.load.assert_awaited_once()
    pipeline.start.assert_awaited_once()


# ---------------------------------------------------------------------------
# Device change: warning, no crash
# ---------------------------------------------------------------------------


async def test_input_device_change_hot_reloads_pipeline():
    """audio.input_device change stops the pipeline, reloads the source,
    and restarts the pipeline without requiring a full restart."""
    old = JarvisConfig()
    new = JarvisConfig()
    new.audio.input_device = "Blue Yeti"

    pipeline = _make_pipeline()
    source = MagicMock()
    source.unload = AsyncMock()
    source.load = AsyncMock()

    await _dispatch(
        _make_event(old, new, "audio.input_device"),
        pipeline=pipeline,
        source=source,
    )

    pipeline.stop.assert_awaited_once()
    source.unload.assert_awaited_once()
    source.load.assert_awaited_once()
    pipeline.start.assert_awaited_once()
    assert source._preferred_device == "Blue Yeti"


async def test_output_device_change_calls_rewire_output():
    """audio.output_device change calls tts.rewire_output() without
    restarting the pipeline."""
    old = JarvisConfig()
    new = JarvisConfig()
    new.audio.output_device = "HDMI Audio"

    tts = MagicMock()
    tts.rewire_output = AsyncMock()
    pipeline = _make_pipeline()

    await _dispatch(
        _make_event(old, new, "audio.output_device"),
        tts=tts,
        pipeline=pipeline,
    )

    tts.rewire_output.assert_awaited_once_with("HDMI Audio")
    pipeline.stop.assert_not_awaited()
    pipeline.start.assert_not_awaited()


# ---------------------------------------------------------------------------
# ResourceMonitor reconfigure
# ---------------------------------------------------------------------------


async def test_resource_monitor_reconfigures_on_lifecycle_change():
    from jarvis.core.resource_monitor import ResourceMonitor
    from jarvis.core.state_machine import StateMachine
    from jarvis.core.events import EventBus

    bus = EventBus(loop=asyncio.get_running_loop())
    sm = StateMachine()
    coord = MagicMock()
    coord.request = AsyncMock()

    monitor = ResourceMonitor(
        bus=bus,
        coordinator=coord,
        sm=sm,
        auto_sleep_enabled=False,
        idle_timeout_minutes=30,
    )
    monitor.start()

    old = JarvisConfig()
    new = JarvisConfig()
    new.lifecycle.auto_sleep_enabled = True
    new.lifecycle.idle_timeout_minutes = 15

    await _dispatch(
        _make_event(old, new, "lifecycle.auto_sleep_enabled", "lifecycle.idle_timeout_minutes"),
        resource_monitor=monitor,
    )

    assert monitor._auto_sleep_enabled is True
    assert monitor._idle_timeout_s == pytest.approx(15 * 60.0)
    monitor.close()


# ---------------------------------------------------------------------------
# MCP servers reload
# ---------------------------------------------------------------------------


async def test_mcp_servers_change_triggers_reload():
    """An mcp_servers edit calls mcp_manager.reload_from_config with the
    new server list so connections converge to the desired set."""
    from jarvis.core.config import MCPServerConfig

    old = JarvisConfig()
    new = JarvisConfig()
    new.mcp_servers = [
        MCPServerConfig(name="trayce", url="http://127.0.0.1:52945/mcp", enabled=True),
    ]

    mcp_manager = MagicMock()
    mcp_manager.reload_from_config = AsyncMock()

    await _dispatch(
        _make_event(old, new, "mcp_servers"),
        mcp_manager=mcp_manager,
    )

    mcp_manager.reload_from_config.assert_awaited_once()
    passed = mcp_manager.reload_from_config.await_args.args[0]
    assert [s.name for s in passed] == ["trayce"]


async def test_unrelated_field_does_not_reload_mcp():
    old = JarvisConfig()
    new = JarvisConfig()
    new.llm.temperature = 0.3

    mcp_manager = MagicMock()
    mcp_manager.reload_from_config = AsyncMock()

    await _dispatch(_make_event(old, new, "llm.temperature"), mcp_manager=mcp_manager)

    mcp_manager.reload_from_config.assert_not_awaited()


# ---------------------------------------------------------------------------
# Unrelated field change doesn't touch unrelated modules
# ---------------------------------------------------------------------------


async def test_unrelated_field_does_not_touch_tts():
    tts = MagicMock()
    tts.speed = 1.0
    tts.volume = 1.0
    tts.unload = AsyncMock()
    tts.load = AsyncMock()

    old = JarvisConfig()
    new = JarvisConfig()
    new.llm.temperature = 0.3

    await _dispatch(_make_event(old, new, "llm.temperature"), tts=tts)

    tts.unload.assert_not_awaited()
    tts.load.assert_not_awaited()
    # speed/volume should not have been mutated
    assert tts.speed == 1.0
    assert tts.volume == 1.0
