"""Composition root for the Jarvis desktop application.

Wires the audio stack (asyncio loop on a dedicated thread) with the Qt UI
(main thread). Entry point: `run() -> int`. See `jarvis/__main__.py`.

Audio thread owns: EventBus, StateMachine, LifecycleManager, AudioPipeline,
and all audio/LLM modules. Qt main thread owns: TrayIcon, OverlayOrb,
HotkeyManager, SettingsWindow.

Cross-thread bridges
--------------------
  Qt → audio:     asyncio.run_coroutine_threadsafe (mode changes, quit signal)
  audio → Qt:     QMetaObject.invokeMethod / QueuedConnection (bus subscribers)
  TTS → OverlayOrb: AmplitudeLatch (lock-free float under the GIL)

Shutdown
--------
_on_quit() (Qt thread):
  1. Signal stop_event on audio loop (call_soon_threadsafe).
  2. Join audio thread with 10 s timeout (audio thread cancels TTS,
     stops pipeline, unloads modules, closes loop).
  3. Close tray, orb, hotkeys, settings (unsubscribes + hides).
  4. qt_app.quit() → exec() returns.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from collections.abc import AsyncIterator
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from jarvis.audio.devices import AudioInputSource
from jarvis.audio.pipeline import AudioPipeline
from jarvis.audio.stt import FasterWhisperSTT
from jarvis.audio.tts import PiperTTS
from jarvis.audio.vad import SileroVAD
from jarvis.audio.wake_word import OpenWakeWord
from jarvis.core.config import (
    JarvisConfig,
    LifecycleConfig,
    MCPServerConfig,
    load_config,
)
from jarvis.core.events import (
    ConfigChanged,
    ConversationalStateChanged,
    EventBus,
    LLMResponseChunk,
    LLMResponseComplete,
    ModeChanged,
    TranscriptionReady,
    WakeWordDetected,
)
from jarvis.core.lifecycle import LifecycleManager
from jarvis.core.mode_coordinator import ModeCoordinator
from jarvis.core.request_context import current_user_transcription
from jarvis.core.resource_monitor import ResourceMonitor
from jarvis.core.state_machine import Mode, StateMachine
from jarvis.llm.conversation import Conversation
from jarvis.llm.intent_router import IntentRouter, StopIntent, ToolIntent, execute_intent
from jarvis.llm.ollama_client import OllamaClient
from jarvis.tools import MCPManager, ToolRegistry, setup_local_tools
from jarvis.ui.clipboard_history_panel import ClipboardHistoryPanel
from jarvis.ui.command_palette import CommandPalette
from jarvis.ui.dashboard_panel import DashboardPanel
from jarvis.ui.deep_research_panel import DeepResearchPanel
from jarvis.ui.help_panel import HelpPanel
from jarvis.ui.hotkeys import HotkeyManager
from jarvis.ui.log_panel import LogPanel
from jarvis.ui.notes_panel import NotesPanel
from jarvis.ui.onboarding_panel import OnboardingPanel
from jarvis.ui.overlay import OverlayOrb, make_amplitude_callback
from jarvis.ui.research_panel import ResearchPanel
from jarvis.ui.settings import SettingsWindow
from jarvis.ui.tray import TrayIcon, ensure_system_tray_available

log = logging.getLogger(__name__)

_PIPER_VOICE_NAME = "en_GB-alan-medium"
_AUDIO_BOOT_TIMEOUT = 120.0  # seconds; first-run model downloads
_AUDIO_SHUTDOWN_TIMEOUT = 10.0  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _voices_dir() -> Path:
    from jarvis.paths import default_voices_dir

    return default_voices_dir()


def _setup_logging(level_name: str = "INFO") -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_router_adapter(
    router: IntentRouter,
    conversation: Conversation,
    registry: ToolRegistry,
):
    """Wrap IntentRouter as a ResponseProducer (Callable[[str], AsyncIterator[str]])."""

    async def producer(transcription: str) -> AsyncIterator[str]:
        print(f"[router] route({transcription!r})")
        has_tool_intent = False
        tool_spoken: list[str] = []
        ctx_token = current_user_transcription.set(transcription)
        try:
            async for intent in router.route(transcription):
                print(f"[router]  -> {type(intent).__name__}")
                if isinstance(intent, StopIntent):
                    return
                if isinstance(intent, ToolIntent):
                    has_tool_intent = True
                    chunks: list[str] = []
                    async for chunk in execute_intent(intent, registry):
                        chunks.append(chunk)
                        yield chunk
                    tool_spoken.extend(chunks)
                else:
                    async for chunk in execute_intent(intent, registry):
                        yield chunk
            if (
                has_tool_intent
                and tool_spoken
                and conversation.has_unanswered_user_turn()
            ):
                conversation.add_assistant_turn("".join(tool_spoken).strip())
        finally:
            current_user_transcription.reset(ctx_token)

    return producer


def _wire_printers(bus: EventBus) -> None:
    """Subscribe stdout printers to the events worth watching in dev."""

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


# ---------------------------------------------------------------------------
# Config change helpers
# ---------------------------------------------------------------------------


def _compute_changed_fields(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Recursively diff two model_dump() dicts, returning dotted field paths."""
    changed: list[str] = []
    for key in set(old) | set(new):
        old_val = old.get(key)
        new_val = new.get(key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            changed.extend(_compute_changed_fields(old_val, new_val, path))
        elif old_val != new_val:
            changed.append(path)
    return changed


async def _handle_config_changed(
    event: ConfigChanged,
    *,
    tts: PiperTTS,
    stt: FasterWhisperSTT,
    wake_word: OpenWakeWord,
    ollama: OllamaClient,
    pipeline: AudioPipeline,
    resource_monitor: ResourceMonitor,
    source: AudioInputSource,
    mcp_manager: MCPManager,
    registry: ToolRegistry,
) -> None:
    """Apply hot-reloadable config changes to running audio modules.

    Cheap updates (attribute writes) take effect immediately.
    Reload paths (voice, LLM model, STT model) briefly interrupt processing.
    Device changes hot-swap the stream (output: ~50 ms gap; input: pipeline
    briefly stops and restarts).
    """
    fields = set(event.changed_fields)
    new = event.new

    # TTS: cheap updates
    if "tts.speed" in fields:
        tts.speed = new.tts.speed
        log.info("hot-reload: tts.speed → %.2f", new.tts.speed)
    if "tts.volume" in fields:
        tts.volume = new.tts.volume
        log.info("hot-reload: tts.volume → %.2f", new.tts.volume)

    # TTS: voice reload (unload → update → reload)
    if "tts.voice" in fields:
        log.info("hot-reload: tts.voice → %r (reloading)", new.tts.voice)
        await tts.unload()
        tts.voice_name = new.tts.voice or _PIPER_VOICE_NAME
        await tts.load()

    # Wake word: sensitivity is a property computed from self.sensitivity
    if "wake_word.sensitivity" in fields:
        wake_word.sensitivity = new.wake_word.sensitivity
        log.info("hot-reload: wake_word.sensitivity → %.2f", new.wake_word.sensitivity)

    # LLM: cheap updates
    if "llm.temperature" in fields:
        ollama.temperature = new.llm.temperature
        log.info("hot-reload: llm.temperature → %.2f", new.llm.temperature)
    if "llm.max_tokens" in fields:
        ollama.max_tokens = new.llm.max_tokens
    if "llm.keep_alive_seconds" in fields:
        ollama.keep_alive_seconds = new.llm.keep_alive_seconds

    # LLM: model reload (evict → update → warm)
    if "llm.model" in fields:
        log.info("hot-reload: llm.model → %r (reloading)", new.llm.model)
        await ollama.unload()
        ollama.model = new.llm.model
        try:
            await ollama.warm()
        except Exception:
            log.warning("hot-reload: LLM warmup after model change failed", exc_info=True)

    # STT: language is cheap
    if "stt.language" in fields:
        stt.language = new.stt.language
        log.info("hot-reload: stt.language → %r", new.stt.language)

    # STT: model/compute reload (stop pipeline → unload → update → load → restart)
    if "stt.model_size" in fields or "stt.compute_type" in fields:
        log.info("hot-reload: stt model change (briefly stopping pipeline)")
        await pipeline.stop()
        await stt.unload()
        stt.model_size = new.stt.model_size
        stt.compute_type = new.stt.compute_type
        await stt.load()
        await pipeline.start()

    # Audio devices: hot-swap output stream; briefly pause pipeline for input.
    if "audio.output_device" in fields:
        log.info("hot-reload: output_device → %r", new.audio.output_device)
        try:
            await tts.rewire_output(new.audio.output_device)
        except Exception:
            log.warning("hot-reload: output device rewire failed", exc_info=True)

    if "audio.input_device" in fields:
        log.info(
            "hot-reload: input_device → %r (briefly stopping pipeline)",
            new.audio.input_device,
        )
        await pipeline.stop()
        await source.unload()
        source._preferred_device = new.audio.input_device
        try:
            await source.load()
        except Exception:
            log.warning("hot-reload: input device open failed", exc_info=True)
        await pipeline.start()

    # ResourceMonitor: cheap update
    if "lifecycle.auto_sleep_enabled" in fields or "lifecycle.idle_timeout_minutes" in fields:
        resource_monitor.reconfigure(
            auto_sleep_enabled=new.lifecycle.auto_sleep_enabled,
            idle_timeout_minutes=new.lifecycle.idle_timeout_minutes,
        )
        log.info(
            "hot-reload: auto_sleep=%s, idle_timeout=%d min",
            new.lifecycle.auto_sleep_enabled, new.lifecycle.idle_timeout_minutes,
        )

    # MCP servers: the Settings → Tools tab edits the whole list, so any
    # change surfaces as the "mcp_servers" path. reload_from_config diffs
    # live connections against the new desired set (add/remove/reconnect).
    if any(f == "mcp_servers" or f.startswith("mcp_servers") for f in fields):
        log.info("hot-reload: mcp_servers changed; reconciling connections")
        try:
            await mcp_manager.reload_from_config(list(new.mcp_servers))
        except Exception:
            log.warning("hot-reload: MCP reload failed", exc_info=True)

    if any(f == "workspace.apps" or f.startswith("workspace") for f in fields):
        from jarvis.tools.local.launch_workspace import LaunchWorkspaceTool

        log.info("hot-reload: workspace.apps changed; refreshing launch_workspace")
        registry.unregister("launch_workspace")
        registry.register(
            LaunchWorkspaceTool(workspace_apps=list(new.workspace.apps))
        )


# ---------------------------------------------------------------------------
# Audio-thread coroutine (the entire audio lifecycle in one coroutine)
# ---------------------------------------------------------------------------


async def _audio_main(
    lm: LifecycleManager,
    pipeline: AudioPipeline,
    ollama: OllamaClient,
    tts: PiperTTS,
    stt: FasterWhisperSTT,
    wake_word: OpenWakeWord,
    bus: EventBus,
    sm: StateMachine,
    coordinator: ModeCoordinator,
    lifecycle_cfg: LifecycleConfig,
    stop_event: asyncio.Event,
    boot_error_holder: list[str | None],
    boot_done: threading.Event,
    source: AudioInputSource,
    mcp_manager: MCPManager,
    mcp_servers: list[MCPServerConfig],
    registry: ToolRegistry,
) -> None:
    """Runs on the audio asyncio loop's thread.

    Boot sequence: load_all → warm LLM → start pipeline → start monitor →
    connect MCP servers. Sets boot_done once ready (or on failure) so the
    main thread can proceed. Then waits for stop_event (set by on_quit on
    the Qt thread). Cleanup: shutdown MCP, close monitor, stop pipeline,
    unload modules.
    """
    log.info("loading audio modules (may take 10-30 s on first run)...")
    try:
        await lm.load_all()
    except Exception as exc:
        log.error("module load failed: %s", exc, exc_info=True)
        boot_error_holder[0] = f"load_failure:{type(exc).__name__}: {exc}"
        boot_done.set()
        return

    log.info("warming LLM...")
    try:
        await ollama.warm()
        log.info("LLM warm.")
    except Exception as exc:
        log.warning("LLM warmup failed (Ollama may not be running): %s", exc)
        boot_error_holder[0] = f"ollama_warning:{exc}"
        # Non-fatal: Jarvis starts; LLM calls will fail until Ollama is started.

    # NOTE: lm.bind(bus) is intentionally NOT called. ModeCoordinator owns
    # all Mode transitions and drives lm.transition_to_mode explicitly so
    # the speak-confirmation phase can precede unload and so wake can
    # await load_all before restarting the pipeline.
    await pipeline.start()

    resource_monitor = ResourceMonitor(
        bus=bus,
        coordinator=coordinator,
        sm=sm,
        auto_sleep_enabled=lifecycle_cfg.auto_sleep_enabled,
        idle_timeout_minutes=lifecycle_cfg.idle_timeout_minutes,
    )
    resource_monitor.start()

    # Wire config hot-reload. The lambda returns a coroutine which the bus awaits.
    bus.subscribe(
        ConfigChanged,
        lambda e: _handle_config_changed(
            e,
            tts=tts,
            stt=stt,
            wake_word=wake_word,
            ollama=ollama,
            pipeline=pipeline,
            resource_monitor=resource_monitor,
            source=source,
            mcp_manager=mcp_manager,
            registry=registry,
        ),
    )

    # MCP connections are best-effort: a missing/offline server logs a
    # warning inside add_server and never blocks boot. Done after
    # boot_done would also work, but connecting here means tools are
    # registered before the user can speak.
    for server_cfg in mcp_servers:
        try:
            await mcp_manager.add_server(server_cfg)
        except Exception:
            log.warning("MCP add_server(%r) raised; continuing", server_cfg.name, exc_info=True)

    boot_done.set()

    log.info('ready — say "hey jarvis".')
    await stop_event.wait()

    # Shutdown: MCP first (network teardown), then monitor before pipeline
    # so the monitor's loop task does not observe a half-torn-down pipeline.
    try:
        await mcp_manager.shutdown()
    except Exception:
        log.warning("MCP shutdown raised; continuing", exc_info=True)
    resource_monitor.close()
    log.info("shutdown: stopping pipeline...")
    # No lm.unbind(): coordinator owns transitions and is not bus-bound.
    await pipeline.stop()
    log.info("shutdown: unloading modules...")
    await lm.unload_all()
    log.info("shutdown: audio stack done.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run() -> int:
    """Compose and launch Jarvis as a desktop application.

    Returns the Qt exit code (0 on clean quit, non-zero on error).
    """
    cfg = load_config()
    _setup_logging(cfg.general.log_level)
    from jarvis.paths import bundled_asset_report, is_frozen

    if is_frozen():
        log.info("bundled assets: %s", bundled_asset_report())
    voices_dir = _voices_dir()

    # ------------------------------------------------------------------
    # 1. Audio loop (created here; started in dedicated thread below)
    # ------------------------------------------------------------------
    audio_loop = asyncio.new_event_loop()

    # ------------------------------------------------------------------
    # 2-4. Core layer
    # ------------------------------------------------------------------
    bus = EventBus(loop=audio_loop)
    sm = StateMachine(bus=bus)

    # ------------------------------------------------------------------
    # 5. Audio modules
    # ------------------------------------------------------------------
    amplitude_latch, amplitude_callback = make_amplitude_callback()

    source = AudioInputSource(
        preferred_device=cfg.audio.input_device,
        prefer_respeaker=cfg.audio.prefer_respeaker,
        bus=bus,
    )
    wake_word = OpenWakeWord(sensitivity=cfg.wake_word.sensitivity)
    from jarvis.paths import (
        default_silero_onnx_path,
        default_whisper_download_root,
    )

    vad = SileroVAD(
        speech_threshold=0.5,
        model_path=default_silero_onnx_path(),
    )
    stt = FasterWhisperSTT(
        model_size=cfg.stt.model_size,
        language=cfg.stt.language,
        compute_type=cfg.stt.compute_type,
        download_root=default_whisper_download_root(),
    )
    tts = PiperTTS(
        voice_name=cfg.tts.voice or _PIPER_VOICE_NAME,
        voices_dir=voices_dir,
        volume=cfg.tts.volume,
        speed=cfg.tts.speed,
        output_device=cfg.audio.output_device,
        on_amplitude=amplitude_callback,
        bus=bus,
    )

    # ------------------------------------------------------------------
    # 6. LLM stack + tool registry
    # ------------------------------------------------------------------
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
    registry = ToolRegistry(cfg.tools)
    setup_local_tools(registry, config=cfg, ollama_client=ollama)
    # Research tools are wired later (step 12) after the Qt app and
    # ResearchPanel are created, because their callbacks reference the panel.
    mcp_manager = MCPManager(registry)
    router = IntentRouter(llm=ollama, conversation=conversation, registry=registry)

    lm = LifecycleManager([source, wake_word, vad, stt, tts, ollama], bus=bus)

    # Re-asserted in BUILD.md Phase 6 design note: AudioInputSource IS in
    # the lifecycle list. On SLEEPING, its unload closes the input stream,
    # releasing the mic device. On wake, load reopens it; the pipeline's
    # start() re-attaches its on_frame callback. SPEC table did not list
    # source explicitly; we close it deliberately to free the device.

    pipeline = AudioPipeline(
        source=source,
        wake_word=wake_word,
        vad=vad,
        stt=stt,
        tts=tts,
        response_producer=_make_router_adapter(router, conversation, registry),
        bus=bus,
        sm=sm,
        on_wake=lambda: conversation.maybe_clear(cfg.llm.conversation_continuity_seconds),
        log_wake_during_speaking=cfg.debug.log_wake_during_speaking,
    )

    # ------------------------------------------------------------------
    # 7. Mode coordinator (Phase 6 Task 1)
    # ------------------------------------------------------------------
    # Owns the speak-then-unload sequence on Sleep, the load-then-restart
    # sequence on Wake, and the Sleep/Wake/Mute race rules. Replaces
    # lm.bind(bus) as the driver for Mode transitions.
    mode_coord = ModeCoordinator(
        sm=sm,
        lm=lm,
        pipeline=pipeline,
        tts=tts,
        sleep_confirmation=cfg.general.sleep_confirmation,
    )

    # ------------------------------------------------------------------
    # 9. Qt application (created before audio boot so dialogs work)
    # ------------------------------------------------------------------
    qt_app = QApplication.instance() or QApplication(sys.argv)

    # ------------------------------------------------------------------
    # 10. System tray availability check
    # ------------------------------------------------------------------
    if not ensure_system_tray_available():
        audio_loop.close()
        return 1

    # ------------------------------------------------------------------
    # 8. Boot audio stack in dedicated thread; wait for ready
    # ------------------------------------------------------------------
    stop_event = asyncio.Event()
    boot_error_holder: list[str | None] = [None]
    boot_done = threading.Event()

    def _run_audio_loop() -> None:
        asyncio.set_event_loop(audio_loop)
        audio_loop.run_until_complete(
            _audio_main(
                lm, pipeline, ollama, tts, stt, wake_word,
                bus, sm, mode_coord, cfg.lifecycle,
                stop_event, boot_error_holder, boot_done,
                source=source,
                mcp_manager=mcp_manager,
                mcp_servers=cfg.mcp_servers,
                registry=registry,
            )
        )
        audio_loop.close()

    audio_thread = threading.Thread(
        target=_run_audio_loop,
        daemon=True,
        name="jarvis-audio",
    )
    audio_thread.start()

    log.info("waiting for audio stack boot (up to %.0f s)...", _AUDIO_BOOT_TIMEOUT)
    if not boot_done.wait(timeout=_AUDIO_BOOT_TIMEOUT):
        QMessageBox.critical(
            None,  # type: ignore[arg-type]
            "Jarvis — Startup Error",
            "Jarvis timed out loading audio modules.\n\n"
            "Check that all prerequisites are installed and try again.",
        )
        audio_loop.call_soon_threadsafe(stop_event.set)
        audio_thread.join(timeout=5.0)
        return 1

    error = boot_error_holder[0]
    if error:
        if error.startswith("ollama_warning:"):
            detail = error[len("ollama_warning:"):]
            QMessageBox.warning(
                None,  # type: ignore[arg-type]
                "Jarvis — Ollama Not Running",
                "Ollama does not appear to be running.\n\n"
                "Jarvis will start, but voice commands that require the LLM will "
                "fail until Ollama is started.\n\n"
                f"Detail: {detail}",
            )
        else:
            detail = error[len("load_failure:"):] if error.startswith("load_failure:") else error
            QMessageBox.critical(
                None,  # type: ignore[arg-type]
                "Jarvis — Startup Error",
                f"A required module failed to load:\n\n{detail}\n\n"
                "Check that all prerequisites are installed (models downloaded, "
                "audio devices connected) and try again.",
            )
            audio_loop.call_soon_threadsafe(stop_event.set)
            audio_thread.join(timeout=5.0)
            return 1

    # ------------------------------------------------------------------
    # 11. Verbose console (dev-time visibility; gate behind config later)
    # ------------------------------------------------------------------
    _wire_printers(bus)

    # ------------------------------------------------------------------
    # 12-14. Qt UI components
    # ------------------------------------------------------------------
    _settings_ref: list[SettingsWindow | None] = [None]
    _cfg_snapshot: list[dict] = [cfg.model_dump(mode="json")]

    _research_panel_ref: list = [None]
    _deep_research_panel_ref: list = [None]
    _notes_panel_ref: list = [None]
    _dashboard_panel_ref: list = [None]
    _help_panel_ref: list = [None]
    _clipboard_panel_ref: list = [None]
    _log_panel_ref: list = [None]
    _command_palette_ref: list = [None]
    _onboarding_ref: list = [None]

    def _on_config_change() -> None:
        new_dict = cfg.model_dump(mode="json")
        old_dict = _cfg_snapshot[0]
        changed = _compute_changed_fields(old_dict, new_dict)
        if not changed:
            return
        old_cfg = JarvisConfig.model_validate(old_dict)
        new_cfg = JarvisConfig.model_validate(new_dict)
        _cfg_snapshot[0] = new_dict
        panel = _research_panel_ref[0]
        if panel is not None and "ui.research_panel_width" in changed:
            panel.set_panel_width(new_cfg.ui.research_panel_width)
        bus.publish(ConfigChanged(old=old_cfg, new=new_cfg, changed_fields=tuple(changed)))

    def _on_test_voice(phrase: str) -> None:
        """Called from the Qt thread; dispatches tts.speak to the audio loop."""
        try:
            asyncio.run_coroutine_threadsafe(tts.speak(phrase), audio_loop)
        except Exception:
            log.exception("test-voice dispatch failed")

    def _open_settings() -> None:
        """Create or raise the settings window. Must run on Qt main thread."""
        if _settings_ref[0] is None:
            _settings_ref[0] = SettingsWindow(
                config=cfg,
                on_change=_on_config_change,
                voices_dir=voices_dir,
                on_test_voice=_on_test_voice,
            )
        win = _settings_ref[0]
        win.show()
        win.raise_()
        win.activateWindow()

    def _open_settings_any_thread() -> None:
        """Thread-safe entry point for hotkey manager (pynput thread)."""
        QTimer.singleShot(0, _open_settings)

    _quit_called = [False]

    def _on_quit() -> None:
        if _quit_called[0]:
            return
        _quit_called[0] = True

        # Signal audio loop → triggers cleanup coroutine in audio thread
        audio_loop.call_soon_threadsafe(stop_event.set)

        # Block Qt main thread until audio stack drains (tray hidden below
        # so the user sees Jarvis disappear immediately; the wait is invisible)
        try:
            tray.hide()
        except Exception:
            log.debug("tray.hide() failed during quit", exc_info=True)

        if audio_thread.is_alive():
            audio_thread.join(timeout=_AUDIO_SHUTDOWN_TIMEOUT)
            if audio_thread.is_alive():
                log.warning("audio thread did not stop within %.0f s", _AUDIO_SHUTDOWN_TIMEOUT)

        # Python-level cleanup: unsubscribes, timer stops, etc.
        tray.close()
        orb.close()
        hotkeys.close()
        research_panel.close_panel()
        if _deep_research_panel_ref[0] is not None:
            _deep_research_panel_ref[0].close_panel()
        if _notes_panel_ref[0] is not None:
            _notes_panel_ref[0].close_panel()
        if _dashboard_panel_ref[0] is not None:
            _dashboard_panel_ref[0].close_panel()
        if _help_panel_ref[0] is not None:
            _help_panel_ref[0].close_panel()
        if _clipboard_panel_ref[0] is not None:
            _clipboard_panel_ref[0].close_panel()
        if _log_panel_ref[0] is not None:
            _log_panel_ref[0].close_panel()
        if _command_palette_ref[0] is not None:
            _command_palette_ref[0].close_palette()
        if _onboarding_ref[0] is not None:
            _onboarding_ref[0].close_panel()
        if _settings_ref[0] is not None:
            _settings_ref[0].close()

        qt_app.quit()

    # Mode requests from tray/hotkeys route through the coordinator.
    # The returned coroutine is awaited on the audio loop by the
    # caller's run_coroutine_threadsafe wrapper.
    def _request_mode(target: Mode):
        return mode_coord.request(target)

    def _tray_open(ref_list):
        panel = ref_list[0]
        if panel is not None:
            panel.open_panel()

    def _tray_open_palette():
        palette = _command_palette_ref[0]
        if palette is not None:
            palette.open_palette()

    tray = TrayIcon(
        sm=sm,
        bus=bus,
        audio_loop=audio_loop,
        hotkeys=cfg.hotkeys,
        on_open_settings=_open_settings,
        on_quit=_on_quit,
        on_mode_request=_request_mode,
        on_open_dashboard=lambda: _tray_open(_dashboard_panel_ref),
        on_open_notes=lambda: _tray_open(_notes_panel_ref),
        on_open_help=lambda: _tray_open(_help_panel_ref),
        on_open_clipboard_history=lambda: _tray_open(_clipboard_panel_ref),
        on_open_logs=lambda: _tray_open(_log_panel_ref),
        on_open_command_palette=_tray_open_palette,
        on_open_tutorial=lambda: _tray_open(_onboarding_ref),
    )
    tray.show()

    orb = OverlayOrb(sm=sm, bus=bus, amplitude_latch=amplitude_latch)

    # Research panel + tool registration. Panel lives on the Qt thread;
    # the tools emit cross-thread Signals to drive it from the audio loop.
    def _on_research_panel_width(width: int) -> None:
        if cfg.ui.research_panel_width != width:
            cfg.ui.research_panel_width = width
            _on_config_change()

    research_panel = ResearchPanel(
        panel_width=cfg.ui.research_panel_width,
        on_width_changed=_on_research_panel_width,
        ollama_model=cfg.llm.model,
    )
    _research_panel_ref[0] = research_panel

    from jarvis.tools.local.research import (
        CloseResearchTool,
        CopyResearchTool,
        ReadMoreTool,
        ResearchTool,
    )
    registry.register(ResearchTool(
        on_start=research_panel.show_for_query,
        on_speak=tts.speak,
    ))
    registry.register(CloseResearchTool(
        close_callback=research_panel.close_panel,
    ))
    registry.register(ReadMoreTool(
        get_next=research_panel.get_next_sentences,
    ))
    registry.register(CopyResearchTool(
        copy_callback=research_panel.copy_summary,
    ))

    def _deep_research_config_provider():
        from jarvis.llm.ollama_client import DEFAULT_ENDPOINT
        from jarvis.tools.local.deep_research_runner import build_deep_research_config

        return build_deep_research_config(
            research=cfg.research,
            main_llm_model=cfg.llm.model,
            ollama_endpoint=DEFAULT_ENDPOINT,
        )

    def _set_deep_research_ultra(enabled: bool) -> str:
        from jarvis.core.config import save_config

        cfg.research.ultra_enabled = enabled
        save_config(cfg)
        if enabled:
            return (
                "Deep research Ultra is on, sir. "
                "Set JARVIS_BRAVE_API_KEY and JARVIS_GROQ_API_KEY for the full stack."
            )
        return "Deep research Ultra is off, sir. Using standard local deep research."

    deep_research_panel = DeepResearchPanel(
        config_provider=_deep_research_config_provider,
    )
    _deep_research_panel_ref[0] = deep_research_panel

    from jarvis.tools.local.deep_research_tools import (
        CloseDeepResearchTool,
        DeepResearchTool,
        DeleteAllDeepResearchTool,
        DeleteDeepResearchTool,
        PauseDeepResearchTool,
        ResumeDeepResearchTool,
    )
    from jarvis.tools.local.deep_research_ultra_tools import (
        DisableDeepResearchUltraTool,
        EnableDeepResearchUltraTool,
    )

    registry.register(DeepResearchTool(
        on_start=deep_research_panel.show_for_query,
        on_speak=tts.speak,
        ultra_enabled=lambda: cfg.research.ultra_enabled,
    ))
    registry.register(PauseDeepResearchTool(
        on_pause=deep_research_panel.pause_active,
    ))
    registry.register(ResumeDeepResearchTool(
        on_resume_latest=deep_research_panel.resume_latest_paused,
    ))
    registry.register(CloseDeepResearchTool(
        close_callback=deep_research_panel.close_panel,
    ))
    registry.register(DeleteDeepResearchTool(
        delete_by_query=deep_research_panel.delete_by_query,
        delete_active=deep_research_panel.delete_active,
    ))
    registry.register(DeleteAllDeepResearchTool(
        delete_all=deep_research_panel.delete_all,
    ))
    registry.register(EnableDeepResearchUltraTool(
        set_ultra=_set_deep_research_ultra,
    ))
    registry.register(DisableDeepResearchUltraTool(
        set_ultra=_set_deep_research_ultra,
    ))

    # --- Notes panel + voice tools ---------------------------------------
    notes_panel = NotesPanel()
    _notes_panel_ref[0] = notes_panel

    from jarvis.tools.local.notes_tools import (
        AppendToNoteTool,
        CloseNotesTool,
        DeleteNoteTool,
        OpenNotesTool,
        ReadNoteTool,
        TakeNoteTool,
    )

    def _take_note(title: str, content: str) -> str:
        return notes_panel.create_and_show(title, content)

    registry.register(TakeNoteTool(on_create=_take_note))
    registry.register(AppendToNoteTool(
        on_append_active=notes_panel.append_to_active,
        on_append_by_title=notes_panel.append_by_title,
    ))
    registry.register(ReadNoteTool(
        on_read_active=notes_panel.read_active,
        on_read_by_title=notes_panel.read_by_title,
    ))
    registry.register(OpenNotesTool(on_open=notes_panel.open_panel))
    registry.register(CloseNotesTool(on_close=notes_panel.close_panel))
    registry.register(DeleteNoteTool(
        on_delete_active=notes_panel.delete_active,
        on_delete_by_title=notes_panel.delete_by_title,
    ))

    # --- Dashboard panel + voice tools -----------------------------------
    from jarvis.tools.local.deep_research_store import list_sessions as _list_dr
    from jarvis.tools.local.notes_store import list_notes as _list_notes

    def _dr_counts() -> tuple[int, int]:
        sessions = _list_dr()
        paused = sum(1 for s in sessions if s.status == "paused")
        return (len(sessions), paused)

    def _notes_count() -> int:
        return len(_list_notes())

    dashboard_panel = DashboardPanel(
        sm=sm,
        amplitude_latch=amplitude_latch,
        config_provider=lambda: cfg,
        deep_research_count_provider=_dr_counts,
        notes_count_provider=_notes_count,
    )
    _dashboard_panel_ref[0] = dashboard_panel

    from jarvis.tools.local.dashboard_tools import (
        CloseDashboardTool,
        ShowDashboardTool,
    )

    registry.register(ShowDashboardTool(on_open=dashboard_panel.open_panel))
    registry.register(CloseDashboardTool(on_close=dashboard_panel.close_panel))

    # --- Help panel + voice tools ----------------------------------------
    help_panel = HelpPanel()
    _help_panel_ref[0] = help_panel

    from jarvis.tools.local.help_tools import OpenHelpTool

    registry.register(OpenHelpTool(on_open=help_panel.open_panel))

    # --- Clipboard history panel + voice tools ---------------------------
    clipboard_panel = ClipboardHistoryPanel()
    _clipboard_panel_ref[0] = clipboard_panel

    from jarvis.tools.local.clipboard_history_tools import (
        ClearClipboardHistoryTool,
        CloseClipboardHistoryTool,
        PasteClipboardItemTool,
        ShowClipboardHistoryTool,
    )

    registry.register(ShowClipboardHistoryTool(
        on_open=clipboard_panel.open_panel,
    ))
    registry.register(CloseClipboardHistoryTool(
        on_close=clipboard_panel.close_panel,
    ))
    registry.register(PasteClipboardItemTool(
        on_paste=clipboard_panel.paste_index,
    ))
    registry.register(ClearClipboardHistoryTool(
        on_clear=lambda: clipboard_panel.clear_all(keep_pinned=True),
    ))

    # --- Live log viewer panel + voice tools -----------------------------
    log_panel = LogPanel()
    _log_panel_ref[0] = log_panel

    from jarvis.tools.local.log_tools import CloseLogsTool, ShowLogsTool

    registry.register(ShowLogsTool(on_open=log_panel.open_panel))
    registry.register(CloseLogsTool(on_close=log_panel.close_panel))

    # --- Command palette -------------------------------------------------
    # Submission routes through the audio loop: we wrap the producer that
    # the AudioPipeline normally drives so palette entries fire the exact
    # same intent-router + tool pipeline as a real STT result, just without
    # the wake-word / VAD gating.
    palette_producer = _make_router_adapter(router, conversation, registry)

    async def _consume_palette_text(text: str) -> None:
        try:
            async for _chunk in palette_producer(text):
                pass
        except Exception:
            log.exception("command palette text execution failed")

    def _submit_palette_text(text: str) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                _consume_palette_text(text), audio_loop
            )
        except Exception:
            log.exception("could not schedule palette text onto audio loop")

    command_palette = CommandPalette(submit_text=_submit_palette_text)
    _command_palette_ref[0] = command_palette

    # --- Onboarding panel (auto-shown on first run) ----------------------
    def _on_onboarding_finished() -> None:
        if not cfg.general.first_run_completed:
            cfg.general.first_run_completed = True
            try:
                _on_config_change()
            except Exception:
                log.exception("config persist failed after onboarding finish")

    onboarding_panel = OnboardingPanel(
        bus=bus,
        amplitude_latch=amplitude_latch,
        on_finished=_on_onboarding_finished,
        on_open_help=help_panel.open_panel,
        on_open_command_palette=command_palette.open_palette,
    )
    _onboarding_ref[0] = onboarding_panel

    if not cfg.general.first_run_completed:
        # Defer to next Qt tick so the rest of the UI exists first.
        QTimer.singleShot(800, onboarding_panel.open_panel)

    hotkeys = HotkeyManager(
        sm=sm,
        bus=bus,
        audio_loop=audio_loop,
        hotkeys=cfg.hotkeys,
        on_mode_request=_request_mode,
        on_open_settings=_open_settings_any_thread,
        on_open_command_palette=lambda: QTimer.singleShot(
            0, command_palette.open_palette
        ),
    )
    hotkeys.register_all()

    # ------------------------------------------------------------------
    # 14. Qt event loop
    # ------------------------------------------------------------------
    return qt_app.exec()
