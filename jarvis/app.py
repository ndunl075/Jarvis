"""Composition root for the Jarvis desktop application.

Wires the audio stack (asyncio loop on a dedicated thread) with the Qt UI
(main thread). Entry point: `run() -> int`, a thin wrapper that builds a
`JarvisApp` and calls `start()`. See `jarvis/__main__.py`.

`JarvisApp` owns both halves and the handlers that bridge them: its
`_build_audio_stack()` constructs what the audio thread owns and
`_build_ui()` what the Qt main thread owns, in the order `start()` calls
them. That order is the subject of most of what follows.

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
JarvisApp._on_quit() (Qt thread):
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
from jarvis.ui.overlay import AmplitudeLatch, OverlayOrb, make_amplitude_callback
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
    # basicConfig() is a no-op once the root logger already has a handler,
    # and the frozen launcher (jarvis/__main__.py) installs a file handler
    # at INFO before run() is reached. Set the level explicitly so the
    # Settings "Log level" is authoritative in packaged builds too --
    # otherwise the DEBUG trace wired by _wire_event_logging() would be
    # unreachable in exactly the build where there is no console to fall
    # back on.
    logging.getLogger().setLevel(level)


def _make_router_adapter(
    router: IntentRouter,
    conversation: Conversation,
    registry: ToolRegistry,
):
    """Wrap IntentRouter as a ResponseProducer (Callable[[str], AsyncIterator[str]]).

    ToolIntents reaching this adapter come from the PATTERN layer
    ("open spotify", "volume up") — snap commands that bypass the LLM
    and whose tool output is the answer, spoken verbatim with no
    inference in the way.

    LLM-chosen tools do NOT arrive here when the tool-result feedback
    loop is on (cfg.llm.max_tool_iterations > 1): the router executes
    those itself so it can feed the results back to the model, and
    yields only the SpeakIntents of the model's final answer. With
    max_tool_iterations == 1 the router falls back to yielding
    ToolIntents and this branch handles them exactly as it always did.
    The contextvar is set before iteration begins, so tools executed
    inside route() still see the transcription that prompted them.
    """

    async def producer(transcription: str) -> AsyncIterator[str]:
        log.debug("[router] route(%r)", transcription)
        has_tool_intent = False
        tool_spoken: list[str] = []
        ctx_token = current_user_transcription.set(transcription)
        try:
            async for intent in router.route(transcription):
                log.debug("[router]  -> %s", type(intent).__name__)
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


def _wire_event_logging(bus: EventBus) -> None:
    """Subscribe DEBUG trace logging to the events worth watching.

    Every line here is per-interaction trace, so every line is DEBUG: at
    the default INFO level none of it is emitted at all, and the lazy %r
    arguments are not even formatted.

    Three of these events (TranscriptionReady, LLMResponseChunk,
    LLMResponseComplete) carry the user's own words. Keeping them is a
    deliberate choice -- a visible transcript is the only way to debug
    "it misheard me" -- but it is a choice the user makes, by raising
    Settings -> Log level to DEBUG. Note the consequence, because it is
    the reason this is a choice and not a default: at DEBUG the
    transcript lands in %APPDATA%/Jarvis/logs/jarvis.log and persists on
    disk until that file rotates away. That is the trade for being able
    to see what Jarvis heard. At any other level the text never leaves
    the process.
    """

    def on_mode(e: ModeChanged) -> None:
        log.debug("[mode] %s -> %s", e.old.name, e.new.name)

    def on_cs(e: ConversationalStateChanged) -> None:
        log.debug("[cs] %s -> %s", e.old.name, e.new.name)

    def on_wake(e: WakeWordDetected) -> None:
        log.debug("[wake] confidence=%.2f", e.confidence)

    def on_transcription(e: TranscriptionReady) -> None:
        log.debug("[stt] %r (%d ms of audio)", e.text, e.duration_ms)

    def on_chunk(e: LLMResponseChunk) -> None:
        log.debug("[resp] %r", e.text)

    def on_complete(e: LLMResponseComplete) -> None:
        log.debug("[done] full response: %r", e.full_text)

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
# Application object
# ---------------------------------------------------------------------------


class JarvisApp:
    """The composition root as an object: build it, start it, tear it down.

    `start()` runs the build phases in a fixed order — the order the old
    monolithic `run()` used, because that order is load-bearing (see the
    module docstring). The phase methods name the steps; they do not
    reorder them.

    Thread ownership is unchanged. `_build_audio_stack()` constructs
    everything the audio thread owns and `_build_ui()` everything the Qt
    main thread owns; `self` is merely the object both halves hang off, so
    every actual crossing still goes through the bridges named above
    (`run_coroutine_threadsafe`, `call_soon_threadsafe`, QueuedConnection,
    `AmplitudeLatch`).

    The point of the class is the handlers. As closures inside `run()`,
    `_on_quit`, `_on_config_change`, `_request_mode` and friends could not
    be reached without booting the audio stack, the Qt UI, the LLM client
    and the tool registry. As methods they are reachable from a test that
    sets two or three attributes on a bare `JarvisApp()`.
    """

    # Set by _load_config.
    cfg: JarvisConfig
    voices_dir: Path

    # The audio thread and the loop it runs (set by _build_audio_stack /
    # _boot_audio_stack). The Qt thread touches these only through
    # call_soon_threadsafe / run_coroutine_threadsafe.
    audio_loop: asyncio.AbstractEventLoop
    audio_thread: threading.Thread
    stop_event: asyncio.Event
    boot_error_holder: list[str | None]
    boot_done: threading.Event

    # Owned by the audio thread (set by _build_audio_stack).
    bus: EventBus
    sm: StateMachine
    amplitude_latch: AmplitudeLatch
    source: AudioInputSource
    wake_word: OpenWakeWord
    vad: SileroVAD
    stt: FasterWhisperSTT
    tts: PiperTTS
    ollama: OllamaClient
    conversation: Conversation
    registry: ToolRegistry
    mcp_manager: MCPManager
    router: IntentRouter
    lm: LifecycleManager
    pipeline: AudioPipeline
    mode_coord: ModeCoordinator

    # Owned by the Qt main thread (set by _create_qt_app / _build_ui).
    qt_app: QApplication
    tray: TrayIcon
    orb: OverlayOrb
    hotkeys: HotkeyManager
    research_panel: ResearchPanel

    def __init__(self) -> None:
        # Handler state. _cfg_snapshot is the last config the UI published,
        # against which _on_config_change diffs the live one.
        self._cfg_snapshot: dict = {}
        self._quit_called = False

        # Panels that _on_quit and the tray handlers must tolerate being
        # absent: these were the `[None]` cells run() carried for exactly
        # the same reason, and they are read before _build_ui has run.
        self.deep_research_panel: DeepResearchPanel | None = None
        self.notes_panel: NotesPanel | None = None
        self.dashboard_panel: DashboardPanel | None = None
        self.help_panel: HelpPanel | None = None
        self.clipboard_panel: ClipboardHistoryPanel | None = None
        self.log_panel: LogPanel | None = None
        self.command_palette: CommandPalette | None = None
        self.onboarding_panel: OnboardingPanel | None = None
        self.settings_window: SettingsWindow | None = None

    # ------------------------------------------------------------------
    # Build phases
    # ------------------------------------------------------------------

    def start(self) -> int:
        """Compose and launch Jarvis. Returns the Qt exit code.

        The phase order below is the one documented at the top of this
        module and must not be rearranged: the Qt application exists
        before the audio boot so the failure dialogs have somewhere to
        appear, and the tray-availability check runs before the audio
        thread starts so a host without a tray exits without having
        loaded a single model.
        """
        self._load_config()

        # 1-7. Audio loop, core layer, audio modules, LLM stack, pipeline.
        self._build_audio_stack()

        # ------------------------------------------------------------------
        # 9. Qt application (created before audio boot so dialogs work)
        # ------------------------------------------------------------------
        self._create_qt_app()

        # ------------------------------------------------------------------
        # 10. System tray availability check
        # ------------------------------------------------------------------
        if not ensure_system_tray_available():
            self.audio_loop.close()
            return 1

        # ------------------------------------------------------------------
        # 8. Boot audio stack in dedicated thread; wait for ready
        # ------------------------------------------------------------------
        boot_exit = self._boot_audio_stack()
        if boot_exit is not None:
            return boot_exit

        # ------------------------------------------------------------------
        # 11. Per-interaction trace (DEBUG only; Settings -> Log level)
        # ------------------------------------------------------------------
        _wire_event_logging(self.bus)

        # ------------------------------------------------------------------
        # 12-14. Qt UI components
        # ------------------------------------------------------------------
        self._build_ui()

        # ------------------------------------------------------------------
        # 14. Qt event loop
        # ------------------------------------------------------------------
        return self.qt_app.exec()

    def _load_config(self) -> None:
        """Load config and install logging before anything else runs."""
        self.cfg = load_config()
        _setup_logging(self.cfg.general.log_level)
        from jarvis.paths import bundled_asset_report, is_frozen

        if is_frozen():
            log.info("bundled assets: %s", bundled_asset_report())
        self.voices_dir = _voices_dir()

    def _build_audio_stack(self) -> None:
        """Steps 1-7: everything the audio thread will own.

        Constructed on the main thread but handed to the audio thread by
        _run_audio_loop; nothing here touches Qt.
        """
        # ------------------------------------------------------------------
        # 1. Audio loop (created here; started in dedicated thread below)
        # ------------------------------------------------------------------
        self.audio_loop = asyncio.new_event_loop()

        # ------------------------------------------------------------------
        # 2-4. Core layer
        # ------------------------------------------------------------------
        self.bus = EventBus(loop=self.audio_loop)
        self.sm = StateMachine(bus=self.bus)

        # ------------------------------------------------------------------
        # 5. Audio modules
        # ------------------------------------------------------------------
        self.amplitude_latch, self.amplitude_callback = make_amplitude_callback()

        self.source = AudioInputSource(
            preferred_device=self.cfg.audio.input_device,
            prefer_respeaker=self.cfg.audio.prefer_respeaker,
            bus=self.bus,
        )
        self.wake_word = OpenWakeWord(sensitivity=self.cfg.wake_word.sensitivity)
        from jarvis.paths import (
            default_silero_onnx_path,
            default_whisper_download_root,
        )

        self.vad = SileroVAD(
            speech_threshold=0.5,
            model_path=default_silero_onnx_path(),
        )
        self.stt = FasterWhisperSTT(
            model_size=self.cfg.stt.model_size,
            language=self.cfg.stt.language,
            compute_type=self.cfg.stt.compute_type,
            download_root=default_whisper_download_root(),
        )
        self.tts = PiperTTS(
            voice_name=self.cfg.tts.voice or _PIPER_VOICE_NAME,
            voices_dir=self.voices_dir,
            volume=self.cfg.tts.volume,
            speed=self.cfg.tts.speed,
            output_device=self.cfg.audio.output_device,
            on_amplitude=self.amplitude_callback,
            bus=self.bus,
        )

        # ------------------------------------------------------------------
        # 6. LLM stack + tool registry
        # ------------------------------------------------------------------
        self.ollama = OllamaClient(
            model=self.cfg.llm.model,
            temperature=self.cfg.llm.temperature,
            max_tokens=self.cfg.llm.max_tokens,
            system_prompt=self.cfg.llm.system_prompt,
            keep_alive_seconds=self.cfg.llm.keep_alive_seconds,
        )
        self.conversation = Conversation(
            system_prompt_provider=lambda: self.cfg.llm.system_prompt,
            max_turns=self.cfg.llm.max_turns,
            inactivity_timeout_seconds=self.cfg.llm.inactivity_timeout_seconds,
        )
        self.registry = ToolRegistry(self.cfg.tools)
        setup_local_tools(self.registry, config=self.cfg, ollama_client=self.ollama)
        # Research tools are wired later (step 12) after the Qt app and
        # ResearchPanel are created, because their callbacks reference the panel.
        self.mcp_manager = MCPManager(self.registry)
        self.router = IntentRouter(
            llm=self.ollama,
            conversation=self.conversation,
            registry=self.registry,
            max_tool_iterations=self.cfg.llm.max_tool_iterations,
        )

        self.lm = LifecycleManager(
            [self.source, self.wake_word, self.vad, self.stt, self.tts, self.ollama],
            bus=self.bus,
        )

        # Re-asserted in BUILD.md Phase 6 design note: AudioInputSource IS in
        # the lifecycle list. On SLEEPING, its unload closes the input stream,
        # releasing the mic device. On wake, load reopens it; the pipeline's
        # start() re-attaches its on_frame callback. SPEC table did not list
        # source explicitly; we close it deliberately to free the device.

        self.pipeline = AudioPipeline(
            source=self.source,
            wake_word=self.wake_word,
            vad=self.vad,
            stt=self.stt,
            tts=self.tts,
            response_producer=_make_router_adapter(self.router, self.conversation, self.registry),
            bus=self.bus,
            sm=self.sm,
            on_wake=lambda: self.conversation.maybe_clear(
                self.cfg.llm.conversation_continuity_seconds
            ),
            log_wake_during_speaking=self.cfg.debug.log_wake_during_speaking,
        )

        # ------------------------------------------------------------------
        # 7. Mode coordinator (Phase 6 Task 1)
        # ------------------------------------------------------------------
        # Owns the speak-then-unload sequence on Sleep, the load-then-restart
        # sequence on Wake, and the Sleep/Wake/Mute race rules. Replaces
        # lm.bind(bus) as the driver for Mode transitions.
        self.mode_coord = ModeCoordinator(
            sm=self.sm,
            lm=self.lm,
            pipeline=self.pipeline,
            tts=self.tts,
            sleep_confirmation=self.cfg.general.sleep_confirmation,
        )

    def _create_qt_app(self) -> None:
        self.qt_app = QApplication.instance() or QApplication(sys.argv)

    def _boot_audio_stack(self) -> int | None:
        """Step 8: start the audio thread and wait for it to report ready.

        Returns an exit code if boot failed fatally (the caller must
        return it), or None to continue.
        """
        self.stop_event = asyncio.Event()
        self.boot_error_holder: list[str | None] = [None]
        self.boot_done = threading.Event()

        self.audio_thread = threading.Thread(
            target=self._run_audio_loop,
            daemon=True,
            name="jarvis-audio",
        )
        self.audio_thread.start()

        log.info("waiting for audio stack boot (up to %.0f s)...", _AUDIO_BOOT_TIMEOUT)
        if not self.boot_done.wait(timeout=_AUDIO_BOOT_TIMEOUT):
            QMessageBox.critical(
                None,  # type: ignore[arg-type]
                "Jarvis — Startup Error",
                "Jarvis timed out loading audio modules.\n\n"
                "Check that all prerequisites are installed and try again.",
            )
            self.audio_loop.call_soon_threadsafe(self.stop_event.set)
            self.audio_thread.join(timeout=5.0)
            return 1

        error = self.boot_error_holder[0]
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
                detail = (
                    error[len("load_failure:"):]
                    if error.startswith("load_failure:")
                    else error
                )
                QMessageBox.critical(
                    None,  # type: ignore[arg-type]
                    "Jarvis — Startup Error",
                    f"A required module failed to load:\n\n{detail}\n\n"
                    "Check that all prerequisites are installed (models downloaded, "
                    "audio devices connected) and try again.",
                )
                self.audio_loop.call_soon_threadsafe(self.stop_event.set)
                self.audio_thread.join(timeout=5.0)
                return 1

    def _run_audio_loop(self) -> None:
        """The audio thread's entry point: own the loop, run _audio_main."""
        asyncio.set_event_loop(self.audio_loop)
        self.audio_loop.run_until_complete(
            _audio_main(
                self.lm, self.pipeline, self.ollama, self.tts, self.stt, self.wake_word,
                self.bus, self.sm, self.mode_coord, self.cfg.lifecycle,
                self.stop_event, self.boot_error_holder, self.boot_done,
                source=self.source,
                mcp_manager=self.mcp_manager,
                mcp_servers=self.cfg.mcp_servers,
                registry=self.registry,
            )
        )
        self.audio_loop.close()

    # ------------------------------------------------------------------
    # Qt main-thread handlers
    #
    # Every method in this section runs on the Qt thread, and every one of
    # them that has to reach the audio stack does so through the bridges
    # named in the module docstring -- never by calling into it directly.
    # ------------------------------------------------------------------

    def _on_config_change(self) -> None:
        new_dict = self.cfg.model_dump(mode="json")
        old_dict = self._cfg_snapshot
        changed = _compute_changed_fields(old_dict, new_dict)
        if not changed:
            return
        old_cfg = JarvisConfig.model_validate(old_dict)
        new_cfg = JarvisConfig.model_validate(new_dict)
        self._cfg_snapshot = new_dict
        panel = self.research_panel
        if panel is not None and "ui.research_panel_width" in changed:
            panel.set_panel_width(new_cfg.ui.research_panel_width)
        self.bus.publish(ConfigChanged(old=old_cfg, new=new_cfg, changed_fields=tuple(changed)))

    def _on_test_voice(self, phrase: str) -> None:
        """Called from the Qt thread; dispatches tts.speak to the audio loop."""
        try:
            asyncio.run_coroutine_threadsafe(self.tts.speak(phrase), self.audio_loop)
        except Exception:
            log.exception("test-voice dispatch failed")

    def _open_settings(self) -> None:
        """Create or raise the settings window. Must run on Qt main thread."""
        if self.settings_window is None:
            self.settings_window = SettingsWindow(
                config=self.cfg,
                on_change=self._on_config_change,
                voices_dir=self.voices_dir,
                on_test_voice=self._on_test_voice,
            )
        win = self.settings_window
        win.show()
        win.raise_()
        win.activateWindow()

    def _open_settings_any_thread(self) -> None:
        """Thread-safe entry point for hotkey manager (pynput thread)."""
        QTimer.singleShot(0, self._open_settings)

    def _on_quit(self) -> None:
        if self._quit_called:
            return
        self._quit_called = True

        # Signal audio loop → triggers cleanup coroutine in audio thread
        self.audio_loop.call_soon_threadsafe(self.stop_event.set)

        # Block Qt main thread until audio stack drains (tray hidden below
        # so the user sees Jarvis disappear immediately; the wait is invisible)
        try:
            self.tray.hide()
        except Exception:
            log.debug("tray.hide() failed during quit", exc_info=True)

        if self.audio_thread.is_alive():
            self.audio_thread.join(timeout=_AUDIO_SHUTDOWN_TIMEOUT)
            if self.audio_thread.is_alive():
                log.warning("audio thread did not stop within %.0f s", _AUDIO_SHUTDOWN_TIMEOUT)

        # Python-level cleanup: unsubscribes, timer stops, etc.
        self.tray.close()
        self.orb.close()
        self.hotkeys.close()
        self.research_panel.close_panel()
        if self.deep_research_panel is not None:
            self.deep_research_panel.close_panel()
        if self.notes_panel is not None:
            self.notes_panel.close_panel()
        if self.dashboard_panel is not None:
            self.dashboard_panel.close_panel()
        if self.help_panel is not None:
            self.help_panel.close_panel()
        if self.clipboard_panel is not None:
            self.clipboard_panel.close_panel()
        if self.log_panel is not None:
            self.log_panel.close_panel()
        if self.command_palette is not None:
            self.command_palette.close_palette()
        if self.onboarding_panel is not None:
            self.onboarding_panel.close_panel()
        if self.settings_window is not None:
            self.settings_window.close()

        self.qt_app.quit()

    # Mode requests from tray/hotkeys route through the coordinator.
    # The returned coroutine is awaited on the audio loop by the
    # caller's run_coroutine_threadsafe wrapper.
    def _request_mode(self, target: Mode):
        return self.mode_coord.request(target)

    def _tray_open(self, panel):
        # The panel is read from its attribute by the caller's lambda, which
        # is why the tray can be wired to panels that do not exist yet.
        if panel is not None:
            panel.open_panel()

    def _tray_open_palette(self):
        palette = self.command_palette
        if palette is not None:
            palette.open_palette()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Steps 12-14: everything the Qt main thread will own."""
        self._cfg_snapshot = self.cfg.model_dump(mode="json")

        self.tray = TrayIcon(
            sm=self.sm,
            bus=self.bus,
            audio_loop=self.audio_loop,
            hotkeys=self.cfg.hotkeys,
            on_open_settings=self._open_settings,
            on_quit=self._on_quit,
            on_mode_request=self._request_mode,
            on_open_dashboard=lambda: self._tray_open(self.dashboard_panel),
            on_open_notes=lambda: self._tray_open(self.notes_panel),
            on_open_help=lambda: self._tray_open(self.help_panel),
            on_open_clipboard_history=lambda: self._tray_open(self.clipboard_panel),
            on_open_logs=lambda: self._tray_open(self.log_panel),
            on_open_command_palette=self._tray_open_palette,
            on_open_tutorial=lambda: self._tray_open(self.onboarding_panel),
        )
        self.tray.show()

        self.orb = OverlayOrb(sm=self.sm, bus=self.bus, amplitude_latch=self.amplitude_latch)

        # Research panel + tool registration. Panel lives on the Qt thread;
        # the tools emit cross-thread Signals to drive it from the audio loop.
        def _on_research_panel_width(width: int) -> None:
            if self.cfg.ui.research_panel_width != width:
                self.cfg.ui.research_panel_width = width
                self._on_config_change()

        self.research_panel = ResearchPanel(
            panel_width=self.cfg.ui.research_panel_width,
            on_width_changed=_on_research_panel_width,
            ollama_model=self.cfg.llm.model,
        )

        from jarvis.tools.local.research import (
            CloseResearchTool,
            CopyResearchTool,
            ReadMoreTool,
            ResearchTool,
        )
        self.registry.register(ResearchTool(
            on_start=self.research_panel.show_for_query,
            on_speak=self.tts.speak,
        ))
        self.registry.register(CloseResearchTool(
            close_callback=self.research_panel.close_panel,
        ))
        self.registry.register(ReadMoreTool(
            get_next=self.research_panel.get_next_sentences,
        ))
        self.registry.register(CopyResearchTool(
            copy_callback=self.research_panel.copy_summary,
        ))

        def _deep_research_config_provider():
            from jarvis.llm.ollama_client import DEFAULT_ENDPOINT
            from jarvis.tools.local.deep_research_runner import build_deep_research_config

            return build_deep_research_config(
                research=self.cfg.research,
                main_llm_model=self.cfg.llm.model,
                ollama_endpoint=DEFAULT_ENDPOINT,
            )

        def _set_deep_research_ultra(enabled: bool) -> str:
            from jarvis.core.config import save_config

            self.cfg.research.ultra_enabled = enabled
            save_config(self.cfg)
            if enabled:
                return (
                    "Deep research Ultra is on, sir. "
                    "Set JARVIS_BRAVE_API_KEY and JARVIS_GROQ_API_KEY for the full stack."
                )
            return "Deep research Ultra is off, sir. Using standard local deep research."

        self.deep_research_panel = DeepResearchPanel(
            config_provider=_deep_research_config_provider,
        )

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

        self.registry.register(DeepResearchTool(
            on_start=self.deep_research_panel.show_for_query,
            on_speak=self.tts.speak,
            ultra_enabled=lambda: self.cfg.research.ultra_enabled,
        ))
        self.registry.register(PauseDeepResearchTool(
            on_pause=self.deep_research_panel.pause_active,
        ))
        self.registry.register(ResumeDeepResearchTool(
            on_resume_latest=self.deep_research_panel.resume_latest_paused,
        ))
        self.registry.register(CloseDeepResearchTool(
            close_callback=self.deep_research_panel.close_panel,
        ))
        self.registry.register(DeleteDeepResearchTool(
            delete_by_query=self.deep_research_panel.delete_by_query,
            delete_active=self.deep_research_panel.delete_active,
        ))
        self.registry.register(DeleteAllDeepResearchTool(
            delete_all=self.deep_research_panel.delete_all,
        ))
        self.registry.register(EnableDeepResearchUltraTool(
            set_ultra=_set_deep_research_ultra,
        ))
        self.registry.register(DisableDeepResearchUltraTool(
            set_ultra=_set_deep_research_ultra,
        ))

        # --- Notes panel + voice tools ---------------------------------------
        self.notes_panel = NotesPanel()

        from jarvis.tools.local.notes_tools import (
            AppendToNoteTool,
            CloseNotesTool,
            DeleteNoteTool,
            OpenNotesTool,
            ReadNoteTool,
            TakeNoteTool,
        )

        def _take_note(title: str, content: str) -> str:
            return self.notes_panel.create_and_show(title, content)

        self.registry.register(TakeNoteTool(on_create=_take_note))
        self.registry.register(AppendToNoteTool(
            on_append_active=self.notes_panel.append_to_active,
            on_append_by_title=self.notes_panel.append_by_title,
        ))
        self.registry.register(ReadNoteTool(
            on_read_active=self.notes_panel.read_active,
            on_read_by_title=self.notes_panel.read_by_title,
        ))
        self.registry.register(OpenNotesTool(on_open=self.notes_panel.open_panel))
        self.registry.register(CloseNotesTool(on_close=self.notes_panel.close_panel))
        self.registry.register(DeleteNoteTool(
            on_delete_active=self.notes_panel.delete_active,
            on_delete_by_title=self.notes_panel.delete_by_title,
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

        self.dashboard_panel = DashboardPanel(
            sm=self.sm,
            amplitude_latch=self.amplitude_latch,
            config_provider=lambda: self.cfg,
            deep_research_count_provider=_dr_counts,
            notes_count_provider=_notes_count,
        )

        from jarvis.tools.local.dashboard_tools import (
            CloseDashboardTool,
            ShowDashboardTool,
        )

        self.registry.register(ShowDashboardTool(on_open=self.dashboard_panel.open_panel))
        self.registry.register(CloseDashboardTool(on_close=self.dashboard_panel.close_panel))

        # --- Help panel + voice tools ----------------------------------------
        self.help_panel = HelpPanel()

        from jarvis.tools.local.help_tools import OpenHelpTool

        self.registry.register(OpenHelpTool(on_open=self.help_panel.open_panel))

        # --- Clipboard history panel + voice tools ---------------------------
        self.clipboard_panel = ClipboardHistoryPanel()

        from jarvis.tools.local.clipboard_history_tools import (
            ClearClipboardHistoryTool,
            CloseClipboardHistoryTool,
            PasteClipboardItemTool,
            ShowClipboardHistoryTool,
        )

        self.registry.register(ShowClipboardHistoryTool(
            on_open=self.clipboard_panel.open_panel,
        ))
        self.registry.register(CloseClipboardHistoryTool(
            on_close=self.clipboard_panel.close_panel,
        ))
        self.registry.register(PasteClipboardItemTool(
            on_paste=self.clipboard_panel.paste_index,
        ))
        self.registry.register(ClearClipboardHistoryTool(
            on_clear=lambda: self.clipboard_panel.clear_all(keep_pinned=True),
        ))

        # --- Live log viewer panel + voice tools -----------------------------
        self.log_panel = LogPanel()

        from jarvis.tools.local.log_tools import CloseLogsTool, ShowLogsTool

        self.registry.register(ShowLogsTool(on_open=self.log_panel.open_panel))
        self.registry.register(CloseLogsTool(on_close=self.log_panel.close_panel))

        # --- Command palette -------------------------------------------------
        # Submission routes through the audio loop: we wrap the producer that
        # the AudioPipeline normally drives so palette entries fire the exact
        # same intent-router + tool pipeline as a real STT result, just without
        # the wake-word / VAD gating.
        palette_producer = _make_router_adapter(self.router, self.conversation, self.registry)

        async def _consume_palette_text(text: str) -> None:
            try:
                async for _chunk in palette_producer(text):
                    pass
            except Exception:
                log.exception("command palette text execution failed")

        def _submit_palette_text(text: str) -> None:
            try:
                asyncio.run_coroutine_threadsafe(
                    _consume_palette_text(text), self.audio_loop
                )
            except Exception:
                log.exception("could not schedule palette text onto audio loop")

        self.command_palette = CommandPalette(submit_text=_submit_palette_text)

        # --- Onboarding panel (auto-shown on first run) ----------------------
        def _on_onboarding_finished() -> None:
            if not self.cfg.general.first_run_completed:
                self.cfg.general.first_run_completed = True
                try:
                    self._on_config_change()
                except Exception:
                    log.exception("config persist failed after onboarding finish")

        self.onboarding_panel = OnboardingPanel(
            bus=self.bus,
            amplitude_latch=self.amplitude_latch,
            on_finished=_on_onboarding_finished,
            on_open_help=self.help_panel.open_panel,
            on_open_command_palette=self.command_palette.open_palette,
        )

        if not self.cfg.general.first_run_completed:
            # Defer to next Qt tick so the rest of the UI exists first.
            QTimer.singleShot(800, self.onboarding_panel.open_panel)

        self.hotkeys = HotkeyManager(
            sm=self.sm,
            bus=self.bus,
            audio_loop=self.audio_loop,
            hotkeys=self.cfg.hotkeys,
            on_mode_request=self._request_mode,
            on_open_settings=self._open_settings_any_thread,
            on_open_command_palette=lambda: QTimer.singleShot(
                0, self.command_palette.open_palette
            ),
        )
        self.hotkeys.register_all()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run() -> int:
    """Compose and launch Jarvis as a desktop application.

    Returns the Qt exit code (0 on clean quit, non-zero on error).
    """
    return JarvisApp().start()
