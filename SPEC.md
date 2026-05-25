# Jarvis Technical Spec

This is the contract. BUILD.md references this document by section. Do not deviate without explicit approval.

## Product summary

Jarvis is a Windows desktop AI voice assistant. Fully local, no subscription, no cloud dependencies (except optional MCP integrations the user configures). Wake-word activated. Capable of conversation and PC actions. Distributed as a Windows installer. The product persona is modeled after the Jarvis AI from Iron Man: calm, dryly witty, British, addresses the user as "sir."

## Directory layout

```
jarvis/
  __init__.py
  __main__.py                # entry point: launches tray + UI thread + audio loop
  core/
    config.py                # pydantic schema, persistence
    events.py                # typed pub/sub bus
    state_machine.py         # Mode + ConversationalState
    lifecycle.py             # Loadable protocol, lifecycle manager
    logging.py               # structured logging setup
  audio/
    pipeline.py              # orchestrator
    devices.py               # input device selection
    wake_word.py             # openWakeWord
    vad.py                   # silero-vad
    stt.py                   # faster-whisper
    tts.py                   # Piper, streaming
  llm/
    ollama_client.py         # async ollama client
    intent_router.py         # hybrid pattern + LLM router
    conversation.py          # turn history
  tools/
    registry.py              # tool protocol, MCP-compatible
    mcp_client.py            # external MCP server connections
    local/
      open_app.py
      open_url.py
      screenshot.py
      volume.py
      files.py
      clipboard.py
      type_text.py
      lock_screen.py
      system_stats.py
  ui/
    tray.py
    overlay.py               # the orb
    hotkeys.py
    settings/
      main_window.py
      tabs/
        general.py
        voice.py
        models.py
        tools.py
        hotkeys.py
        about.py
  platform/
    base.py                  # platform abstraction interface
    windows.py
    darwin.py                # stub for now
    linux.py                 # stub for now
  dev/
    audio_loopback.py        # phase 2 dev harness
tests/
  fixtures/
    audio/
  core/
  audio/
  llm/
  tools/
  ui/
```

No `win32` imports outside `platform/windows.py`. No file outside `ui/` may import from `PySide6`. No file outside `audio/` may import audio libraries. These rules exist to keep modules swappable.

## Dependencies

Pin major versions in `pyproject.toml`:

- `pydantic` >= 2.0
- `PySide6` (LGPL, fine for our distribution model)
- Tray is provided by Qt's `QSystemTrayIcon` (no `pystray`)
- `openwakeword`
- `faster-whisper`
- `silero-vad` (prefer ONNX runtime version to avoid bundling torch)
- `piper-tts` (Python bindings)
- `sounddevice` (preferred over pyaudio: cleaner API, fewer Windows install issues)
- `httpx` for the Ollama client (async support)
- `psutil`
- `pyautogui` for typing/automation
- `pillow` for screenshots
- `pynput` or `keyboard` for global hotkeys (decide in Phase 5; `keyboard` requires admin on Windows for some keys, `pynput` does not)

Dev:
- `pytest`, `pytest-asyncio`
- `ruff`
- `pyright` or `mypy`
- `pyinstaller`

## Config schema

Located at `%APPDATA%/Jarvis/config.json`. Pydantic v2 model. Required top-level fields:

```python
class JarvisConfig(BaseModel):
    schema_version: int = 1
    general: GeneralConfig
    audio: AudioConfig
    wake_word: WakeWordConfig
    stt: STTConfig
    tts: TTSConfig
    llm: LLMConfig
    tools: ToolsConfig
    hotkeys: HotkeysConfig
    mcp_servers: list[MCPServerConfig]
```

Each sub-model:

- `GeneralConfig`: `start_with_windows: bool`, `minimize_to_tray: bool`, `show_overlay: bool`, `log_level: Literal["DEBUG","INFO","WARNING","ERROR"]`
- `AudioConfig`: `input_device: str | None` (None means auto-detect), `output_device: str | None` (None means system default; matched by case-insensitive name substring against sounddevice's output device list), `prefer_respeaker: bool = True` (auto-detect ReSpeaker if connected, otherwise fall back to default input device)
- `WakeWordConfig`: `enabled: bool`, `sensitivity: float` (0.0 to 1.0), `model: str = "hey_jarvis"`
- `STTConfig`: `model_size: Literal["tiny","base","small"] = "base"`, `language: str = "en"`, `compute_type: Literal["int8","float16","float32"] = "int8"`
- `TTSConfig`: `voice: str = "en_GB-alan-medium"`, `speed: float = 1.0`, `volume: float = 1.0`
- `LLMConfig`: `model: str = "qwen2.5:7b-instruct"`, `temperature: float = 0.7`, `max_tokens: int = 1024` (per-response cap), `system_prompt: str` (see default below), `keep_alive_seconds: int = 1800` (Ollama VRAM-resident duration; SLEEPING mode evicts via keep_alive=0), `max_turns: int = 10` (conversation history length, distinct from `max_tokens`; system prompt is not counted as a turn), `inactivity_timeout_seconds: int = 300` (idle time after which the next user message starts a fresh conversation)
- `ToolsConfig`: `enabled: dict[str, bool]` (tool name to on/off)
- `HotkeysConfig`: `mute: str`, `push_to_talk: str | None`, `open_settings: str`
- `MCPServerConfig`: `name: str`, `url_or_command: str`, `enabled: bool`, `tools_enabled: dict[str, bool]`

### Default system prompt

```
You are Jarvis, an AI assistant inspired by the one from Iron Man. You are calm, dryly witty, efficient, and address the user as "sir." Keep responses concise, usually one or two sentences. You have access to tools for controlling the PC; use them when appropriate. Never break character.
```

### Migration

Include a `migrate(old_config: dict) -> dict` function dispatched on `schema_version`. Stub for v1 to v2 even though v2 doesn't exist; just establish the pattern.

## Event bus

Typed pub/sub. Events are dataclasses or pydantic models. Subscribers register a handler for a specific event type. Delivery is async, on the main asyncio loop. Publishers can be sync or async; the bus handles the boundary.

Required event types (initial set):

- `ModeChanged(old: Mode, new: Mode)`
- `ConversationalStateChanged(old: ConversationalState, new: ConversationalState)`
- `ConfigChanged(path: tuple[str, ...], old_value, new_value)` where `path` is the dotted path into the config tree
- `WakeWordDetected(confidence: float)`
- `TranscriptionReady(text: str, duration_ms: int)`
- `LLMResponseChunk(text: str)`
- `LLMResponseComplete(full_text: str)`
- `ToolInvoked(tool_name: str, args: dict)`
- `ToolResult(tool_name: str, result: dict | str, error: str | None)`

The bus must guarantee that handlers for the same event are called in registration order, and that one handler raising does not prevent others from running. Log exceptions; do not propagate.

## State machine

Two orthogonal state axes.

```python
class Mode(Enum):
    ACTIVE = "active"
    MUTED = "muted"
    SLEEPING = "sleeping"

class ConversationalState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
```

`ConversationalState` is only meaningful when `Mode == ACTIVE`. In MUTED or SLEEPING, ConversationalState is forced to IDLE.

### Legal Mode transitions

- ACTIVE to/from MUTED
- ACTIVE to/from SLEEPING
- MUTED to/from SLEEPING

All other Mode transitions raise `IllegalTransition`.

On every Mode transition:
- Cancel any in-progress TTS
- Force ConversationalState to IDLE
- Trigger lifecycle changes (see Lifecycle Contract)
- Emit `ModeChanged`

### Legal ConversationalState transitions (only in ACTIVE mode)

- IDLE to LISTENING (wake word detected, or push-to-talk)
- LISTENING to THINKING (VAD endpoint hit, transcription complete)
- LISTENING to IDLE (silence timeout, no speech detected)
- THINKING to SPEAKING (LLM started producing tokens)
- THINKING to IDLE (LLM produced no output, or tool-only intent with no spoken response)
- SPEAKING to IDLE (TTS complete)
- SPEAKING to LISTENING (barge-in: VAD detected user speech mid-response)
- Any state to IDLE (cancel/reset)

All other transitions raise `IllegalTransition`.

## Lifecycle contract

```python
class Loadable(Protocol):
    name: str
    is_loaded: bool
    async def load(self) -> None: ...
    async def unload(self) -> None: ...
```

Modules implementing `Loadable`: `wake_word`, `vad`, `stt`, `tts`, `ollama_client` (the eviction case), and any other module holding heavy resources.

`LifecycleManager`:
- Maintains an ordered list of loadables
- `load_all()` loads in declared order
- `unload_all()` unloads in reverse order
- `transition_to_mode(mode)` dispatches the right load/unload calls per the table below

| Mode transition       | Action                                                                         |
|-----------------------|--------------------------------------------------------------------------------|
| to ACTIVE (from MUTED) | No-op (everything stays loaded)                                                |
| to ACTIVE (from SLEEP) | Load all audio modules; emit "waking" UI signal; LLM loads lazily on first use |
| to MUTED (from ACTIVE) | No-op on modules; pipeline starts ignoring wake word events                    |
| to MUTED (from SLEEP)  | Same as to ACTIVE then suppress wake word                                      |
| to SLEEPING            | Unload wake_word, vad, stt, tts; call ollama with `keep_alive: 0` to evict    |

Ollama eviction: send a minimal request (or use the dedicated unload endpoint if available in the version the user has) with `keep_alive: 0` to force the model out of VRAM. Do not skip this; it is the entire point of Sleep mode.

## Audio pipeline

The pipeline owns the audio source and routes frames through stages. Stages communicate via async queues. The pipeline subscribes to state events and reacts:

- IDLE: feed frames only to wake_word
- LISTENING: feed frames to vad + stt buffer; when vad signals endpoint, send buffer to stt, emit `TranscriptionReady`
- THINKING: ignore audio frames except for barge-in detection (vad only)
- SPEAKING: feed frames to vad for barge-in; if speech detected, cancel TTS and transition to LISTENING

Audio frame format: 16kHz mono int16, 30ms frames (480 samples). All stages assume this format. Resampling, if needed for a given stage, happens at the stage boundary, not in the pipeline.

## Intent routing

The router receives a transcription string and returns an `Intent`:

```python
@dataclass
class SpeakIntent:
    text: str  # what to say back; goes through TTS

@dataclass
class ToolIntent:
    tool_name: str
    args: dict
    spoken_response: str | None  # optional confirmation to speak

@dataclass
class CompoundIntent:
    intents: list[Intent]  # for "open spotify and tell me the weather" style

Intent = SpeakIntent | ToolIntent | CompoundIntent
```

### Routing strategy

1. **Pattern layer** (deterministic, fast, no LLM):
   - "open <app>" to ToolIntent("open_app", {"name": <app>})
   - "search <query>" or "google <query>" to ToolIntent("open_url", {"url": "https://www.google.com/search?q=..."})
   - "volume up/down/mute" to ToolIntent("volume", {...})
   - "screenshot" / "take a screenshot" to ToolIntent("screenshot", {})
   - "lock the screen" / "lock my pc" to ToolIntent("lock_screen", {})
   - "what time is it" to SpeakIntent (computed locally, no LLM)
   - Patterns are case-insensitive, tolerant of filler words ("hey jarvis can you please...")

2. **LLM layer** (anything else):
   - Pass to Ollama with the tool registry as available functions
   - Model returns either a text response or a function call
   - Convert to SpeakIntent or ToolIntent

The router is the highest-impact UX module. The pattern layer is what makes it feel snappy; the LLM layer is what makes it feel smart. Document the pattern table clearly and make it easy to extend.

## Tool registry

```python
class Tool(Protocol):
    name: str                                  # snake_case, unique
    description: str                           # for the LLM
    args_schema: type[BaseModel]               # pydantic model for args
    requires_confirmation: bool                # for risky tools
    async def execute(self, args: BaseModel) -> ToolResult: ...
```

`ToolResult` is `{ success: bool, output: str | dict | None, error: str | None }`.

Registry:
- `register(tool: Tool)` / `unregister(name: str)`
- `list_enabled() -> list[Tool]` (filtered by config)
- `get(name: str) -> Tool | None`
- `as_openai_functions() -> list[dict]` (the schema format Ollama expects for tool calling)

External MCP tools register the same way; `mcp_client` adapts the MCP protocol to the `Tool` interface.

## Tray menu

Built with `QSystemTrayIcon`. Menu items:

```
● Jarvis: <Mode>           [non-clickable status row]
─────────────────────
[Mute / Unmute]            [hotkey shown on right]
[Sleep / Wake]
─────────────────────
Settings...
About
─────────────────────
Quit
```

Icon variants: active, muted (mic with slash), sleeping (zzz). Active icon may pulse subtly during ConversationalState != IDLE (optional polish).

## Overlay (orb)

`QWidget` with flags: `Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WA_TranslucentBackground | Qt.WindowTransparentForInput`.

Position: bottom-center of the primary screen by default, configurable.

States (drives animation):
- Hidden (ConversationalState == IDLE)
- Listening: gentle pulsing circle
- Thinking: rotating gradient or shimmering
- Speaking: waveform reactive to TTS amplitude (sample TTS output at ~30Hz for amplitude envelope)

Animation implementation: QPainter with a QTimer at 60fps is the simplest approach and plenty performant. QML is overkill unless you want fancier effects later.

The overlay subscribes to `ConversationalStateChanged` only. It never reads Mode; if Mode is not ACTIVE, ConversationalState will be IDLE and the overlay will be hidden.

## Settings tabs

Each tab is a `QWidget` with the controls listed. All controls bind to the config layer; changes write to config and emit `ConfigChanged`.

- **General:** start with windows (checkbox), minimize to tray (checkbox), show overlay (checkbox), log level (dropdown)
- **Voice:** input device (dropdown, refreshable), wake word sensitivity (slider), output voice (dropdown of installed Piper voices + "download more..." button that opens browser to Piper voice catalog), speed (slider), volume (slider), test button (plays "hello, this is jarvis")
- **Models:** whisper size (dropdown: tiny/base/small), Ollama model (dropdown populated from `ollama list`), temperature (slider), max tokens (number), system prompt (multiline text, defaulting to the Jarvis persona prompt above)
- **Tools:** list of all registered tools (local + MCP) with enable checkboxes; section for MCP servers with add/edit/remove/test buttons
- **Hotkeys:** capture-style input for each hotkey
- **About:** version, GitHub link, Ollama install status, model status, "check for updates" (stub for now)

## First-launch flow

1. Detect Ollama: try connecting to `http://localhost:11434`. If it fails, show a dialog explaining Ollama is required, with a button to open the Ollama download page.
2. Once Ollama is running, check if the configured model (`qwen2.5:7b-instruct` by default) is pulled. If not, run `ollama pull <model>` and show progress.
3. Verify Piper voice file (`en_GB-alan-medium`) is present in the install directory (it should be, since the installer ships it).
4. On success, dismiss the wizard, show the tray icon, transition to ACTIVE.

## Logging

Structured logs to `%APPDATA%/Jarvis/logs/jarvis.log`. Rotate at 10MB, keep 5 files. Log level from config. Include module name, level, timestamp, and a correlation id for each "interaction" (one wake word event through to TTS completion shares one id).

## Threading model

- One asyncio event loop, runs in a dedicated thread (the audio thread)
- Qt main thread runs the UI event loop
- Communication between them via Qt signals (UI to audio: emit signal; audio to UI: schedule a callback on the Qt thread via `QMetaObject.invokeMethod`)
- Never call PySide6 from the asyncio loop directly. Never call asyncio things from the Qt loop directly.

## Out of scope (explicitly)

- Vision and gesture recognition
- Cloud LLMs
- Mac and Linux runtime support (skeleton only, no testing, no packaging)
- Auto-update mechanism (stub the button; defer to a later version)
- Multi-language support (English only for v1)
- Multi-user / multi-profile config
- Voice cloning or non-Piper TTS engines (revisit post-v1 if alan-medium proves unsatisfactory)

## Resolved decisions

These were open questions during planning and are now locked:

- **Default LLM:** `qwen2.5:7b-instruct` (chosen for tool-calling reliability over the smaller llama3.2:3b)
- **Default TTS voice:** `en_GB-alan-medium` (closest match to the Iron Man Jarvis vibe in the standard Piper catalog)
- **Persona:** baked into the default system prompt above. Calm, dryly witty, British, addresses user as "sir."
- **Primary mic target:** desk-mounted USB cardioid (e.g., FIFINE K669B). ReSpeaker is auto-detected if connected but not required.

## Open questions to resolve in implementation

These are flagged so Claude Code does not silently make decisions:

1. **Hotkey library choice** (Phase 5): `pynput` vs `keyboard`. Decide based on Windows admin requirements and key coverage at the time of implementation.
2. **Tool-call confirmation UX** (Phase 4): for `requires_confirmation=True` tools, do we show a toast and require voice confirmation, or auto-execute with a "say cancel within 3 seconds" pattern? Decide in Phase 4.
