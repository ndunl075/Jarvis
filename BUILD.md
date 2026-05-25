# Jarvis Build Plan

This file is the entry point for Claude Code. Read this first, then read `SPEC.md` before writing any code. Do not skip phases. Do not write code for a later phase before the current phase is complete and verified.

## What you are building

A fully local, free, no-subscription AI voice assistant for Windows, packaged as a real installable Windows application. See `SPEC.md` for the full technical contract. Read it before coding.

## Project principles (non-negotiable)

1. **Quality over speed.** This is mission-critical. No shortcuts on architecture.
2. **Spec-driven.** If something is not in `SPEC.md`, do not invent it. Stop and ask.
3. **Module boundaries are sacred.** The interfaces defined in `SPEC.md` (config, events, lifecycle, state machine) are the contracts every module obeys. Never bypass them.
4. **Windows-first, but platform-abstracted.** All OS-specific code goes through `platform/`. No `win32` imports outside that folder.
5. **Test as you go.** Each module gets unit tests before the next module starts.
6. **No mock-driven development.** If you cannot test a module without mocking everything, the module is wrong. Fix the design.

## Model usage guidance

Tasks in this plan are tagged with a recommended model:

- **[ARCH]** — Architecture-critical. Use the strongest reasoning model available. These decisions compound; getting them wrong costs days. If you only have one model, slow down and think carefully on these.
- **[IMPL]** — Implementation. Routine code against an existing spec. Any capable model handles this well.
- **[GLUE]** — Wiring, packaging, scripts. Mechanical work.

Treat the tags as priority signals: spend more thinking time on [ARCH] tasks regardless of which model is running.

## Phases

Each phase has a definition of done. Do not advance until it is met.

---

### Phase 0: Repo setup [GLUE]

**Goal:** clean repo skeleton, no functionality.

Tasks:
1. Create the directory structure exactly as defined in `SPEC.md` § Directory Layout. Empty `__init__.py` files in every package.
2. Set up `pyproject.toml` with the dependencies from `SPEC.md` § Dependencies. Pin major versions.
3. Set up `pytest`, `ruff`, and `pyright` (or `mypy`) configs.
4. Create `.gitignore` for Python, PyInstaller build artifacts, and model files.
5. Create `README.md` stub.
6. Verify `pip install -e .` works in a fresh venv.

**Done when:** repo installs cleanly, `pytest` runs (zero tests, exits 0), `ruff check .` passes.

---

### Phase 1: Core contracts [ARCH]

**Goal:** the four interfaces every module depends on. This is the single highest-leverage phase. Get it right.

Tasks:
1. Implement `core/config.py` per `SPEC.md` § Config Schema. Pydantic v2. Loads from and saves to `%APPDATA%/Jarvis/config.json`. Includes validation, defaults, and a migration hook for future schema versions.
2. Implement `core/events.py` per `SPEC.md` § Event Bus. Typed pub/sub. Async-friendly. Subscribers receive events on the asyncio loop, not on the publisher's thread.
3. Implement `core/state_machine.py` per `SPEC.md` § State Machine. Two orthogonal axes: `Mode` and `ConversationalState`. Explicit transition table. Illegal transitions raise. Emits events on every transition.
4. Implement `core/lifecycle.py` per `SPEC.md` § Lifecycle Contract. Defines the `Loadable` protocol with `load()` / `unload()`. Lifecycle manager orchestrates module load/unload in declared order.
5. Write unit tests covering: config round-trip, event bus delivery and ordering, every legal state transition, every illegal state transition raises, lifecycle load/unload ordering.

**Done when:** all four modules pass tests, type-check clean, and a smoke test shows config changes emitting events that the state machine and a fake module can react to.

**Do not start Phase 2 until a human has reviewed Phase 1.** This is the layer everything else sits on.

---

### Phase 2: Audio pipeline [ARCH for design, IMPL for each module]

**Goal:** end-to-end audio path with no LLM yet. You can say "hey jarvis," it transcribes, and it echoes the transcription via TTS.

Tasks:
1. **[ARCH]** `audio/pipeline.py` — the orchestrator that wires wake word → VAD → STT → (placeholder) → TTS, driven by the state machine. Defines the in-memory audio frame format and the queue topology between stages. Handles barge-in: if VAD detects speech during SPEAKING state, cancel TTS and transition to LISTENING.
2. **[IMPL]** `audio/devices.py` — enumerates input devices, auto-selects ReSpeaker if present, falls back to default. Exposes a `Loadable` audio source.
3. **[IMPL]** `audio/wake_word.py` — openWakeWord wrapper, "hey jarvis" model, configurable sensitivity, emits `WakeWordDetected` events.
4. **[IMPL]** `audio/vad.py` — silero-vad wrapper, used both for endpointing during LISTENING and barge-in during SPEAKING.
5. **[IMPL]** `audio/stt.py` — faster-whisper wrapper. Default model `base.en`, int8 quantized, CPU. Configurable.
6. **[IMPL]** `audio/tts.py` — Piper wrapper, streams audio sentence-by-sentence so playback can begin before generation completes. Cancellable mid-stream.
7. **[IMPL]** Tests for each module against recorded fixture audio (commit small WAV fixtures to `tests/fixtures/audio/`).

**Done when:** running `python -m jarvis.dev.audio_loopback` lets you say "hey jarvis, hello world" and hear "hello world" spoken back. Barge-in works: interrupt the TTS by speaking and it stops.

---

### Phase 3: LLM and intent layer [ARCH for router, IMPL for client]

**Goal:** Jarvis can answer open-ended questions and execute simple commands.

Tasks:
1. **[IMPL]** `llm/ollama_client.py` — async client for Ollama. Streaming token output. Supports `keep_alive` parameter (critical for the Sleep mode model eviction). Configurable model name.
2. **[ARCH]** `llm/intent_router.py` — the hybrid router from `SPEC.md` § Intent Routing. Pattern-matches deterministic commands first ("open spotify," "volume up," "screenshot"), escalates ambiguous input to the LLM with tool-calling. Returns a typed `Intent` result. Document the routing table clearly; this is the brain of how Jarvis feels to use.
3. **[IMPL]** `llm/conversation.py` — turn history with configurable window. Resets after N minutes of inactivity (configurable). Injects system prompt from config.
4. **[IMPL]** Tests: router returns correct intent for each pattern category, conversation history truncates correctly, ollama client handles streaming and cancellation.

**Done when:** the audio loopback from Phase 2 now goes through the LLM. You can say "hey jarvis, what's the capital of france" and get a spoken answer. You can say "hey jarvis, take a screenshot" and the router returns a tool intent (no execution yet — that's Phase 4).

---

### Phase 4: Tool registry and local tools [ARCH for registry, IMPL for tools]

**Goal:** Jarvis can take actions on the PC.

Tasks:
1. **[ARCH]** `tools/registry.py` — the MCP-compatible tool interface from `SPEC.md` § Tool Registry. Every tool, local or remote, implements the same protocol. Tools declare a JSON schema for their arguments. Registry exposes a discovery method the LLM uses for function calling. Per-tool enable/disable from config.
2. **[IMPL]** `tools/local/` — one file per tool: `open_app.py`, `open_url.py`, `screenshot.py`, `volume.py`, `files.py`, `clipboard.py`, `type_text.py`, `lock_screen.py`, `system_stats.py`. Each implements the registry protocol. Each has tests.
3. **[IMPL]** `tools/mcp_client.py` — connects to external MCP servers, registers their tools into the local registry. Handles connection lifecycle, retries, and graceful degradation when a server is offline.
4. **[IMPL]** Wire the router from Phase 3 to actually execute tool intents through the registry.

**Done when:** all local tools work end-to-end via voice. "Hey jarvis, open spotify" opens it. "Take a screenshot." "Lock the screen." "What's my CPU usage." Each tool can be disabled in config and the LLM stops being told about it.

---

### Phase 5: UI layer [ARCH for orb design, IMPL for everything else]

**Goal:** the visible app. Tray icon, overlay orb, settings window.

Tasks:
1. **[IMPL]** `ui/tray.py` — system tray icon with the menu defined in `SPEC.md` § Tray Menu. Reflects current `Mode` visually. Menu items dispatch mode transitions through the state machine, never directly to modules.
2. **[ARCH]** `ui/overlay.py` — the Siri-style orb. Frameless, transparent, always-on-top, click-through. Subscribes to `ConversationalState` events only (never to `Mode`). Animation states: idle (hidden), listening (pulsing), thinking (rotating), speaking (waveform reactive to TTS amplitude). Decide on the animation approach (QPainter vs QML) and document it.
3. **[IMPL]** `ui/settings/main_window.py` and the tab modules (`general.py`, `voice.py`, `models.py`, `tools.py`, `hotkeys.py`, `about.py`). Each tab reads from and writes to the config layer; never touches modules directly. Per `SPEC.md` § Settings Tabs.
4. **[IMPL]** `ui/hotkeys.py` — global hotkey registration. Default bindings: Mute, Wake (push-to-talk alternative), Open Settings.
5. **[IMPL]** Tests where feasible (config round-trip via UI is hard to test; focus on the non-UI logic).

**Done when:** the app launches with a tray icon, the overlay appears during voice interactions, and every setting in the settings window persists across restarts and takes effect without restarting the app where reasonable (model size changes may require a reload, voice changes should be live).

---

### Phase 6: Mode lifecycle and Ollama eviction [ARCH]

**Goal:** Sleep mode actually frees RAM and VRAM.

Tasks:
1. **[ARCH]** Implement the full `Mode` transition logic in `core/lifecycle.py`. ACTIVE → MUTED is cheap. ACTIVE → SLEEPING unloads whisper, piper, openWakeWord, and evicts the Ollama model via `keep_alive: 0`. SLEEPING → ACTIVE reloads everything with a "waking" tray indicator.
2. **[IMPL]** Verify with Task Manager that SLEEPING actually drops the process RSS by the expected amount and Ollama's VRAM usage drops to zero.
3. **[IMPL]** Mid-response Mute behavior: cancels TTS immediately, transitions to IDLE, then to MUTED. No audio bleeds into the muted state.
4. **[DEFERRED]** AEC (Acoustic Echo Cancellation) via webrtc-audio-processing. Wake-word interrupt confirmed working without AEC on tested hardware (2026-05-19). Deferred to v1.1 if user reports indicate AEC is needed for other hardware setups.

**Done when:** mode transitions are clean, observable in Task Manager, and never leak audio or hang.

---

### Phase 7: Packaging [GLUE, but read carefully]

**Goal:** a real `Jarvis-Setup.exe` file someone can download and run.

Tasks:
1. **[IMPL]** PyInstaller spec file. One-folder build, not one-file. Use `--collect-all PySide6`. Include model files for whisper (base.en), piper (chosen voice), openWakeWord. Do not bundle Ollama or LLM weights.
2. **[IMPL]** Inno Setup script. Installs to Program Files, creates Start Menu shortcut, optional autostart, proper uninstaller, registers in Add/Remove Programs.
3. **[IMPL]** First-launch experience: detects Ollama, prompts to install if missing (open the Ollama download page in browser), runs `ollama pull` for the configured model with progress shown in the UI.
4. **[IMPL]** README with install instructions, the SmartScreen warning explanation, and a screenshot.
5. **[GLUE]** A `build.ps1` or `build.bat` script that runs PyInstaller, then Inno Setup, and outputs `dist/Jarvis-Setup-vX.Y.Z.exe`.

**Done when:** running `build.ps1` on a clean machine produces a working installer that installs, runs, and uninstalls cleanly.

---

## How to use this plan in Claude Code

1. At the start of every session, read `BUILD.md` and `SPEC.md` first. Do not skim. The interfaces in SPEC.md are referenced by file and section name throughout the build.
2. State which phase and task you are working on before writing code.
3. If a task is **[ARCH]**, write the design as a comment block or a short markdown note in the file before writing the code. Include the alternatives you considered and why you rejected them. This is not optional for [ARCH] tasks.
4. If you find a contradiction between this plan and SPEC.md, stop and ask. Do not paper over it.
5. After completing a task, run the tests for that task and report the result before moving to the next task.
6. Never modify SPEC.md without explicit approval. BUILD.md can be amended as you learn things; SPEC.md is the contract.

## Token efficiency tips for the operator

- Don't paste the full BUILD.md into every prompt. Once Claude Code has it indexed in the project, reference tasks by section: "begin Phase 2 Task 3."
- For [IMPL] tasks, a short prompt is fine: "implement Phase 4 Task 2, screenshot.py, per spec."
- For [ARCH] tasks, give Claude Code room to reason. Don't rush it.
- Keep the test suite green between tasks. Debugging compounds quickly when multiple modules are broken.
