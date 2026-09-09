# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Jarvis is pre-1.0: while the major version is `0`, the public surface
(config schema, tool names, `SPEC.md` contracts) may change in a minor release.
Config-schema changes always ship with a migration, so upgrading in place never
loses settings.

## [Unreleased]

### Added

- Tool-result feedback loop. When the model calls a tool, its result is now fed
  back as a `role: "tool"` message and the model is re-invoked, so it can act on
  what came back: "what's the weather, and if it's below 10 open my coat app",
  or "list my Downloads folder and tell me which one is the invoice". Previously
  every interaction was one-shot — the result was spoken at you and the model
  never saw it, so nothing that took two steps was possible.
- `llm.max_tool_iterations` (config schema v21, default 3) caps how many model
  invocations one turn may use. Each iteration is another full inference, so
  this is a latency budget; on reaching it Jarvis speaks the tool output it has
  rather than looping. Set it to `1` to disable the loop and restore the old
  one-shot behaviour, where a tool's own output is spoken verbatim.
- Continuous integration (`.github/workflows/ci.yml`): ruff, pyright, and pytest
  run as blocking gates on `ubuntu-latest` and `windows-latest` across Python
  3.11 and 3.12, using the existing `pyproject.toml` configuration.
- `CONTRIBUTING.md` — dev environment setup, the test/lint/type commands, the
  `manual` pytest marker convention, and the project's architectural rules.
- `SECURITY.md` — responsible-disclosure policy, supported versions, and an
  explicit statement of the local-execution, MCP, and Ultra-mode attack surface.
- `CHANGELOG.md` — this file.
- Issue forms for bug reports and feature requests, an issue-template config
  pointing questions at Discussions, and a pull request template.
- `requirements.lock` — a universal, hash-pinned resolution of the runtime
  dependencies for reproducible builds, covering both the Windows and Linux
  wheel sets in one file.

### Changed

- Snap commands are unaffected by the feedback loop: "open spotify", "volume
  up", "research X" and every other pattern-matched phrase still bypass the LLM
  entirely. Only tools the model chooses go through the loop, and those now
  answer in Jarvis's own words instead of reading the tool's raw output.
- `pyautogui` and `pynput` are now declared with a `sys_platform == "win32"`
  marker. Both are Windows-only input libraries, both are already imported
  lazily at their call sites, and both already degrade to a logged warning or a
  failed `ToolResult` when absent — but their Linux install path builds
  `pygetwindow` / `pyrect` / `python3-Xlib` from source and fails on a bare CI
  runner. No behaviour change on Windows.

## [0.0.1] - 2026-05-26

First public release. Everything below runs on the user's own PC; the only
network calls are the opt-in ones listed under *Deep research* and *Weather*.

> Note: `0.0.1` covers both commits in the repository's history — the initial
> public release (2026-05-25) and the see-screen vision tool (2026-05-26). No
> version bump occurred between them, so they are recorded as one entry.

### Added

**Voice loop**

- "Hey Jarvis" wake word (openWakeWord) with configurable sensitivity, plus a
  push-to-talk hotkey as an alternative.
- Voice activity detection and endpointing (silero-vad).
- On-device speech-to-text (faster-whisper; `tiny` / `base` / `small`).
- Streaming text-to-speech (Piper, `en_GB-alan-medium` by default), with
  barge-in — speaking over Jarvis cancels the response.
- Stop and cancel words: "stop", "shut up", "never mind", "be quiet",
  "that's enough".
- Local LLM through Ollama with streaming, tool calling, and windowed
  conversation history.
- The Jarvis persona: calm, dryly witty, British, addresses the user as "sir".

**Modes and state**

- Active / Muted / Sleeping modes, reachable from the tray, hotkeys, and voice.
- Sleep mode evicts the model from VRAM (`keep_alive: 0`) so the GPU is freed.
- Idle / Listening / Thinking / Speaking conversational states, which drive the
  overlay orb.

**Apps and system control**

- Open any app by name, including Microsoft Store apps via `shell:AppsFolder`
  URIs; close an app by name; launch a Steam game by title.
- Open the whole configured workspace in one command.
- Open a URL or website.
- Take a screenshot (saved to `~\Pictures\Jarvis_Screenshots` and copied to the
  clipboard).
- See the screen — multimodal vision through a local Ollama model
  (`llava:7b` by default), triggered by "what's on my screen", "look at my
  screen", "describe my screen". Model name, max image dimension, max tokens,
  and temperature are configurable under Settings → Models → Vision.
- Lock the screen; volume up / down / mute / unmute.
- Type dictated text into the focused window.
- Report CPU and memory usage aloud.
- "What time is it", answered locally with no LLM round trip.

**Web, search, and research**

- Google search by voice.
- Quick research panel — DuckDuckGo snippets summarised by the local Ollama
  model, with "read more", "continue", and "copy that" follow-ups.
- Deep research: a planner/worker two-model split with sub-question
  decomposition, diversified queries, full-page fetch, gap-fill, a table of
  contents, an executive overview, numbered citations, and a markdown report
  saved to `%APPDATA%\Jarvis\deep_research\<session>\report.md` after every step.
- Pause, resume, close, and delete deep-research sessions (one or all).
- Optional Ultra mode for deep research — opt-in Brave Search, Groq, and Jina
  keys, resolved from environment variables first.
- Weather for a saved location or any place by name (Open-Meteo, no key).

**Notes**

- Voice-captured markdown notes stored as individual `.md` files under
  `%APPDATA%\Jarvis\notes\`: take, append to, read aloud, and delete by title.
- Notes panel with inline markdown editing, auto-save, and a folder shortcut.

**Music and media**

- "Play [song / artist / genre]" opens the top YouTube result with autoplay;
  open YouTube directly.

**Clipboard**

- Read or clear the clipboard.
- Clipboard history panel — capped, deduplicated, persisted to
  `%APPDATA%\Jarvis\clipboard_history.json`, and skips obvious credential
  payloads. Items can be pinned so they survive a clear, and re-loaded onto the
  live clipboard by voice ("paste my last copy", "paste item 3").

**User interface**

- System tray icon with Active / Muted / Sleeping variants and a full menu.
- Floating overlay orb: listening pulse, thinking shimmer, speaking waveform.
- Live dashboard HUD — mode, state, uptime, CPU and RAM meters, mic level,
  models in use, foreground app, and note / research counts.
- Help panel ("what can you do") with live search and plain-English examples,
  backed by a single capability catalog shared with the Settings → Help tab.
- Command palette (`Ctrl+Shift+P`, rebindable) — a keyboard launcher that also
  accepts free-form phrases through the same intent router as speech.
- Live log viewer with level filter, search, and colour-coded lines.
- Settings window: General, Voice, Models, Tools, Hotkeys, Help, About.
- First-run tutorial: mic test, wake-word test, and a sample command.

**Extensibility**

- MCP-compatible tool registry; external MCP servers can be added from
  Settings → Tools and have their tools adapted into the local namespace.
- Hybrid intent router — a regex pattern layer for deterministic commands, LLM
  tool calling for everything else, and compound intents ("open Spotify **and**
  tell me the weather").
- Per-tool enable/disable, a configurable workspace app list, and configurable
  hotkeys.

**Operations and packaging**

- Portable Windows zip build via PyInstaller (`build.ps1`).
- Structured logs at `%APPDATA%\Jarvis\logs\jarvis.log`.
- A per-interaction correlation ID linking wake → STT → LLM → TTS in the logs.
- Versioned config schema (v20) with a numbered migration per bump, so existing
  installs upgrade in place.
- No analytics, no telemetry, no crash reporting.

### Security

- Documented privacy model: every network destination, what triggers it, and
  what stays on the machine.
- Documented prompt-injection threat model for deep research, with
  "ignore embedded instructions" guidance pinned into the research prompts.
- API keys resolve from environment variables before `config.json`, are masked
  in the Settings UI, are never logged, and are sent only over HTTPS to the
  named provider.

[Unreleased]: https://github.com/ndunl075/Jarvis/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/ndunl075/Jarvis/releases/tag/v0.0.1
