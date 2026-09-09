# Jarvis

A fully local, no-subscription AI voice assistant for Windows. Wake word, speech-to-text, Ollama LLM, and tools — all on your PC.

Technical details: `SPEC.md` · Build notes: `BUILD.md`

---

## What Jarvis can do

Everything below runs on your PC. Anything that reaches the internet is called out explicitly under [Security & privacy model](#security--privacy-model).

**Core voice loop**
- Wake-word activation ("Hey Jarvis") with configurable sensitivity
- Voice activity detection + endpointing (silero-vad)
- Speech-to-text on-device (faster-whisper, `tiny` / `base` / `small`)
- Streaming text-to-speech (Piper, `en_GB-alan-medium` by default)
- Barge-in: interrupt Jarvis mid-response by speaking
- Push-to-talk hotkey (alternative to wake word)
- Local LLM via Ollama with streaming, tool calling, and conversation history
- Jarvis persona (calm, dryly witty, British, addresses you as "sir")
- Stop / cancel words: "stop", "shut up", "never mind", "be quiet", "that's enough"

**Modes & state**
- Active / Muted / Sleeping modes (tray + hotkeys + voice)
- Sleep mode evicts the LLM from VRAM so your GPU is free
- Idle / Listening / Thinking / Speaking states drive the overlay orb

**Apps & system control**
- Open any app by name (Start menu + Microsoft Store via `shell:AppsFolder` URIs)
- Close an app by name
- Launch a Steam game by title
- **Open your whole workspace** — every app you configure, in one command
- Open a URL or website
- Take a screenshot (saved to `~\Pictures\Jarvis_Screenshots` and copied to clipboard)
- **See your screen** — multimodal vision via local Ollama: "what's on my screen", "look at my screen", "describe my screen". Requires pulling a vision model (`ollama pull llava:7b`); pick it in Settings → Models → Vision.
- Lock the screen
- Volume up / down / mute / unmute
- Type dictated text into the currently focused window
- Report CPU and memory usage out loud
- "What time is it" answered locally with zero LLM latency
- First-run tutorial (mic test, wake-word test, sample command)

**Web, search & research**
- Google search by voice ("search…", "google…", "search up…", "search for…")
- **Quick research panel** — DuckDuckGo snippets summarized by your local Ollama model
- "Read more" / "continue" / "copy that" follow-ups inside the research panel
- **Deep research** (Comet-style, fully local) — planner + worker two-model split, sub-question decomposition, diverse queries, full-page fetch, gap-fill, TOC, executive overview, numbered citations, markdown report saved per session
- Pause / resume / close / delete deep research sessions (one or all)
- Optional **Ultra mode** for deep research (opt-in Brave + Groq + Jina keys)
- **Weather** for your saved location or any place by name (Open-Meteo, no key)

**Notes (voice-captured markdown)**
- "Take a note about…" / "jot this down…" / "write this down…" — saves a new `.md` file
- "Add this to my meeting note: …" appends to the matching note
- "Read my groceries note" reads it aloud
- "Delete the groceries note" or "delete this note"
- Notes panel with inline markdown editing, auto-save, and folder shortcut

**Music & media**
- "Play [song / artist / genre]" — top YouTube result with autoplay
- Open YouTube directly

**Clipboard**
- Read or clear the current clipboard contents
- **Clipboard history** panel — capped, deduplicated, persisted, credential-payload aware
- "Paste my last copy" / "paste item 3" reloads the chosen entry onto the clipboard
- Pin items so they survive `clear history`

**UI surfaces**
- System tray icon with Active / Muted / Sleeping variants and full menu
- Floating overlay orb (listening pulse, thinking shimmer, speaking waveform)
- **Live dashboard HUD** — mode, state, uptime, CPU/RAM meters, mic level, models in use, foreground app, note + research counts
- **Help panel** ("what can you do") with live search and plain-English examples
- **Command palette** (`Ctrl+Shift+P`, rebindable) — keyboard launcher for any command, also accepts free-form phrases
- **Live log viewer** — tails `jarvis.log`, level filter, search, color-coded, opens file
- Settings window: General · Voice · Models · Tools · Hotkeys · Help · About

**Extensibility**
- MCP-compatible tool registry; add external MCP servers from Settings → Tools
- Hybrid intent router: regex pattern layer for snap commands, LLM tool-calling for everything else
- Compound intents ("open Spotify **and** tell me the weather")
- Per-tool enable/disable, configurable workspace app list, configurable hotkeys

**Ops & packaging**
- Portable Windows zip build via PyInstaller
- Structured rotating logs at `%APPDATA%\Jarvis\logs\jarvis.log` (10 MB × 5)
- Per-interaction correlation ID linking wake → STT → LLM → TTS in logs
- Versioned config schema with migration hook
- No analytics, no telemetry, no crash reporting

---

## Download (Windows)

**[Download Jarvis for Windows (zip)](https://github.com/ndunl075/Jarvis/releases/latest/download/Jarvis-0.0.1-windows-x64.zip)**

That link works after you [publish a GitHub Release](#publishing-a-download-on-github) with a zip asset named exactly `Jarvis-0.0.1-windows-x64.zip`. Until the first release exists, use the [Releases](https://github.com/ndunl075/Jarvis/releases) page instead.

### Quick start

1. **Install [Ollama](https://ollama.com/download)** (the LLM is not bundled).
2. Pull the default model (or your choice in Settings later):
   ```text
   ollama pull qwen2.5:7b-instruct
   ```
3. **Download and extract the full zip** to a folder, e.g. `C:\Program Files\Jarvis`.
   - You must keep `Jarvis.exe`, `models\`, and `voices\` in the same folder.
   - Do not run a lone copied `.exe` without the rest of the folder.
4. Run **`Jarvis.exe`** from that folder.
5. Say **“Hey Jarvis”**. Use headphones for best wake-word / barge-in behavior.

**SmartScreen:** unsigned builds may show a warning → **More info → Run anyway**.

**Logs:** `%APPDATA%\Jarvis\logs\jarvis.log`

---

## Recommended PC specs

Jarvis runs on CPU by default. A GPU helps Ollama respond faster but is not required.

| | Minimum | Recommended |
|---|---------|-------------|
| **OS** | Windows 10/11 64-bit | Windows 11 64-bit |
| **CPU** | 4-core x64 (2018+) | 6+ cores (Intel 10th gen / Ryzen 3000 or newer) |
| **RAM** | 8 GB | **16 GB** (LLM + Whisper + desktop apps) |
| **Disk** | ~3 GB free | **~10 GB** (app zip ~1 GB + Ollama models ~4–8 GB) |
| **GPU** | None (CPU inference) | NVIDIA/AMD with **8+ GB VRAM** for faster Ollama |
| **Audio** | Microphone + speakers/headphones | Headset mic (reduces echo with wake word) |

### What uses resources

| Component | Bundled in zip? | Notes |
|-----------|-----------------|-------|
| Wake word (openWakeWord) | Yes | Light CPU |
| Speech-to-text (Whisper) | Yes (`base.en` in build) | CPU; switch to `tiny.en` in Settings for less CPU |
| Voice (Piper TTS) | Yes | Light CPU |
| **LLM (Ollama)** | **No — you install** | Default `qwen2.5:7b-instruct` ~4–5 GB; first reply can take 15–45 s on CPU |

**Faster on weak PCs:** `ollama pull qwen2.5:3b-instruct` (or another small model) and set it under **Settings → Models**.

**Slower STT, less CPU:** Settings → Models → Whisper `tiny.en`.

---

## Publishing a download on GitHub

This is how you get the **“click link → browser downloads zip”** behavior.

### One-time: build the zip

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
.\build.ps1
```

Upload this file (name must match the version in `pyproject.toml`):

```text
dist\Jarvis-0.0.1-windows-x64.zip
```

### Create the release (GitHub website)

1. Open **https://github.com/ndunl075/Jarvis/releases**
2. Click **Draft a new release** (or **Create a new release**).
3. **Choose a tag**, e.g. `v0.0.1` (create the tag if it does not exist).
4. **Release title:** e.g. `Jarvis 0.0.1 (Windows)`
5. **Description:** short notes (what’s included, need Ollama, extract full folder).
6. Under **Attach binaries**, drag **`Jarvis-0.0.1-windows-x64.zip`** onto the page.
7. Click **Publish release**.

### Direct download link (for README / sharing)

After publish, this URL downloads the zip (replace version if you bump `pyproject.toml`):

```text
https://github.com/ndunl075/Jarvis/releases/latest/download/Jarvis-0.0.1-windows-x64.zip
```

- `releases/latest` always points at the newest release.
- The filename after `/download/` must **exactly** match the uploaded asset name.

Optional: pin a specific version:

```text
https://github.com/ndunl075/Jarvis/releases/download/v0.0.1/Jarvis-0.0.1-windows-x64.zip
```

### CLI (optional)

With [GitHub CLI](https://cli.github.com/) installed and authenticated:

```powershell
gh release create v0.0.1 dist\Jarvis-0.0.1-windows-x64.zip `
  --title "Jarvis 0.0.1 (Windows)" `
  --notes "Windows x64 portable zip. Requires Ollama. Extract full folder before running Jarvis.exe."
```

---

## Build from source

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
.\build.ps1
```

Outputs:

- `dist\Jarvis\Jarvis.exe` — portable folder
- `dist\Jarvis-0.0.1-windows-x64.zip` — upload to GitHub Releases

Ollama is **not** bundled.

## Development

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m jarvis
pytest
ruff check .
```

### Workspace

Say **open my workspace**, **launch workspace**, or **jarvis open my workspace** to start every app in your workspace at once. Configure the list under **Settings → General → Workspace** (installed app name, `.exe` path, or `shell:AppsFolder\…` URI for Store apps). Defaults are Cursor and Apple Music; change them to match your machine.

### Web research (optional)

Voice command **“research …”** or **“look up …”** opens a side panel: Jarvis fetches web snippets (DuckDuckGo) and summarizes them with your **local Ollama** model (same one as Settings → Models). No Anthropic or other cloud LLM API key. You need **Ollama running** and an internet connection for search.

### Deep research (Comet-style, fully local)

Voice **“deep research …”** or **“do deep research on …”** runs a multi-stage investigation with a **two-model split**:

- **Planner** (smart model, e.g. `qwen2.5:7b-instruct`) decomposes the topic into sub-questions, expands each into 2–3 diverse search queries, reviews per-section notes to spot gaps and queues follow-up searches, and writes the final executive overview.
- **Worker** (fast model, e.g. `qwen2.5:3b-instruct`) extracts per-source bullets as **cited claims** like `Fact about X [1] [3]`.

Each sub-question fetches the **full page text** of the top diversified sources (one per domain). The final report includes a TOC, executive overview, findings sections with citations, and a numbered **References** list — saved to `%APPDATA%\Jarvis\deep_research\<session>\report.md` after every step (so a crash never loses progress).

- **Pause:** say **“pause deep research”** or click **PAUSE** (finishes the current step, then saves).
- **Resume:** say **“resume deep research”** or pick a paused session and click **RESUME**.
- **Folder:** opens the reports directory in Explorer.

Tune everything in **Settings → Tools → Deep research** (planner model, worker model, depth, breadth, max page chars, full-page fetch toggle, gap-fill toggle). Blank model fields fall back to your main Models tab model.

### Notes (voice-captured markdown)

Markdown scratchpad with hands-free capture. Each note lives in `%APPDATA%\Jarvis\notes\<id>.md` so you can edit them in any editor.

- **Take a note:** say **“take a note about the meeting being moved to Friday”**, **“jot down: buy milk”**, or **“write this down: project Phoenix kicks off Monday”**. A new markdown file is saved with a derived title and the notes panel slides in.
- **Open / close the panel:** **“open my notes”**, **“show notes”**, **“close notes”**.
- **Append:** **“add this to my meeting note: agenda finalized”** appends to the most recent note whose title contains *meeting* (case-insensitive). Without a title it appends to whichever note is currently selected in the panel.
- **Read aloud:** **“read this note”** (the open one) or **“read my groceries note”**.
- **Delete:** **“delete this note”** or **“delete the groceries note”**. The panel also has a `DELETE` menu for one-or-all removal with a confirmation.
- **Edit inline:** click `EDIT` in the panel header to write in plain markdown; auto-saves on blur or after ~800 ms of inactivity.
- **Folder button** opens the notes folder in Explorer.

### Live dashboard

Voice **“show dashboard”**, **“open dashboard”**, or **“open my system stats”** slides in a HUD with:

- Current **mode** (active / muted / sleeping) and **conversational state** (listening / thinking / speaking)
- **Uptime** since launch, plus a live clock
- **CPU** and **RAM** meters (color-shifts green → amber → red)
- Live **mic level** EMA — easy way to confirm your input device is hot
- **Main model**, **research planner**, and **research worker** in use
- **Foreground app** title (Windows-only)
- Counters for **saved notes** and **deep research sessions** (paused vs total)

Updates every ~1.5 s while open, idle when closed. Say **“close dashboard”** to hide. Also reachable from the tray icon → **Show dashboard**.

### Help — “What can I say?”

If anyone in your house picks up Jarvis and isn't sure where to start, there are three ways in:

- Voice: **“what can you do”**, **“show help”**, or **“what can I say”** opens a slide-in panel with a live search box and plain-English example phrases grouped by topic (Apps, Music, Web, Research, Notes, System, Clipboard…).
- Tray icon → **What can I say?**
- **Settings → Help** for the same list inside the settings window — handy when configuring Jarvis for the first time.

Every capability shows what it does in one sentence plus 2–3 example phrases to say out loud. No jargon.

### Command palette

Keyboard launcher for everything Jarvis can do — useful when speaking out loud isn't appropriate (calls, libraries, anywhere quiet). Press **`Ctrl+Shift+P`** (rebindable under Settings → Hotkeys → Command palette), type a few letters, hit Enter. Up/Down to navigate, Esc to dismiss. The list filters live against every example phrase from the Help catalog; you can also type any free-form phrase and Jarvis will run it as if you'd spoken it. Submissions go through the same intent router as STT, so patterns and LLM fallback both work.

### Clipboard history

A capped, pinnable history of recent text clipboard items. Always-available via the tray icon → **Clipboard history**, or by voice:

- **“show clipboard history”** / **“what have I copied”** opens the panel.
- **“paste my last copy”** or **“paste item 3”** loads the chosen item back onto the live clipboard — the next Ctrl+V uses it.
- **“clear my clipboard history”** (pinned items survive by default).

The panel polls the system clipboard every ~700 ms while open, deduplicates consecutive captures, skips obvious credential payloads (anything containing `password=`, `secret=`, `api_key=`), pins items so they don't get evicted, and persists everything to `%APPDATA%\Jarvis\clipboard_history.json`. Pin / delete / copy from the right-click menu; double-click any item to copy it back.

### Live log viewer

Real-time tail of `%APPDATA%\Jarvis\logs\jarvis.log` with a level dropdown (ALL / DEBUG / INFO / WARNING / ERROR) and a substring search. Lines are color-coded by severity. Refreshes every second while open, handles log rotation, and never grows unbounded (keeps the last few thousand lines in memory).

- Voice: **“show logs”**, **“show errors”**, **“close logs”**.
- Tray → **Show logs**.
- **OPEN FILE** button reveals the log in your default text editor for copy-into-bug-reports use.

### First-run tutorial

The very first time Jarvis launches it pops a friendly 3-step welcome:

1. **Welcome + mic test** — live waveform proves your microphone is hot.
2. **Wake word test** — say *“Hey Jarvis”*; the step checks off the first time the wake-word detector fires.
3. **Try a command** — buttons that open the Help panel and the command palette so you have somewhere to go next.

Skip/Finish at any point. The walkthrough remembers your decision in `general.first_run_completed` and won't pop up again. You can re-open it any time from the tray icon → **Show tutorial**. Existing installs upgrading from an earlier schema version are auto-marked as completed so the tutorial doesn't surprise people who've been using Jarvis already.

---

## Security & privacy model

Jarvis is designed to run locally; understand the boundaries before you trust it with sensitive workflows.

### What stays local

- Wake word, VAD, STT (Whisper), TTS (Piper), and the main LLM (Ollama) all run on your PC. Voice audio is never sent to a third party.
- Notes (`%APPDATA%\Jarvis\notes\`), deep-research reports (`%APPDATA%\Jarvis\deep_research\`), screenshots (`~\Pictures\Jarvis_Screenshots\`), and logs (`%APPDATA%\Jarvis\logs\`) are local-only.
- Configuration lives in `%APPDATA%\Jarvis\config.json` on your machine. Any API keys and MCP auth tokens you enter are encrypted there with Windows DPAPI, so they are readable only by your Windows user account; the fields are masked in the Settings UI too. The rest of the file stays plain JSON you can hand-edit.

### What leaves your PC (and only when you opt in)

| Feature | Destination | When |
|---|---|---|
| `research` / `deep_research` web search | `html.duckduckgo.com` | Whenever you trigger research |
| Deep research page fetches | The URLs returned by search | Whenever you trigger research |
| Deep research Ultra search | `api.search.brave.com` | Only if you provide a Brave key (env `JARVIS_BRAVE_API_KEY` or Settings → Tools) |
| Deep research Ultra planner | `api.groq.com` | Only if you provide a Groq key (env `JARVIS_GROQ_API_KEY` or Settings → Tools) |
| Deep research Ultra page extraction | `r.jina.ai` | Only when Ultra mode is enabled |
| Weather | `api.open-meteo.com` (no key) and `ipapi.co` (one-time IP geolocation when no coords set) | Whenever you ask for weather |
| First-run STT model fetch | `huggingface.co` (anonymous, model weights) | First launch only |

No analytics, telemetry, or crash reporting is sent anywhere.

### Threat model: prompt injection in fetched web pages

The deep-research feature feeds the contents of web pages into your local LLM. A malicious page could contain text crafted to talk the model into ignoring its instructions or producing misleading citations. Jarvis pins the deep-research prompts with explicit "ignore embedded instructions" guidance, but you should still treat deep-research output as you would treat any AI-summarized article — verify important facts against the linked sources.

The voice-control tools (`open_app`, `open_url`, `launch_steam_game`, etc.) can be invoked by the LLM. If you ingest untrusted text via deep research, treat the LLM's tool decisions with the same skepticism you would treat a stranger's commands — keep the assistant in a context where the worst case (an extra tab opening, an app starting) is acceptable.

### API keys

- Resolution order: environment variable first (`JARVIS_BRAVE_API_KEY`, `JARVIS_GROQ_API_KEY`), then the value in `config.json`, then empty (graceful fallback to the free pipeline). Env vars never get written to disk by Jarvis.
- Keys stored in `config.json` are encrypted at rest with Windows DPAPI and appear as `"dpapi:<base64>"`. DPAPI ties the ciphertext to your Windows user account, so copying `config.json` to another machine or another account leaves the keys unreadable: Jarvis logs a warning, treats them as empty, and you re-enter them in Settings. Everything else in the file is untouched.
- On non-Windows hosts (contributor dev machines — Jarvis ships for Windows) there is no DPAPI, so keys fall back to plaintext with a startup warning. Prefer env vars there.
- Keys are sent only over HTTPS to the named providers. They are never logged or printed.
- Revoking a key in your provider dashboard is sufficient; nothing in Jarvis caches it server-side.

### Reporting a security issue

If you find a vulnerability, please open a private security advisory on GitHub rather than filing a public issue.
