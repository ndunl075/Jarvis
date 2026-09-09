"""Jarvis configuration: pydantic v2 schema, JSON persistence, migration hook.

Design notes (kept here so future readers see the rationale without digging history):

- Path is injected (load_config(path) / save_config(cfg, path)). The default lives
  in `default_config_path()`; OS-specific helpers will move to `platform/` once
  that layer exists. Until then, the one branch here is contained.
- `extra="forbid"` rejects unknown keys to catch typos in hand-edited JSON.
  Migrations must explicitly drop removed fields.
- `validate_assignment=True` makes bad UI writes fail at the assignment site
  rather than at next save.
- Save is atomic: write tmp, replace.
- Migrations are a chain of v(N) -> v(N+1) functions, applied in sequence.
  Today there is only one schema version, so MIGRATIONS is empty; the comment
  in MIGRATIONS is the template for adding the first real migration.
- This module deliberately does not depend on `events.py`. ConfigChanged events
  are the responsibility of a future `ConfigManager` wrapper (Phase 5-ish)
  which will own (JarvisConfig, EventBus), expose set(path, value), compute
  the diff, and emit. Decision: chokepoint emission rather than emit-from-UI
  so any future caller (first-launch flow, migration completion, test
  fixtures) gets the event for free.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from os import environ
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CURRENT_SCHEMA_VERSION = 21

# Prompts from prior schema versions — used by migrations to detect and
# replace the old default without overwriting user-customised prompts.
_SYSTEM_PROMPT_V3 = (
    "You are Jarvis, an AI assistant inspired by Iron Man. Calm, dryly "
    "witty, British, address the user as 'sir'. Keep responses concise "
    "— one or two sentences usually.\n"
    "\n"
    "CRITICAL TOOL USAGE RULES:\n"
    "- Default behavior is to answer in words, not call tools.\n"
    "- Only call a tool when the user gives a clear, explicit command "
    "matching that exact tool's purpose.\n"
    "- For greetings, factual questions, math, opinions, or unclear "
    "input: answer in words. Do not call any tool.\n"
    "- If you can't understand what the user said, ask for clarification. "
    "Do not guess and call a tool.\n"
    "- Never call tools speculatively or 'just in case'."
)

# Prompt shipped with schema v6 — over-restricted conversational tasks
# (Jarvis would refuse counting/math with "I'm designed to assist with
# computer tasks"). Replaced by v7 default below.
_SYSTEM_PROMPT_V6 = (
    "You are Jarvis, an AI assistant inspired by Iron Man. Calm, dryly "
    "witty, British, address the user as 'sir'. Keep responses concise "
    "— one or two sentences usually.\n"
    "\n"
    "YOUR JOB:\n"
    "- Answer questions, have conversations, perform mental tasks "
    "(math, counting, reasoning, jokes, advice).\n"
    "- For factual questions, just answer. For mental tasks like counting "
    "or arithmetic, just do them.\n"
    "- Match the user's energy and intent. Be useful.\n"
    "\n"
    "TOOL USAGE:\n"
    "- You have tools for specific actions (screenshots, opening apps, "
    "system stats, etc).\n"
    "- ONLY call a tool when the user explicitly asks for that specific "
    "action.\n"
    "- For everything else — questions, conversations, math, counting, "
    "reasoning — answer in words. Do not call any tool.\n"
    "- If the user's request is unclear, ask for clarification. "
    "Don't refuse, don't guess and call a tool."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, an AI assistant inspired by Iron Man. Calm, dryly "
    "witty, British, address the user as 'sir'. Keep responses concise "
    "— one or two sentences usually.\n"
    "\n"
    "YOUR JOB:\n"
    "- Answer questions, have conversations, do mental tasks (math, counting, "
    "reasoning, jokes, opinions, advice, storytelling).\n"
    "- When the user asks you to count, do math, tell a story, or have a "
    "conversation: just do it directly in your response.\n"
    "- When the user asks for a specific computer action (open an app, take a "
    "screenshot, check CPU stats, lock the screen, etc.): call the tool — do not "
    "only say you did it in text.\n"
    "\n"
    "TOOL USAGE:\n"
    "- Only call a tool when the user explicitly asks for that specific "
    "computer action.\n"
    "- For everything else (questions, conversations, counting, math, jokes, "
    "reasoning, advice): answer in words. Do not call any tool.\n"
    "- Read the full utterance and use context to pick the right tool. "
    "Ambiguous 'play X' is common — decide from clues:\n"
    "  * Music (song, artist, album, genre, 'some jazz', 'that track'): "
    "play_youtube_music with query = song/artist terms only; it finds and "
    "opens the video with autoplay.\n"
    "  * Video game (game title, Steam, 'boot up', 'launch the game'): "
    "launch_steam_game.\n"
    "  * Desktop program (app name, 'open', browser, editor): open_app.\n"
    "  * Whole workspace / dev setup ('open my workspace', 'launch workspace'): "
    "launch_workspace.\n"
    "  * In-depth report with pause/resume ('deep research [topic]'): "
    "deep_research. Quick summary only: research.\n"
    "  * Stronger free-tier pipeline ('enable deep research ultra'): "
    "enable_deep_research_ultra; disable with disable_deep_research_ultra.\n"
    "- Never guess between Steam and YouTube from title overlap alone; use "
    "what the user actually asked for.\n"
    "- If the user's request is unclear, ask for clarification. Don't refuse.\n"
    "- Tools require no admin permissions. Don't hallucinate restrictions."
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GeneralConfig(_Base):
    start_with_windows: bool = False
    minimize_to_tray: bool = True
    show_overlay: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Speak a short confirmation phrase ("Going to sleep, sir.") before
    # ACTIVE/MUTED -> SLEEPING transitions. The coordinator awaits the
    # phrase before unloading TTS; a Wake request during the phrase
    # cancels the transition. Set False for silent sleep.
    sleep_confirmation: bool = True
    # Flipped to True once the first-run onboarding walkthrough is
    # dismissed or finished. The Qt main thread auto-shows the
    # OnboardingPanel when this is False.
    first_run_completed: bool = False


class AudioConfig(_Base):
    # Defaults match the current dev machine (TONOR mic, Creative Pebble Pro
    # speaker). Production users override via Settings → Voice. The substring-
    # matching in audio/devices.py and tts.py falls back to system default with
    # a warning when the named device is not found, so these strings are safe on
    # other hardware.
    # "Speakers (Creative Pebble Pro)" is preferred over "Pebble Pro": Windows
    # re-enumeration produces two problematic extra entries for the same
    # physical speaker — a phantom "Speakers (2- Creative Pebble Pro)" (sorts
    # before the real one) and a Bluetooth HFP entry
    # "Headset (... (Creative Pebble Pro))" (8 kHz mono, wrong for TTS).
    # The "Speakers (" prefix uniquely matches device 43
    # "Speakers (Creative Pebble Pro)" and excludes both noise entries.
    input_device: str | None = "TONOR"
    output_device: str | None = "Speakers (Creative Pebble Pro)"
    prefer_respeaker: bool = True


class WakeWordConfig(_Base):
    enabled: bool = True
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    model: str = "hey_jarvis"


class STTConfig(_Base):
    # tiny.en is the default: 2-3x faster than base.en on CPU with modest
    # accuracy loss for English speech. Users who notice transcription errors
    # or who have a GPU can switch to base.en (or base/small) in
    # Settings → Models. The .en suffix restricts the model to English only,
    # which shaves another ~10% vs the multilingual variant.
    model_size: Literal["tiny", "tiny.en", "base", "base.en", "small", "small.en"] = "tiny.en"
    language: str = "en"
    compute_type: Literal["int8", "float16", "float32"] = "int8"


class TTSConfig(_Base):
    voice: str = "en_GB-alan-medium"
    speed: float = Field(default=1.0, gt=0.0, le=4.0)
    volume: float = Field(default=1.0, ge=0.0, le=1.0)


class LLMConfig(_Base):
    model: str = "qwen2.5:7b-instruct"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Ollama keep_alive: how long the daemon keeps the model resident in
    # VRAM between requests. 1800 s = 30 min handles realistic
    # conversational gaps without re-paying cold-load latency. SLEEPING
    # mode explicitly evicts via keep_alive=0; this is the "active use"
    # default, not a memory-pressure ceiling.
    keep_alive_seconds: int = 1800
    # Conversation history length (number of turn messages retained,
    # distinct from max_tokens which caps per-response output). The
    # system prompt is not counted -- it's standing context, not a turn.
    max_turns: int = Field(default=10, gt=0)
    # Idle time after which the next user message starts a fresh
    # conversation (history cleared, system prompt re-applied). Stale
    # context degrades LLM response quality and creates surprising
    # "you said X two hours ago" behavior.
    inactivity_timeout_seconds: int = Field(default=300, gt=0)
    # How long after the last assistant response a new wake-word activation
    # is still considered a follow-up (history preserved). After this window
    # the activation starts a fresh session (history cleared).
    # Trade-offs: 10 s — aggressive follow-ups don't link cleanly;
    # 300 s — cross-task contamination risk returns. 60 s matches typical
    # conversational follow-up tempo without leaking prior task context.
    conversation_continuity_seconds: int = Field(default=60, gt=0)
    # Maximum LLM invocations per user turn when the model calls tools.
    # The router feeds each tool result back as a role:"tool" message and
    # re-invokes the model so it can act on what it got back ("check the
    # weather, and if it's cold open my coat app"). Every extra iteration
    # is another full inference, so this is a latency budget rather than
    # a capability ceiling: 3 covers the two-step requests people
    # actually make while capping the worst case at three inferences on
    # a local 7B. Set to 1 to disable the feedback loop and restore
    # one-shot dispatch — the tool's own output is spoken verbatim,
    # lowest latency, no follow-up reasoning.
    max_tool_iterations: int = Field(default=3, ge=1, le=10)


class ToolsConfig(_Base):
    # Convention: a tool absent from `enabled` is treated as enabled.
    # The registry is the single source of truth for this lookup.
    # type_into_active_window is disabled by default: it simulates keystrokes
    # into the active window and LLM misfires can type unintended content into
    # user apps. Enable via Settings → Tools.
    enabled: dict[str, bool] = Field(
        default_factory=lambda: {"type_into_active_window": False}
    )


class HotkeysConfig(_Base):
    mute: str = "ctrl+shift+m"
    push_to_talk: str | None = None
    open_settings: str = "ctrl+shift+,"
    # Opens the command palette — keyboard-driven fuzzy launcher for
    # every Jarvis capability. Set to "" to disable the binding.
    command_palette: str = "ctrl+shift+p"


# Default Trayce MCP endpoint. The port is overridden at connect time by
# %LOCALAPPDATA%\Trayce\port.txt when present (see tools/mcp_client.py).
DEFAULT_TRAYCE_URL = "http://127.0.0.1:52945/mcp"


class MCPServerConfig(_Base):
    # Defaults describe a Trayce connection so a bare MCPServerConfig() is
    # the Trayce entry; the fresh-config default list ships it disabled so
    # users opt in via Settings → Tools.
    name: str = "trayce"
    url: str = DEFAULT_TRAYCE_URL
    enabled: bool = True
    # When True, the auth token is read from
    # %LOCALAPPDATA%\Trayce\http-auth-token.txt at connect time (the
    # Trayce convention). When False, `auth_token` below is used verbatim.
    auth_token_from_file: bool = True
    auth_token: str | None = None


def _default_mcp_servers() -> list[MCPServerConfig]:
    """Fresh installs ship a disabled Trayce entry so the Settings → Tools
    toggle is discoverable without the user hand-authoring JSON."""
    return [MCPServerConfig(name="trayce", url=DEFAULT_TRAYCE_URL, enabled=False)]


class LifecycleConfig(_Base):
    # Automatically sleep after this many minutes of total inactivity
    # (no wake-word, no STT, no LLM, no TTS). Opt-in: False by default.
    auto_sleep_enabled: bool = False
    idle_timeout_minutes: int = Field(default=30, gt=0, le=1440)
    # Reserved for future rules — no-op in Task 2.
    auto_sleep_on_low_battery: bool = False
    auto_sleep_on_user_idle: bool = False


class DebugConfig(_Base):
    # Log wake-word score every ~10 frames while Jarvis is SPEAKING.
    # Diagnostic confirmed working (wake-word interrupt verified 2026-05-19).
    # AEC deferred — see BUILD.md Phase 6 Task 4 and pipeline.py.
    log_wake_during_speaking: bool = False


class UIConfig(_Base):
    # Width of the research panel in pixels. Persisted so the user's last
    # resize survives restarts. Clamped to [280, 700] by the panel itself.
    research_panel_width: int = Field(default=420, ge=280, le=700)


class ResearchConfig(_Base):
    # Quick research uses the main llm.model. Deep research uses a smart
    # planner + a fast worker so the planner reasons about structure and
    # the worker (3B-class) does cheap bullet extraction over many pages.
    #
    # Empty model strings fall back to the main llm.model so a fresh
    # install works without forcing the user to pull a second model.
    api_key: str | None = None  # legacy; ignored. Kept for migration compat.
    planner_model: str = ""
    worker_model: str = ""
    depth: int = Field(default=5, ge=2, le=10)
    breadth: int = Field(default=5, ge=2, le=10)
    fetch_pages: bool = True
    max_page_chars: int = Field(default=6000, ge=1000, le=20000)
    enable_gap_fill: bool = True
    # Ultra: Brave search + Jina Reader + Groq 70B planner + 3x gap-fill (free APIs).
    ultra_enabled: bool = False
    brave_api_key: str = ""
    groq_api_key: str = ""
    ultra_planner_model: str = "groq/llama-3.3-70b-versatile"
    ultra_gap_fill_iterations: int = Field(default=3, ge=1, le=5)


def _default_cursor_path() -> str:
    return str(
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "cursor"
        / "Cursor.exe"
    )


def _default_workspace_apps() -> list[WorkspaceAppEntry]:
    """Dev-machine defaults; users customize in Settings → General."""
    return [
        WorkspaceAppEntry(
            label="Cursor",
            kind="executable",
            target=_default_cursor_path(),
        ),
        WorkspaceAppEntry(
            label="Apple Music",
            kind="shell",
            target=(
                r"shell:AppsFolder\AppleInc.AppleMusicWin_nzyj5cx40ttqa!AppleMusic"
            ),
        ),
    ]


class WorkspaceAppEntry(_Base):
    """One app to launch when the user says 'open my workspace'."""

    label: str = Field(min_length=1, description="Display name, e.g. Cursor")
    kind: Literal["installed_app", "executable", "shell"] = Field(
        description=(
            "installed_app: fuzzy-match Start Menu name; "
            "executable: full path to .exe; "
            "shell: explorer.exe argument (e.g. shell:AppsFolder\\...)"
        ),
    )
    target: str = Field(
        min_length=1,
        description="App name, .exe path, or shell URI depending on kind",
    )


class WorkspaceConfig(_Base):
    apps: list[WorkspaceAppEntry] = Field(default_factory=_default_workspace_apps)


class VisionConfig(_Base):
    """Settings for the `see_screen` tool (multimodal Ollama call).

    The vision model is independent from the conversational `llm.model`
    because text-only models (qwen2.5:7b-instruct, etc.) cannot accept
    images. Users must `ollama pull llava:7b` (or another vision-capable
    model) for the feature to work. Empty `model` disables see_screen
    cleanly (the tool returns a clear "no vision model configured" message
    instead of failing late inside Ollama).
    """

    # llava:7b is a 4-5 GB int4 multimodal model that runs on CPU and
    # ships with the Ollama default registry. moondream / minicpm-v are
    # other reasonable defaults; we don't try to pick one for the user.
    model: str = "llava:7b"
    # Long-edge pixel cap before we ship the image to Ollama. Vision
    # models do their own internal resize; this cap keeps the HTTP body
    # small (1280 px PNG is typically <500 KB) and the prompt-eval time
    # bounded. Lowering to 640 makes weak GPUs much snappier at a small
    # cost to text legibility in the screenshot.
    max_image_dim: int = Field(default=1280, ge=320, le=4096)
    max_tokens: int = Field(default=512, ge=64, le=4096)
    # Lower temperature than chat default (0.7) — vision descriptions
    # benefit from determinism so repeated calls describe the same UI
    # consistently rather than reinventing nouns each time.
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class WeatherConfig(_Base):
    # No hard-coded default coordinates: the weather tool auto-detects the
    # user's approximate location via ipapi.co on first use when both
    # values are None, then persists the result. Users can also enter
    # coordinates manually or click "Detect from IP" under
    # Settings → General. Shipping None defaults avoids embedding any
    # maintainer-specific location in the open-source repo.
    latitude: float | None = None
    longitude: float | None = None
    unit: Literal["celsius", "fahrenheit"] = "fahrenheit"


class JarvisConfig(_Base):
    schema_version: int = CURRENT_SCHEMA_VERSION
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    hotkeys: HotkeysConfig = Field(default_factory=HotkeysConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=_default_mcp_servers)


# --- migration -------------------------------------------------------------

Migration = Callable[[dict], dict]

# Migration[N] takes a dict at schema_version N and returns a dict at N+1.
# Register here when introducing a new schema version. Template:
#
#     def _migrate_vN_to_vN1(data: dict) -> dict:
#         data["some_new_section"] = {"field": "default"}
#         data.pop("removed_field", None)
#         data["schema_version"] = N + 1
#         return data
#
#     MIGRATIONS[N] = _migrate_vN_to_vN1


def _migrate_v1_to_v2(data: dict) -> dict:
    """Backfill audio device defaults introduced in v2.

    v1 configs stored None for both devices (system default). v2 uses
    TONOR/Pebble Pro as dev-machine defaults. Only fills in None values
    so explicitly-set devices (including null → some-other-device edits)
    are NOT overwritten.
    """
    audio = data.setdefault("audio", {})
    if audio.get("input_device") is None:
        audio["input_device"] = "TONOR"
    if audio.get("output_device") is None:
        audio["output_device"] = "Pebble Pro"
    data["schema_version"] = 2
    return data


def _migrate_v2_to_v3(data: dict) -> dict:
    """Schema v3: default STT model changed from 'base' to 'tiny.en'.

    This migration is intentionally a no-op on existing field values: any
    model_size already in the config (including 'base', 'base.en', 'small',
    etc.) was an explicit user choice and must not be overwritten. Only
    fresh installs (no config file on disk) receive the new 'tiny.en'
    default via STTConfig's field default.
    """
    data["schema_version"] = 3
    return data


def _migrate_v3_to_v4(data: dict) -> dict:
    """Schema v4: replace the over-restrictive v3 system prompt.

    Only replaces the prompt when it exactly matches the v3 default so
    that user-customised prompts are left untouched.
    """
    llm = data.setdefault("llm", {})
    if llm.get("system_prompt") == _SYSTEM_PROMPT_V3:
        llm["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    data["schema_version"] = 4
    return data


def _migrate_v4_to_v5(data: dict) -> dict:
    """Schema v5: introduce general.sleep_confirmation.

    Existing v4 configs simply gain the new field at its default (True).
    GeneralConfig's field default would supply it on validate, but writing
    it explicitly here keeps round-tripped JSON stable and surfaces the
    change in diffs.
    """
    general = data.setdefault("general", {})
    general.setdefault("sleep_confirmation", True)
    data["schema_version"] = 5
    return data


def _migrate_v5_to_v6(data: dict) -> dict:
    """Schema v6: introduce lifecycle section (auto-sleep config).

    Existing v5 configs gain the section with all-default values.
    The four fields are written explicitly so the on-disk JSON matches
    what a fresh install would produce and diffs are stable.
    """
    data.setdefault("lifecycle", {
        "auto_sleep_enabled": False,
        "idle_timeout_minutes": 30,
        "auto_sleep_on_low_battery": False,
        "auto_sleep_on_user_idle": False,
    })
    data["schema_version"] = 6
    return data


def _migrate_v6_to_v7(data: dict) -> dict:
    """Schema v7: fix over-restrictive system prompt + add new LLM/debug fields.

    - Replaces the v6 default system prompt only when it matches exactly.
      User-customised prompts are left untouched.
    - Adds llm.conversation_continuity_seconds (default 60) so on-disk
      JSON stays stable after the round-trip without relying solely on
      pydantic's field default.
    - Adds the debug section (all defaults False).
    """
    llm = data.setdefault("llm", {})
    if llm.get("system_prompt") == _SYSTEM_PROMPT_V6:
        llm["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    llm.setdefault("conversation_continuity_seconds", 60)
    data.setdefault("debug", {"log_wake_during_speaking": False})
    data["schema_version"] = 7
    return data


def _migrate_v7_to_v8(data: dict) -> dict:
    """Schema v8: enable wake-during-speaking debug logging + add weather.

    - Flips debug.log_wake_during_speaking to True on all existing configs
      for a temporary AEC diagnosis window (see pipeline.py / DebugConfig).
      Set back to False in the JSON once the scores have been inspected.
    - Adds the weather section with all defaults (None lat/lon = IP auto-detect,
      unit = fahrenheit).
    """
    debug = data.setdefault("debug", {})
    debug["log_wake_during_speaking"] = True
    data.setdefault("weather", {
        "latitude": None,
        "longitude": None,
        "unit": "fahrenheit",
    })
    data["schema_version"] = 8
    return data


def _migrate_v8_to_v9(data: dict) -> dict:
    """Schema v9: fill Reston, VA defaults for weather coords + disable debug flag.

    - Sets weather.latitude / longitude to Reston, VA (38.9586, -77.357) when
      they are None (i.e. the user never explicitly configured a location). The
      v9→v10 migration replaces these with Oakwood, OH on the next load.
    - Sets debug.log_wake_during_speaking = False. The AEC diagnostic window
      is over: wake-word interrupt was confirmed working on tested hardware
      (2026-05-19). See BUILD.md Phase 6 Task 4.
    """
    weather = data.setdefault("weather", {})
    if weather.get("latitude") is None:
        weather["latitude"] = 38.9586
    if weather.get("longitude") is None:
        weather["longitude"] = -77.3570
    debug = data.setdefault("debug", {})
    debug["log_wake_during_speaking"] = False
    data["schema_version"] = 9
    return data


def _migrate_v9_to_v10(data: dict) -> dict:
    """Schema v10: change default weather location from Reston, VA to Oakwood, OH.

    Replaces coordinates only when they exactly match the Reston defaults
    introduced in v9 (38.9586, -77.357). Any other value — including
    user-configured or IP-detected coordinates — is left untouched.
    """
    weather = data.setdefault("weather", {})
    if weather.get("latitude") == 38.9586 and weather.get("longitude") == -77.3570:
        weather["latitude"] = 39.7187
        weather["longitude"] = -84.1736
    data["schema_version"] = 10
    return data


def _migrate_v10_to_v11(data: dict) -> dict:
    """Schema v11: tighten the output-device default set by the v1→v2 migration.

    The v1→v2 migration wrote audio.output_device = "Pebble Pro", which is
    a broad substring that also matches Windows phantom re-enumerations such
    as "(2- Creative Pebble Pro)" and the Bluetooth headset entry. Replace
    with "Speakers (Creative Pebble Pro)" which uniquely identifies the
    wired speaker and is the AudioConfig field default since v2.

    Only rewrites the exact old default; any other value (including
    user-customised strings) is left untouched.
    """
    audio = data.setdefault("audio", {})
    if audio.get("output_device") == "Pebble Pro":
        audio["output_device"] = "Speakers (Creative Pebble Pro)"
    data["schema_version"] = 11
    return data


def _migrate_v11_to_v12(data: dict) -> dict:
    """Schema v12: reshape MCPServerConfig for the Trayce MCP client.

    - Renames each server's `url_or_command` -> `url`.
    - Drops the unused `tools_enabled` map (per-tool enable/disable now
      flows through tools.enabled with server-prefixed names, e.g.
      `trayce_search_context`).
    - Adds `auth_token_from_file` (default True) and `auth_token`
      (default None) to each server.
    - Ensures a disabled Trayce entry exists so the Settings → Tools
      toggle is present for users upgrading from a config that had no
      MCP servers.
    """
    servers = data.get("mcp_servers")
    if not isinstance(servers, list):
        servers = []
    migrated: list[dict] = []
    for srv in servers:
        if not isinstance(srv, dict):
            continue
        url = srv.pop("url_or_command", None) or srv.get("url") or DEFAULT_TRAYCE_URL
        srv.pop("tools_enabled", None)
        srv["url"] = url
        srv.setdefault("auth_token_from_file", True)
        srv.setdefault("auth_token", None)
        migrated.append(srv)
    if not any(s.get("name") == "trayce" for s in migrated):
        migrated.append({
            "name": "trayce",
            "url": DEFAULT_TRAYCE_URL,
            "enabled": False,
            "auth_token_from_file": True,
            "auth_token": None,
        })
    data["mcp_servers"] = migrated
    data["schema_version"] = 12
    return data


def _migrate_v12_to_v13(data: dict) -> dict:
    """Schema v13: add ui section with research_panel_width.

    Existing v12 configs gain the section at the default width (420 px).
    Users who have resized the panel will have their preference persisted
    by the panel's on_width_change callback going forward; this migration
    just ensures the key exists so model_validate succeeds.
    """
    data.setdefault("ui", {"research_panel_width": 420})
    data["schema_version"] = 13
    return data


def _migrate_v13_to_v14(data: dict) -> dict:
    """Schema v14: add research section with optional api_key."""
    data.setdefault("research", {"api_key": None})
    data["schema_version"] = 14
    return data


def _migrate_v14_to_v15(data: dict) -> dict:
    """Schema v15: customizable workspace apps (open my workspace)."""
    if "workspace" not in data:
        data["workspace"] = {
            "apps": [e.model_dump(mode="json") for e in _default_workspace_apps()],
        }
    data["schema_version"] = 15
    return data


def _migrate_v15_to_v16(data: dict) -> dict:
    """Schema v16: deep-research planner/worker model split + tuning knobs."""
    research = data.setdefault("research", {"api_key": None})
    research.setdefault("planner_model", "")
    research.setdefault("worker_model", "")
    research.setdefault("depth", 5)
    research.setdefault("breadth", 5)
    research.setdefault("fetch_pages", True)
    research.setdefault("max_page_chars", 6000)
    research.setdefault("enable_gap_fill", True)
    data["schema_version"] = 16
    return data


def _migrate_v16_to_v17(data: dict) -> dict:
    """Schema v17: deep research Ultra mode (free-tier retrieval + Groq planner)."""
    research = data.setdefault("research", {"api_key": None})
    research.setdefault("ultra_enabled", False)
    research.setdefault("brave_api_key", "")
    research.setdefault("groq_api_key", "")
    research.setdefault("ultra_planner_model", "groq/llama-3.3-70b-versatile")
    research.setdefault("ultra_gap_fill_iterations", 3)
    data["schema_version"] = 17
    return data


# Coordinates that prior migrations auto-stamped onto user configs. v18
# resets them so the open-source build doesn't leak any maintainer-specific
# location to users who never set a location themselves.
_STAMPED_WEATHER_COORDS: tuple[tuple[float, float], ...] = (
    (38.9586, -77.3570),  # Reston, VA — v8→v9 default fill
    (39.7187, -84.1736),  # Oakwood, OH — v9→v10 default replacement
)


def _migrate_v18_to_v19(data: dict) -> dict:
    """Schema v19: command_palette hotkey + first_run_completed flag.

    Pre-existing installs keep their existing hotkeys; we add the new
    binding with the standard default. The first-run flag is set True
    for existing users so they don't get an unexpected tutorial pop-up
    after the upgrade.
    """
    hotkeys = data.setdefault("hotkeys", {})
    hotkeys.setdefault("command_palette", "ctrl+shift+p")
    general = data.setdefault("general", {})
    general.setdefault("first_run_completed", True)
    data["schema_version"] = 19
    return data


def _migrate_v19_to_v20(data: dict) -> dict:
    """Schema v20: introduce vision section for the see_screen tool.

    Existing v19 configs gain the section with the shipping defaults
    (llava:7b vision model, 1280 px long edge, 512 max tokens, temp 0.2).
    Users who haven't pulled llava:7b will hear a clear "no vision model
    pulled" message when they first say "see my screen" — better than
    autoselecting the user's main text-only model and getting an opaque
    Ollama error mid-tool-call.
    """
    data.setdefault("vision", {
        "model": "llava:7b",
        "max_image_dim": 1280,
        "max_tokens": 512,
        "temperature": 0.2,
    })
    data["schema_version"] = 20
    return data


def _migrate_v17_to_v18(data: dict) -> dict:
    """Schema v18: privacy reset for migration-stamped weather coordinates.

    Earlier migrations (v8→v9, v9→v10) auto-filled the weather section
    with the maintainer's location when users had no coordinates set.
    Reset those exact stamped values back to None so the weather tool's
    ipapi.co fallback runs on first use and the user gets *their* local
    forecast. Any other coordinate (including a user who genuinely
    entered one of the stamped values by hand) is left untouched if it
    does not match the exact stamped pair.
    """
    weather = data.setdefault("weather", {})
    lat = weather.get("latitude")
    lon = weather.get("longitude")
    if (lat, lon) in _STAMPED_WEATHER_COORDS:
        weather["latitude"] = None
        weather["longitude"] = None
    data["schema_version"] = 18
    return data


def _migrate_v20_to_v21(data: dict) -> dict:
    """Schema v21: llm.max_tool_iterations for the tool-result feedback loop.

    Existing configs get the shipping default (3), which switches the
    loop on for them. That IS the intended upgrade: before it the model
    never saw what a tool returned, so nothing needing two steps
    ("list my Downloads and tell me which is the invoice") could work
    at all. Users who prefer the previous one-shot behaviour — the
    tool's own output spoken verbatim, exactly one inference per turn —
    set it to 1.
    """
    llm = data.setdefault("llm", {})
    llm.setdefault("max_tool_iterations", 3)
    data["schema_version"] = 21
    return data


MIGRATIONS: dict[int, Migration] = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
    4: _migrate_v4_to_v5,
    5: _migrate_v5_to_v6,
    6: _migrate_v6_to_v7,
    7: _migrate_v7_to_v8,
    8: _migrate_v8_to_v9,
    9: _migrate_v9_to_v10,
    10: _migrate_v10_to_v11,
    11: _migrate_v11_to_v12,
    12: _migrate_v12_to_v13,
    13: _migrate_v13_to_v14,
    14: _migrate_v14_to_v15,
    15: _migrate_v15_to_v16,
    16: _migrate_v16_to_v17,
    17: _migrate_v17_to_v18,
    18: _migrate_v18_to_v19,
    19: _migrate_v19_to_v20,
    20: _migrate_v20_to_v21,
}


class ConfigMigrationError(ValueError):
    """Raised when migration cannot proceed (missing version, unknown future
    version, missing migration, or a migration that didn't bump the version)."""


def migrate(data: dict) -> dict:
    """Apply registered migrations in sequence until data is at
    CURRENT_SCHEMA_VERSION. Returns the same dict object (possibly mutated)."""
    if "schema_version" not in data:
        raise ConfigMigrationError("config missing schema_version")
    version = data["schema_version"]
    if not isinstance(version, int):
        raise ConfigMigrationError(f"schema_version must be int, got {type(version).__name__}")
    if version > CURRENT_SCHEMA_VERSION:
        raise ConfigMigrationError(
            f"config schema_version {version} is newer than supported "
            f"{CURRENT_SCHEMA_VERSION}; downgrade is not supported"
        )
    while version < CURRENT_SCHEMA_VERSION:
        migration = MIGRATIONS.get(version)
        if migration is None:
            raise ConfigMigrationError(
                f"no migration registered for schema_version {version}"
            )
        data = migration(data)
        new_version = data.get("schema_version")
        if new_version != version + 1:
            raise ConfigMigrationError(
                f"migration {version} -> {version + 1} produced "
                f"schema_version={new_version!r}"
            )
        version = new_version
    return data


# --- persistence -----------------------------------------------------------


def default_config_path() -> Path:
    """Default config location.

    Windows: `%APPDATA%/Jarvis/config.json`
    Other:   `~/.jarvis/config.json` (skeleton; full platform support is
             out of scope for v1, see SPEC § Out of scope).
    """
    if sys.platform == "win32":
        appdata = environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Jarvis" / "config.json"
    return Path.home() / ".jarvis" / "config.json"


def load_config(path: Path | None = None) -> JarvisConfig:
    """Load config from disk, applying migrations. If the file does not exist,
    write a default config to that path and return it."""
    p = path if path is not None else default_config_path()
    if not p.exists():
        cfg = JarvisConfig()
        save_config(cfg, p)
        return cfg
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ConfigMigrationError(
            f"config root must be a JSON object, got {type(raw).__name__}"
        )
    raw = migrate(raw)
    return JarvisConfig.model_validate(raw)


def save_config(config: JarvisConfig, path: Path | None = None) -> None:
    """Write config atomically (tmp + replace)."""
    p = path if path is not None else default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2)
    tmp.replace(p)
