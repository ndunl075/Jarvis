"""Single source of truth for "what can Jarvis do" — used by the Help
panel, the Settings → Help tab, and the command palette.

Each category groups related capabilities. Each capability has a plain-
English name, a one-sentence description aimed at non-technical users,
at least one example phrase the user can say out loud, and `tools`: the
registry names it is backed by.

Why this catalogue is hand-written and not generated from the registry
---------------------------------------------------------------------
The obvious follow-on to moving voice patterns onto the tools (see
`jarvis.tools.registry.VoicePattern`) is to move this catalogue there
too. It was tried and rejected, because a capability is a *user task* and
a tool is an *implementation*, and the two do not line up:

- The mapping is many-to-many, and grouping by tool reads worse. "Deep
  research" is one thing a user does; it is six tools (start, pause,
  resume, close, and the two ultra toggles). "Open or close notes" is one
  card with one description; it is two tools. Generating one card per
  tool would turn 32 human-scale entries into 42 mechanical ones and
  split several of them mid-explanation.
- Five capabilities have no tool at all — waking, sleeping, muting, the
  command palette, and the tutorial. Their "examples" are not even
  utterances ("Press Ctrl+Shift+P", "Tray icon → Show tutorial"). A
  generated catalogue would still need a hand-written list beside it for
  these, i.e. two mechanisms where there is now one.
- Category membership, category order, the glyphs, and the order of cards
  within a category are all editorial. The Help panel is read top to
  bottom; that sequence is not derivable from anything the tools know.

What IS fixed here is the drift the registry could have prevented: a card
describing a capability the registry does not have, or a tool nobody
documents. `tools` states the link explicitly and
`tests/ui/test_capabilities.py` checks it against
`jarvis.tools.catalogue` in both directions, so the prose stays editorial
while the claims stay true.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    examples: tuple[str, ...]
    # Registry tool names backing this capability. Empty for the handful
    # of capabilities that are not tools (wake / sleep / mute, the
    # command palette, the tutorial). Verified against the tool catalogue
    # by tests/ui/test_capabilities.py — see module docstring.
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityCategory:
    name: str
    icon_glyph: str  # ASCII / unicode mark shown in the UI tab
    capabilities: tuple[Capability, ...]


CAPABILITY_CATEGORIES: tuple[CapabilityCategory, ...] = (
    CapabilityCategory(
        name="Getting started",
        icon_glyph="▷",
        capabilities=(
            Capability(
                name="Wake Jarvis up",
                description=(
                    "Say the wake phrase and wait for the orb to glow. "
                    "Then speak your command in one breath."
                ),
                examples=("Hey Jarvis", "Hey Jarvis, what time is it?"),
                tools=(),
            ),
            Capability(
                name="Put Jarvis to sleep",
                description=(
                    "Stops listening and unloads heavy models so your PC "
                    "stays quick. Wake him back up with the wake phrase."
                ),
                examples=("go to sleep", "stop listening"),
                tools=(),
            ),
            Capability(
                name="Mute or unmute",
                description=(
                    "Mute pauses Jarvis without unloading anything — handy "
                    "on calls or during music."
                ),
                examples=("mute", "unmute"),
                tools=(),
            ),
        ),
    ),
    CapabilityCategory(
        name="Apps and workspace",
        icon_glyph="▢",
        capabilities=(
            Capability(
                name="Open an app",
                description=(
                    "Launches a desktop program by name. Works for things "
                    "in your Start menu and Microsoft Store apps."
                ),
                examples=(
                    "open Spotify",
                    "open Chrome",
                    "open Notepad",
                ),
                tools=("open_app",),
            ),
            Capability(
                name="Close an app",
                description="Closes a running program by name.",
                examples=("close Spotify", "close Chrome"),
                tools=("close_app",),
            ),
            Capability(
                name="Open your whole workspace",
                description=(
                    "Launches every app in your saved workspace at once. "
                    "Set the list up under Settings → General → Workspace."
                ),
                examples=(
                    "open my workspace",
                    "launch workspace",
                ),
                tools=("launch_workspace",),
            ),
            Capability(
                name="Launch a Steam game",
                description="Opens a game in your Steam library by title.",
                examples=("launch Cyberpunk", "boot up Stardew Valley"),
                tools=("launch_steam_game",),
            ),
        ),
    ),
    CapabilityCategory(
        name="Music and media",
        icon_glyph="♪",
        capabilities=(
            Capability(
                name="Play music",
                description=(
                    "Finds the song or artist on YouTube and plays the top "
                    "result with autoplay."
                ),
                examples=(
                    "play some jazz",
                    "play Pink Floyd",
                    "play Bohemian Rhapsody",
                ),
                tools=("play_youtube_music",),
            ),
            Capability(
                name="Change volume",
                description="System volume up, down, mute, or unmute.",
                examples=("volume up", "volume down", "mute", "unmute"),
                tools=("volume",),
            ),
        ),
    ),
    CapabilityCategory(
        name="Web and search",
        icon_glyph="◇",
        capabilities=(
            Capability(
                name="Search the web",
                description="Opens a Google search in your default browser.",
                examples=(
                    "search for puppies",
                    "google AI news",
                    "search up pasta recipes",
                ),
                tools=("open_url",),
            ),
            Capability(
                name="Open a website",
                description="Opens any URL you tell him to.",
                examples=("open youtube dot com", "open github"),
                tools=("open_url",),
            ),
        ),
    ),
    CapabilityCategory(
        name="Research",
        icon_glyph="✎",
        capabilities=(
            Capability(
                name="Quick research",
                description=(
                    "Pops open a side panel with a short web summary and "
                    "speaks the first two sentences out loud."
                ),
                examples=(
                    "research black holes",
                    "look up the Roman Empire",
                ),
                tools=(
                    "research",
                    "close_research",
                    "read_more",
                    "copy_research",
                ),
            ),
            Capability(
                name="Deep research",
                description=(
                    "A longer, multi-stage investigation. Jarvis plans "
                    "sub-questions, reads sources, writes a full report "
                    "with citations, and saves it as a markdown file you "
                    "can re-open from the panel. You can pause and resume."
                ),
                examples=(
                    "deep research nuclear fusion",
                    "do deep research on quantum computing",
                    "pause deep research",
                    "resume deep research",
                ),
                tools=(
                    "deep_research",
                    "pause_deep_research",
                    "resume_deep_research",
                    "close_deep_research",
                    "enable_deep_research_ultra",
                    "disable_deep_research_ultra",
                ),
            ),
            Capability(
                name="Delete research",
                description=(
                    "Removes one or all saved deep research sessions."
                ),
                examples=(
                    "delete deep research on solar power",
                    "delete all deep research",
                ),
                tools=(
                    "delete_deep_research",
                    "delete_all_deep_research",
                ),
            ),
        ),
    ),
    CapabilityCategory(
        name="Notes",
        icon_glyph="✐",
        capabilities=(
            Capability(
                name="Take a note",
                description=(
                    "Saves whatever you say next as a new markdown note "
                    "in your notes folder."
                ),
                examples=(
                    "take a note about the meeting being moved to Friday",
                    "jot this down: buy milk and eggs",
                    "write this down: project Phoenix kicks off Monday",
                ),
                tools=("take_note",),
            ),
            Capability(
                name="Open or close notes",
                description="Brings up the notes panel to browse and edit.",
                examples=(
                    "open my notes",
                    "show notes",
                    "close notes",
                ),
                tools=(
                    "open_notes",
                    "close_notes",
                ),
            ),
            Capability(
                name="Add to an existing note",
                description=(
                    "Appends to the matching note (or the one currently "
                    "open in the panel)."
                ),
                examples=(
                    "add this to my meeting note: agenda finalized",
                    "add another item to my groceries note: paper towels",
                ),
                tools=("append_to_note",),
            ),
            Capability(
                name="Read a note",
                description="Reads a note out loud.",
                examples=(
                    "read my meeting note",
                    "read this note",
                ),
                tools=("read_note",),
            ),
            Capability(
                name="Delete a note",
                description="Removes a saved note.",
                examples=(
                    "delete the groceries note",
                    "delete this note",
                ),
                tools=("delete_note",),
            ),
        ),
    ),
    CapabilityCategory(
        name="System",
        icon_glyph="⌘",
        capabilities=(
            Capability(
                name="Show the dashboard",
                description=(
                    "A live HUD with CPU, RAM, mic level, current mode, "
                    "model in use, and active foreground app."
                ),
                examples=(
                    "show dashboard",
                    "open dashboard",
                    "how is my computer doing",
                ),
                tools=(
                    "show_dashboard",
                    "close_dashboard",
                ),
            ),
            Capability(
                name="Get the weather",
                description=(
                    "Reads the current weather for your saved location. "
                    "Set it under Settings → General → Weather location."
                ),
                examples=("what's the weather", "weather today"),
                tools=("get_weather",),
            ),
            Capability(
                name="Take a screenshot",
                description="Captures the screen and copies it to clipboard.",
                examples=("take a screenshot", "screenshot"),
                tools=("screenshot",),
            ),
            Capability(
                name="See your screen",
                description=(
                    "Captures your screen and describes what's on it "
                    "using a vision-capable local LLM. Requires a "
                    "multimodal Ollama model (e.g. llava:7b). Set the "
                    "model under Settings → Models → Vision."
                ),
                examples=(
                    "what's on my screen",
                    "look at my screen",
                    "describe my screen",
                    "what do you see",
                ),
                tools=("see_screen",),
            ),
            Capability(
                name="Lock the screen",
                description="Locks Windows just like Win+L.",
                examples=("lock the screen", "lock my pc"),
                tools=("lock_screen",),
            ),
            Capability(
                name="System stats by voice",
                description="Reports CPU and memory usage out loud.",
                examples=("how much memory am I using",),
                tools=("report_cpu_and_memory_percentages",),
            ),
            Capability(
                name="Help",
                description=(
                    "Opens this exact list — a clear, plain-English guide "
                    "to everything Jarvis can do."
                ),
                examples=(
                    "what can you do",
                    "show help",
                    "show capabilities",
                ),
                tools=("open_help",),
            ),
        ),
    ),
    CapabilityCategory(
        name="Clipboard and typing",
        icon_glyph="✂",
        capabilities=(
            Capability(
                name="Copy / paste / read clipboard",
                description="Manage the Windows clipboard hands-free.",
                examples=(
                    "what's on my clipboard",
                    "clear my clipboard",
                ),
                tools=("clipboard",),
            ),
            Capability(
                name="Type into the focused window",
                description=(
                    "Types whatever you dictate into whichever window "
                    "currently has focus."
                ),
                examples=(
                    "type hello world",
                    "type my email address",
                ),
                tools=("type_into_active_window",),
            ),
            Capability(
                name="Clipboard history",
                description=(
                    "A scrollable list of recent text you've copied. "
                    "Pin items to keep them around, double-click to load "
                    "back onto the clipboard for the next Ctrl+V."
                ),
                examples=(
                    "show clipboard history",
                    "what have I copied",
                    "paste my last copy",
                    "paste item 3",
                    "clear my clipboard history",
                ),
                tools=(
                    "show_clipboard_history",
                    "close_clipboard_history",
                    "clear_clipboard_history",
                    "paste_clipboard_item",
                ),
            ),
        ),
    ),
    CapabilityCategory(
        name="Power-user tools",
        icon_glyph="⚡",
        capabilities=(
            Capability(
                name="Command palette",
                description=(
                    "Keyboard launcher. Press Ctrl+Shift+P and start typing "
                    "to find any command — handy on calls or anywhere "
                    "speaking out loud isn't appropriate."
                ),
                examples=(
                    "Press Ctrl+Shift+P",
                    "(silent — type to filter, Enter to run)",
                ),
                tools=(),
            ),
            Capability(
                name="Live log viewer",
                description=(
                    "Tails Jarvis's log file with a level filter and search "
                    "box. Use it to see what's happening behind the scenes "
                    "or copy text into a bug report."
                ),
                examples=(
                    "show logs",
                    "show errors",
                    "close logs",
                ),
                tools=(
                    "show_logs",
                    "close_logs",
                ),
            ),
            Capability(
                name="Show the tutorial again",
                description=(
                    "The first-run welcome walkthrough — mic test, wake-"
                    "word test, and a quick command suggestion."
                ),
                examples=(
                    "Tray icon → Show tutorial",
                ),
                tools=(),
            ),
        ),
    ),
)


def all_capabilities() -> tuple[Capability, ...]:
    out: list[Capability] = []
    for cat in CAPABILITY_CATEGORIES:
        out.extend(cat.capabilities)
    return tuple(out)


def search_capabilities(query: str) -> list[tuple[CapabilityCategory, Capability]]:
    """Return (category, capability) pairs matching ``query`` (substring,
    case-insensitive) in name, description, or examples."""
    q = (query or "").strip().lower()
    if not q:
        return [(c, cap) for c in CAPABILITY_CATEGORIES for cap in c.capabilities]
    hits: list[tuple[CapabilityCategory, Capability]] = []
    for cat in CAPABILITY_CATEGORIES:
        for cap in cat.capabilities:
            haystack = " ".join(
                (cap.name, cap.description, *cap.examples, cat.name)
            ).lower()
            if q in haystack:
                hits.append((cat, cap))
    return hits
