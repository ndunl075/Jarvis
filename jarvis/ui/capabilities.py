"""Single source of truth for "what can Jarvis do" — used by the Help
panel and the Settings → Help tab.

Each category groups related capabilities. Each capability has a plain-
English name, a one-sentence description aimed at non-technical users,
and at least one example phrase the user can say out loud.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    examples: tuple[str, ...]


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
            ),
            Capability(
                name="Put Jarvis to sleep",
                description=(
                    "Stops listening and unloads heavy models so your PC "
                    "stays quick. Wake him back up with the wake phrase."
                ),
                examples=("go to sleep", "stop listening"),
            ),
            Capability(
                name="Mute or unmute",
                description=(
                    "Mute pauses Jarvis without unloading anything — handy "
                    "on calls or during music."
                ),
                examples=("mute", "unmute"),
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
            ),
            Capability(
                name="Close an app",
                description="Closes a running program by name.",
                examples=("close Spotify", "close Chrome"),
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
            ),
            Capability(
                name="Launch a Steam game",
                description="Opens a game in your Steam library by title.",
                examples=("launch Cyberpunk", "boot up Stardew Valley"),
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
            ),
            Capability(
                name="Change volume",
                description="System volume up, down, mute, or unmute.",
                examples=("volume up", "volume down", "mute", "unmute"),
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
            ),
            Capability(
                name="Open a website",
                description="Opens any URL you tell him to.",
                examples=("open youtube dot com", "open github"),
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
            ),
            Capability(
                name="Open or close notes",
                description="Brings up the notes panel to browse and edit.",
                examples=(
                    "open my notes",
                    "show notes",
                    "close notes",
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
            ),
            Capability(
                name="Read a note",
                description="Reads a note out loud.",
                examples=(
                    "read my meeting note",
                    "read this note",
                ),
            ),
            Capability(
                name="Delete a note",
                description="Removes a saved note.",
                examples=(
                    "delete the groceries note",
                    "delete this note",
                ),
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
            ),
            Capability(
                name="Get the weather",
                description=(
                    "Reads the current weather for your saved location. "
                    "Set it under Settings → General → Weather location."
                ),
                examples=("what's the weather", "weather today"),
            ),
            Capability(
                name="Take a screenshot",
                description="Captures the screen and copies it to clipboard.",
                examples=("take a screenshot", "screenshot"),
            ),
            Capability(
                name="Lock the screen",
                description="Locks Windows just like Win+L.",
                examples=("lock the screen", "lock my pc"),
            ),
            Capability(
                name="System stats by voice",
                description="Reports CPU and memory usage out loud.",
                examples=("how much memory am I using",),
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
