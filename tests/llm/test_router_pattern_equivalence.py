"""Golden-file equivalence net for the router's pattern layer.

Why this file exists
--------------------
The pattern table used to be a single hand-ordered literal inside
``jarvis.llm.intent_router``. It is now assembled from ``voice_patterns``
declared on the individual tool classes (see
``jarvis.tools.voice_patterns``). That move is only safe if the assembled
table is *byte-for-byte the same table, in the same order* — the layer is
first-match-wins and the order encodes real precedence
(``^open\\s+(.+)$`` must lose to ``open my notes``, ``open the dashboard``,
``open logs``, ``open my workspace``, ``open clipboard history``,
``open help``; ``read this note`` must beat ``read <title> note``; and so
on).

Two goldens, captured from the pre-refactor implementation and committed
next to this file:

``router_pattern_order_golden.json``
    The ordered list of regex sources in the assembled table. A diff here
    is a routing-precedence change and must be justified in review.

``router_pattern_golden.json``
    ``utterance -> resolved intent`` for every utterance in ``CORPUS``.
    Covers every example phrase in ``jarvis/ui/capabilities.py``, every
    voice example in the README feature list, and at least one utterance
    per pattern, plus the near-miss pairs that prove ordering.

Regenerate deliberately, never reflexively::

    python3 tests/llm/_regen_router_goldens.py

A regeneration that changes ``router_pattern_order_golden.json`` or any
existing key in ``router_pattern_golden.json`` is a behaviour change.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import cast

import pytest

from jarvis.core.config import ToolsConfig
from jarvis.llm.conversation import Conversation
from jarvis.llm.intent_router import (
    IntentRouter,
    SpeakIntent,
    StopIntent,
    ToolIntent,
    _build_patterns,
)
from jarvis.llm.ollama_client import OllamaClient
from jarvis.tools.registry import ToolRegistry
from jarvis.ui.capabilities import CAPABILITY_CATEGORIES

_GOLDEN_DIR = Path(__file__).parent
_ORDER_GOLDEN = _GOLDEN_DIR / "router_pattern_order_golden.json"
_INTENT_GOLDEN = _GOLDEN_DIR / "router_pattern_golden.json"

# Frozen so SpeakIntent("It's ...") is reproducible.
FIXED_TIME = datetime.datetime(2024, 3, 7, 13, 5, 0)


def fixed_time_provider() -> datetime.datetime:
    return FIXED_TIME


# --- corpus ----------------------------------------------------------
#
# Source 1: every example phrase in jarvis/ui/capabilities.py (asserted
#           exhaustive by test_corpus_covers_every_capability_example).
# Source 2: every voice example quoted in the README feature list.
# Source 3: at least one utterance per pattern in the table, plus the
#           near-miss pairs whose resolution depends on pattern order.

CORPUS: tuple[str, ...] = (
    # -- capabilities.py: Getting started
    "Hey Jarvis",
    "Hey Jarvis, what time is it?",
    "go to sleep",
    "stop listening",
    "mute",
    "unmute",
    # -- capabilities.py: Apps and workspace
    "open Spotify",
    "open Chrome",
    "open Notepad",
    "close Spotify",
    "close Chrome",
    "open my workspace",
    "launch workspace",
    "launch Cyberpunk",
    "boot up Stardew Valley",
    # -- capabilities.py: Music and media
    "play some jazz",
    "play Pink Floyd",
    "play Bohemian Rhapsody",
    "volume up",
    "volume down",
    # -- capabilities.py: Web and search
    "search for puppies",
    "google AI news",
    "search up pasta recipes",
    "open youtube dot com",
    "open github",
    # -- capabilities.py: Research
    "research black holes",
    "look up the Roman Empire",
    "deep research nuclear fusion",
    "do deep research on quantum computing",
    "pause deep research",
    "resume deep research",
    "delete deep research on solar power",
    "delete all deep research",
    # -- capabilities.py: Notes
    "take a note about the meeting being moved to Friday",
    "jot this down: buy milk and eggs",
    "write this down: project Phoenix kicks off Monday",
    "open my notes",
    "show notes",
    "close notes",
    "add this to my meeting note: agenda finalized",
    "add another item to my groceries note: paper towels",
    "read my meeting note",
    "read this note",
    "delete the groceries note",
    "delete this note",
    # -- capabilities.py: System
    "show dashboard",
    "open dashboard",
    "how is my computer doing",
    "what's the weather",
    "weather today",
    "take a screenshot",
    "screenshot",
    "what's on my screen",
    "look at my screen",
    "describe my screen",
    "what do you see",
    "lock the screen",
    "lock my pc",
    "how much memory am I using",
    "what can you do",
    "show help",
    "show capabilities",
    # -- capabilities.py: Clipboard and typing
    "what's on my clipboard",
    "clear my clipboard",
    "type hello world",
    "type my email address",
    "show clipboard history",
    "what have I copied",
    "paste my last copy",
    "paste item 3",
    "clear my clipboard history",
    # -- capabilities.py: Power-user tools
    "Press Ctrl+Shift+P",
    "(silent — type to filter, Enter to run)",
    "show logs",
    "show errors",
    "close logs",
    "Tray icon → Show tutorial",
    # -- README feature list
    "stop",
    "shut up",
    "never mind",
    "be quiet",
    "that's enough",
    "What time is it",
    "search the web for tide tables",
    "google the weather in oslo",
    "search up how to poach an egg",
    "search for a decent risotto",
    "read more",
    "continue",
    "copy that",
    "Take a note about the roof leak",
    "jot this down buy cat food",
    "write this down call the dentist",
    "Add this to my meeting note: bring the slides",
    "Read my groceries note",
    "Delete the groceries note",
    "Play Rachmaninoff",
    "Paste my last copy",
    "paste item 7",
    # -- one utterance per pattern, in table order
    "volume mute",
    "volume unmute",
    "take screenshot",
    "lock pc",
    "lock my pc",
    "what is the time",
    "what's the time",
    "close research",
    "keep going",
    "copy research",
    "copy the summary",
    "enable ultra research",
    "turn on deep research ultra",
    "use ultra mode",
    "disable ultra research",
    "turn off deep research ultra",
    "use normal deep research",
    "jarvis, normal deep research",
    "jarvis, deep research on tidal power",
    "close deep research",
    "remove all deep research history",
    "clear all deep research sessions",
    "remove the deep research about wind farms",
    "delete the deep research",
    "remove deep research",
    "look up sourdough",
    "jarvis, open my workspace",
    "start workspace",
    "bring up the dashboard",
    "show me the system stats",
    "open system stats",
    "close the dashboard",
    "bring up my clipboard history",
    "close the clipboard history",
    "clear the clipboard history",
    "paste item 12",
    "paste last copy",
    "bring up the logs",
    "show me the log",
    "show me errors",
    "close the logs",
    "open help",
    "open my capabilities",
    "what can i say",
    "see my screen",
    "read the monitor",
    "describe the display",
    "what is on the display",
    "what can you see on my screen",
    "can you see my screen",
    "bring up my notes",
    "close my notes",
    "note that the boiler needs servicing",
    "remember this the bins go out Tuesday",
    "read the current note",
    "read me my groceries notes",
    "delete the current note",
    "append this to the groceries note: rice",
    "google puppies",
    "open steam",
    # -- ordering proofs: the generic `open <app>` catch-all must LOSE
    #    to every more specific "open ..." pattern above it.
    "open notes",
    "open the dashboard",
    "open clipboard history",
    "open logs",
    "open the logs",
    "open workspace",
    "open capabilities",
    "open the notes",
    "open my dashboard",
    "open my clipboard history",
    # -- ordering proofs: specific note verbs must beat the generic
    #    `<title> note` captures.
    "read my current note",
    "delete the report note",
    "read the report note",
    # -- ordering proofs: deep research must beat quick research, and
    #    `continue` alone must beat `continue deep research`.
    "continue deep research",
    "research quantum computing",
    "deep research on the Krebs cycle",
    # -- normalization / filler stripping
    "could you please open spotify",
    "can you open spotify",
    "would you please take a screenshot",
    "hey jarvis, open spotify",
    "Hey Jarvis, volume up.",
    "OPEN SPOTIFY!!",
    "   open   spotify   ",
    # -- misses that must fall through to the LLM
    "tell me a joke",
    "",
    "what",
    "open",
    "stop the music",
    "cancel that download",
)


# --- serialization ---------------------------------------------------


def canonical(intent) -> dict:
    """Stable JSON shape for an intent. Tool name + args is the contract
    the brief cares about; text is included for SpeakIntent so the local
    time answer is pinned too."""
    if intent is None:
        return {"kind": "none"}
    if isinstance(intent, StopIntent):
        return {"kind": "stop"}
    if isinstance(intent, SpeakIntent):
        return {"kind": "speak", "text": intent.text}
    if isinstance(intent, ToolIntent):
        return {
            "kind": "tool",
            "tool": intent.tool_name,
            "args": intent.args,
            "spoken_response": intent.spoken_response,
        }
    raise AssertionError(f"unexpected intent type {type(intent)!r}")


def resolve_corpus(router: IntentRouter) -> dict[str, dict]:
    return {u: canonical(router._try_pattern(u)) for u in CORPUS}


def _router_without_registry() -> IntentRouter:
    """The pattern layer in isolation. No registry means _try_pattern
    never applies the is-it-registered gate, so every pattern in the
    table is reachable — which is what we want to pin."""
    # _try_pattern never reaches the LLM or the conversation, so these
    # two are placeholders; cast at the seam rather than build clients
    # the pattern layer will not touch.
    return IntentRouter(
        llm=cast(OllamaClient, object()),
        conversation=cast(Conversation, object()),
        time_provider=fixed_time_provider,
    )


# --- tests -----------------------------------------------------------


def test_pattern_table_order_matches_golden():
    """The assembled table must be in exactly the pre-refactor order.

    This is the direct guard on first-match-wins precedence: a pattern
    that moves earlier or later changes which tool wins an ambiguous
    utterance, and the corpus test below can only catch that where the
    corpus happens to contain the ambiguous phrase. This catches it
    unconditionally."""
    expected = json.loads(_ORDER_GOLDEN.read_text(encoding="utf-8"))
    actual = [p.pattern for p, _builder in _build_patterns(fixed_time_provider)]
    assert actual == expected


def test_every_corpus_utterance_resolves_identically_to_golden():
    expected = json.loads(_INTENT_GOLDEN.read_text(encoding="utf-8"))
    actual = resolve_corpus(_router_without_registry())
    # Compare per-utterance so a failure names the phrase that moved.
    for utterance in CORPUS:
        assert actual[utterance] == expected[utterance], (
            f"routing changed for {utterance!r}: "
            f"{expected[utterance]} -> {actual[utterance]}"
        )
    assert set(actual) == set(expected)


def test_corpus_covers_every_capability_example():
    """Every example phrase shown to users in the Help panel / command
    palette is pinned by the golden above.

    If this fails you added a capability example: add the phrase to
    CORPUS and regenerate the goldens (only the new key should appear in
    the diff)."""
    examples = {
        phrase
        for cat in CAPABILITY_CATEGORIES
        for cap in cat.capabilities
        for phrase in cap.examples
    }
    missing = sorted(examples - set(CORPUS))
    assert not missing, (
        "capability example phrases missing from the equivalence corpus: "
        f"{missing}. Add them to CORPUS and regenerate the goldens."
    )


@pytest.mark.parametrize(
    ("utterance", "tool"),
    [
        # The five "open ..." phrases the catch-all would steal.
        ("open my notes", "open_notes"),
        ("open the dashboard", "show_dashboard"),
        ("open clipboard history", "show_clipboard_history"),
        ("open logs", "show_logs"),
        ("open my workspace", "launch_workspace"),
        ("open help", "open_help"),
        # ... and one it is supposed to keep.
        ("open spotify", "open_app"),
    ],
)
def test_generic_open_catch_all_stays_last(utterance: str, tool: str):
    """Named separately from the golden because this is *the* regression
    the registry-driven assembly could reintroduce: if patterns were
    ordered by registration order rather than by explicit priority,
    'open my notes' would launch an app called 'my notes'."""
    intent = _router_without_registry()._try_pattern(utterance)
    assert isinstance(intent, ToolIntent)
    assert intent.tool_name == tool


def test_registry_gate_still_falls_through_for_unregistered_tools():
    """A pattern whose tool is not registered must return None (fall
    through to the LLM) rather than dispatch an unknown tool."""
    registry = ToolRegistry(ToolsConfig())
    router = IntentRouter(
        llm=cast(OllamaClient, object()),
        conversation=cast(Conversation, object()),
        registry=registry,
        time_provider=fixed_time_provider,
    )
    # Empty registry: every tool-backed pattern falls through...
    assert router._try_pattern("open spotify") is None
    # ... but non-tool intents are unaffected.
    assert isinstance(router._try_pattern("what time is it"), SpeakIntent)
    assert isinstance(router._try_pattern("stop"), StopIntent)
