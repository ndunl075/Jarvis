"""Hybrid intent router: pattern layer for snap commands, LLM for the rest.

Two layers
----------
1. Pattern layer (deterministic, fast, no LLM call). First-match-wins
   table of (regex, intent_builder) tuples applied to the user's
   transcription after filler-word stripping. Misses fall through to
   the LLM. SPEC: this is what makes Jarvis feel snappy -- "open
   spotify" should never wait for a 7B-parameter model.

2. LLM layer. The transcription is added to the conversation history,
   OllamaClient.stream_chat() is invoked, and each Ollama chunk's
   content becomes a SpeakIntent yielded immediately (no buffering).
   Tool calls are emitted as ToolIntents (currently arrive batched on
   the final chunk; defensive accumulation point ready if Ollama later
   streams them).

Streaming contract with the pipeline
------------------------------------
route() is an async generator yielding Intent values. The pipeline
(Phase 4 wiring) collects consecutive SpeakIntents into a queue feeding
PiperTTS.speak_stream() so token-level chunks get sentence-segmented
inside TTS rather than firing one synthesis per token. ToolIntents
break the speak run cleanly (end-of-stream sentinel + speak_task
completion + tool execution). See the Phase 3 design note for the
dispatch pattern.

Cancellation
------------
The async generator propagates CancelledError up through
ollama_client.stream_chat() into the httpx stream's context manager
(verified by ollama_client tests). add_assistant_turn() is outside any
try/except, so a cancelled stream simply never reaches it -- partial
assistant text never enters conversation history.

Deliberate tradeoffs (documented at point of choice)
----------------------------------------------------
- User turn IS preserved on cancellation. The user said it, that's
  fact. Two consecutive user turns with no assistant between is
  technically malformed history (most chat APIs assume strict
  alternation), but tracking "did this turn get a response?" adds
  state that's not worth Phase 3's complexity. Future-polish knob if
  the malformed shape causes observable model-quality issues.

- Tool execution is sequential after a SpeakIntent run completes.
  No mid-sentence tool firing in the current implementation: a stream
  of Speak("foo ") -> Speak("bar") -> Tool(...) drains the speak run
  fully before the tool fires. The architecture supports interleaving
  (ToolIntent breaks the speak run via the pipeline's queue sentinel)
  but the current router doesn't decompose responses to make use of
  it. Acceptable for v1; revisit if a UX win emerges from interleaving.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

if TYPE_CHECKING:
    from jarvis.llm.conversation import Conversation
    from jarvis.llm.ollama_client import OllamaClient
    from jarvis.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


# --- Intent types ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpeakIntent:
    """Text to be spoken. The router yields multiple of these for an
    LLM response (one per Ollama content chunk); the pipeline pumps
    them into PiperTTS.speak_stream which sentence-segments internally."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolIntent:
    """A tool to execute. spoken_response is an optional confirmation
    line (e.g. 'Done, sir.') the pipeline speaks after dispatch."""

    tool_name: str
    args: dict
    spoken_response: str | None = None


@dataclass(frozen=True, slots=True)
class CompoundIntent:
    """Group of intents from a single user utterance. The router itself
    does not currently emit this -- iterator-of-flat-intents handles
    'open spotify and tell me the weather' naturally via multiple
    yields. Reserved for SPEC compliance and future patterns that need
    explicit grouping (atomic dispatch semantics, etc.)."""

    intents: tuple


@dataclass(frozen=True, slots=True)
class StopIntent:
    """User wants Jarvis to stop talking without a verbal response.

    Emitted when the transcription matches _STOP_PATTERN ('stop',
    'shut up', 'nevermind', etc.). The pipeline adapter yields nothing
    for this intent so TTS never fires. The pipeline's zero-yield path
    (first_chunk_seen stays False) transitions ConversationalState to
    IDLE automatically. Neither a user turn nor an assistant turn is
    added to conversation history."""


Intent = SpeakIntent | ToolIntent | CompoundIntent | StopIntent

# The shape the Phase 4 pipeline will accept in place of the Phase 2
# string-based ResponseProducer.
IntentProducer = Callable[[str], AsyncIterator[Intent]]


# --- normalization ---------------------------------------------------

# Leading filler patterns. Stripped iteratively until no more match.
_FILLERS: tuple[re.Pattern, ...] = (
    re.compile(r"^hey jarvis,?\s+"),
    re.compile(r"^(?:can|could|would) you (?:please )?"),
    re.compile(r"^please\s+"),
)

_TRAILING_PUNCT = re.compile(r"[\s.!?,]+$")


def _normalize(text: str) -> str:
    """Lowercase, strip, drop trailing punctuation, peel off fillers
    iteratively. Returns the form the pattern table sees."""
    s = text.lower().strip()
    s = _TRAILING_PUNCT.sub("", s)
    # Iteratively peel fillers (e.g. "could you please open spotify"
    # -> "please open spotify" -> "open spotify").
    while True:
        new_s = s
        for pat in _FILLERS:
            new_s = pat.sub("", new_s, count=1)
        if new_s == s:
            break
        s = new_s
    return s


# --- pattern layer ---------------------------------------------------

# Standalone stop commands. Checked before the normal pattern table so
# "hey jarvis, stop" never accidentally falls through to a tool pattern.
# Normalized text (lowercase, no trailing punctuation) is matched here.
# Multi-word variants ("shut up", "be quiet", "never mind") handled via
# \s+ so Whisper's spacing choices don't matter. Does NOT match commands
# with an object ("stop the music", "cancel that download") — those
# intentionally miss and fall through to the LLM.
_STOP_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:stop|shut\s+up|quiet|be\s+quiet|never\s*mind|cancel"
    r"|that'?s\s+enough|enough)$",
    re.IGNORECASE,
)


def _what_time_intent(
    time_provider: Callable[[], datetime.datetime],
) -> SpeakIntent:
    now = time_provider()
    formatted = now.strftime("%I:%M %p").lstrip("0")
    return SpeakIntent(text=f"It's {formatted}, sir.")


def _build_patterns(
    time_provider: Callable[[], datetime.datetime],
) -> list[tuple[re.Pattern, Callable[[re.Match], Intent]]]:
    """First-match-wins table. More specific patterns first."""
    return [
        # volume up/down/mute/unmute
        (
            re.compile(r"^volume\s+(up|down|mute|unmute)$"),
            lambda m: ToolIntent("volume", {"action": m.group(1)}),
        ),
        # screenshot, take a screenshot, take screenshot
        (
            re.compile(r"^(?:take\s+(?:a\s+)?)?screenshot$"),
            lambda m: ToolIntent("screenshot", {}),
        ),
        # lock the screen, lock my pc, lock pc, lock screen
        (
            re.compile(r"^lock\s+(?:the\s+)?(?:screen|my\s+pc|pc)$"),
            lambda m: ToolIntent("lock_screen", {}),
        ),
        # what time is it / what's the time / what is the time
        (
            re.compile(
                r"^what(?:'s|\s+is)\s+(?:the\s+)?time(?:\s+is\s+it)?$"
            ),
            lambda m: _what_time_intent(time_provider),
        ),
        (
            re.compile(r"^what\s+time\s+is\s+it$"),
            lambda m: _what_time_intent(time_provider),
        ),
        # research panel tools (before generic google search)
        (
            re.compile(r"^close\s+research$"),
            lambda m: ToolIntent("close_research", {}),
        ),
        (
            re.compile(r"^(?:read\s+more|continue|keep\s+going)$"),
            lambda m: ToolIntent("read_more", {}),
        ),
        (
            re.compile(r"^copy\s+(?:that|research|the\s+summary)$"),
            lambda m: ToolIntent("copy_research", {}),
        ),
        # deep research ultra toggle (before topic deep research)
        (
            re.compile(
                r"^(?:jarvis,?\s+)?(?:enable|turn\s+on|use)\s+"
                r"(?:(?:deep\s+research\s+)?ultra(?:\s+(?:research|mode))?|ultra\s+research)$"
            ),
            lambda m: ToolIntent("enable_deep_research_ultra", {}),
        ),
        (
            re.compile(
                r"^(?:jarvis,?\s+)?(?:disable|turn\s+off)\s+"
                r"(?:(?:deep\s+research\s+)?ultra(?:\s+(?:research|mode))?|ultra\s+research)$"
            ),
            lambda m: ToolIntent("disable_deep_research_ultra", {}),
        ),
        (
            re.compile(
                r"^(?:jarvis,?\s+)?(?:use\s+)?normal\s+deep\s+research$"
            ),
            lambda m: ToolIntent("disable_deep_research_ultra", {}),
        ),
        # deep research (must precede quick "research …")
        (
            re.compile(
                r"^(?:jarvis,?\s+)?(?:do\s+)?deep\s+research(?:\s+on)?\s+(.+)$"
            ),
            lambda m: ToolIntent("deep_research", {"query": m.group(1)}),
        ),
        (
            re.compile(r"^pause\s+deep\s+research$"),
            lambda m: ToolIntent("pause_deep_research", {}),
        ),
        (
            re.compile(r"^resume\s+deep\s+research$"),
            lambda m: ToolIntent("resume_deep_research", {}),
        ),
        (
            re.compile(r"^continue\s+deep\s+research$"),
            lambda m: ToolIntent("resume_deep_research", {}),
        ),
        (
            re.compile(r"^close\s+deep\s+research$"),
            lambda m: ToolIntent("close_deep_research", {}),
        ),
        (
            re.compile(r"^(?:delete|remove|clear)\s+all\s+deep\s+research(?:\s+(?:history|sessions))?$"),
            lambda m: ToolIntent("delete_all_deep_research", {}),
        ),
        (
            re.compile(
                r"^(?:delete|remove)\s+(?:the\s+)?deep\s+research(?:\s+(?:on|about))?\s+(.+)$"
            ),
            lambda m: ToolIntent("delete_deep_research", {"query": m.group(1)}),
        ),
        (
            re.compile(r"^(?:delete|remove)\s+(?:the\s+)?deep\s+research$"),
            lambda m: ToolIntent("delete_deep_research", {"query": ""}),
        ),
        (
            re.compile(r"^research\s+(.+)$"),
            lambda m: ToolIntent("research", {"query": m.group(1)}),
        ),
        (
            re.compile(r"^look\s+up\s+(.+)$"),
            lambda m: ToolIntent("research", {"query": m.group(1)}),
        ),
        # open / launch / start (my) workspace — includes "jarvis open my workspace"
        (
            re.compile(
                r"^(?:jarvis,?\s+)?(?:open|launch|start)\s+(?:my\s+)?workspace$"
            ),
            lambda m: ToolIntent("launch_workspace", {}),
        ),
        # Dashboard. The "my/the" alternatives matter: without them the
        # catch-all "open <anything>" pattern below would steal "open my
        # dashboard" and route it to open_app.
        (
            re.compile(
                r"^(?:show|open|bring\s+up)\s+(?:the\s+|my\s+)?dashboard$"
            ),
            lambda m: ToolIntent("show_dashboard", {}),
        ),
        (
            re.compile(
                r"^(?:show|open)\s+(?:me\s+)?(?:the\s+|my\s+)?system\s+stats$"
            ),
            lambda m: ToolIntent("show_dashboard", {}),
        ),
        (
            re.compile(r"^close\s+(?:the\s+|my\s+)?dashboard$"),
            lambda m: ToolIntent("close_dashboard", {}),
        ),
        # Clipboard history
        (
            re.compile(
                r"^(?:show|open|bring\s+up)\s+(?:the\s+|my\s+)?clipboard\s+history$"
            ),
            lambda m: ToolIntent("show_clipboard_history", {}),
        ),
        (
            re.compile(r"^what\s+have\s+i\s+copied\??$"),
            lambda m: ToolIntent("show_clipboard_history", {}),
        ),
        (
            re.compile(r"^close\s+(?:the\s+|my\s+)?clipboard\s+history$"),
            lambda m: ToolIntent("close_clipboard_history", {}),
        ),
        (
            re.compile(r"^clear\s+(?:the\s+|my\s+)?clipboard\s+history$"),
            lambda m: ToolIntent("clear_clipboard_history", {}),
        ),
        (
            re.compile(r"^paste\s+(?:item\s+)?(?:my\s+)?last\s+copy$"),
            lambda m: ToolIntent("paste_clipboard_item", {"index": 1}),
        ),
        (
            re.compile(r"^paste\s+item\s+(\d{1,2})$"),
            lambda m: ToolIntent(
                "paste_clipboard_item", {"index": int(m.group(1))}
            ),
        ),
        # Logs
        (
            re.compile(
                r"^(?:show|open|bring\s+up)\s+(?:me\s+)?(?:the\s+|my\s+)?logs?$"
            ),
            lambda m: ToolIntent("show_logs", {}),
        ),
        (
            re.compile(r"^show\s+(?:me\s+)?errors$"),
            lambda m: ToolIntent("show_logs", {}),
        ),
        (
            re.compile(r"^close\s+(?:the\s+|my\s+)?logs?$"),
            lambda m: ToolIntent("close_logs", {}),
        ),
        # Help / capabilities
        (
            re.compile(r"^(?:show|open)\s+(?:the\s+|my\s+)?help$"),
            lambda m: ToolIntent("open_help", {}),
        ),
        (
            re.compile(r"^(?:show|open)\s+(?:my\s+|the\s+)?capabilities$"),
            lambda m: ToolIntent("open_help", {}),
        ),
        (
            re.compile(r"^what\s+can\s+(?:you|i)\s+(?:do|say)\??$"),
            lambda m: ToolIntent("open_help", {}),
        ),
        # See screen (vision). MUST precede the notes patterns below
        # because "read my screen" would otherwise match the
        # `read (.+) note` regex and route to read_note("screen").
        # `look at` / `see` / `read` / `describe` cover the natural
        # phrasings; the trailing "screen|display|monitor" anchors the
        # tool so generic verbs (open, close, lock) keep their own
        # patterns. Snappier than letting the LLM tool-call.
        (
            re.compile(
                r"^(?:look\s+at|see|read|describe)\s+(?:my\s+|the\s+)?"
                r"(?:screen|display|monitor)$"
            ),
            lambda m: ToolIntent("see_screen", {}),
        ),
        # "what's on my screen" / "what is on the screen" — the suffix
        # is required so "what's" alone never triggers a tool call.
        (
            re.compile(
                r"^what(?:'s|\s+is)\s+on\s+(?:my\s+|the\s+)?"
                r"(?:screen|display|monitor)$"
            ),
            lambda m: ToolIntent("see_screen", {}),
        ),
        # "what do you see" / "what can you see" optionally followed by
        # "on my screen". "see" here is the visual verb, not the tool
        # name — kept separate from the look-at pattern above for clarity.
        (
            re.compile(
                r"^what\s+(?:do|can)\s+you\s+see"
                r"(?:\s+on\s+(?:my\s+|the\s+)?(?:screen|display|monitor))?$"
            ),
            lambda m: ToolIntent("see_screen", {}),
        ),
        # "can you see my screen" / "see my screen"
        (
            re.compile(r"^(?:can\s+you\s+)?see\s+my\s+screen$"),
            lambda m: ToolIntent("see_screen", {}),
        ),
        # Notes
        (
            re.compile(r"^(?:open|show|bring\s+up)\s+(?:my\s+|the\s+)?notes$"),
            lambda m: ToolIntent("open_notes", {}),
        ),
        (
            re.compile(r"^close\s+(?:my\s+|the\s+)?notes$"),
            lambda m: ToolIntent("close_notes", {}),
        ),
        (
            re.compile(
                r"^(?:take\s+a\s+note|jot\s+(?:this\s+)?down|write\s+(?:this\s+)?down|note\s+that|remember\s+this)"
                r"(?:\s+(?:that|about|saying))?[:\s]+(.+)$"
            ),
            lambda m: ToolIntent("take_note", {"content": m.group(1)}),
        ),
        (
            re.compile(r"^read\s+(?:this|the\s+current|my\s+current)\s+note$"),
            lambda m: ToolIntent("read_note", {"title": ""}),
        ),
        (
            re.compile(
                r"^read\s+(?:me\s+)?(?:my\s+|the\s+)?(.+?)\s+notes?$"
            ),
            lambda m: ToolIntent("read_note", {"title": m.group(1)}),
        ),
        (
            re.compile(r"^delete\s+(?:this|the\s+current)\s+note$"),
            lambda m: ToolIntent("delete_note", {"title": ""}),
        ),
        (
            re.compile(
                r"^delete\s+(?:the\s+)?(.+?)\s+note$"
            ),
            lambda m: ToolIntent("delete_note", {"title": m.group(1)}),
        ),
        (
            re.compile(
                r"^(?:add|append)\s+(?:this\s+)?to\s+(?:my\s+|the\s+)?(.+?)\s+note[:\s]+(.+)$"
            ),
            lambda m: ToolIntent(
                "append_to_note", {"title": m.group(1), "content": m.group(2)}
            ),
        ),
        # search <query>, google <query>, search for <query>, search up
        # <query>, google for <query>. The `(?:up|for)\s+` is inside the
        # optional group with a trailing \s+ so a query starting with
        # 'forty' isn't shaved to 'ty' by the regex hungrily eating
        # 'for'. quote_plus (not quote) — Google's search URL expects
        # '+' for spaces; the live test caught the encoding mismatch.
        (
            re.compile(r"^(?:search|google)\s+(?:(?:up|for)\s+)?(.+)$"),
            lambda m: ToolIntent(
                "open_url",
                {
                    "url": (
                        "https://www.google.com/search?q="
                        + quote_plus(m.group(1))
                    ),
                },
            ),
        ),
        # open <app> -- catches "open spotify", "open my browser", etc.
        # play / launch / start games or music -> LLM picks the right tool.
        # Placed last because it's the most permissive.
        (
            re.compile(r"^open\s+(.+)$"),
            lambda m: ToolIntent("open_app", {"name": m.group(1)}),
        ),
    ]


# --- router ----------------------------------------------------------


class IntentRouter:
    def __init__(
        self,
        *,
        llm: OllamaClient,
        conversation: Conversation,
        tools: list[dict] | None = None,
        registry: ToolRegistry | None = None,
        time_provider: Callable[[], datetime.datetime] = datetime.datetime.now,
    ) -> None:
        # `registry` is the live source of the tool schemas — querying it
        # fresh on every route() call means a settings-driven enable/disable
        # takes effect on the very next turn (no rebuild of the router).
        # `tools` is the legacy static-list path retained for the existing
        # test suite and any caller that wants to inject a curated schema
        # set; if both are provided, registry wins.
        self._llm = llm
        self._conv = conversation
        self._registry = registry
        self._static_tools = tools or []
        self._patterns = _build_patterns(time_provider)

    def _current_tools(self) -> list[dict]:
        if self._registry is not None:
            return self._registry.as_openai_functions()
        return self._static_tools

    async def route(self, transcription: str) -> AsyncIterator[Intent]:
        """Route a transcription into a stream of Intents.

        Pattern matches yield exactly one Intent and return. LLM responses
        yield SpeakIntents per content chunk and ToolIntents per
        function call (currently batched on the final Ollama chunk)."""
        pattern_intent = self._try_pattern(transcription)
        if pattern_intent is not None:
            yield pattern_intent
            return

        # LLM path. Buffer text until the stream ends — if the model also
        # emitted tool calls, speak only the tool result (once), not the
        # narration + tool confirmation ("Opening Chrome…" twice).
        messages = self._conv.add_user_turn(transcription)
        full_text_parts: list[str] = []
        tool_calls_seen: list[dict] = []
        async for chunk in self._llm.stream_chat(messages, tools=self._current_tools()):
            if chunk.content:
                full_text_parts.append(chunk.content)
            if chunk.tool_calls:
                # Per current Ollama behavior, tool_calls arrive complete
                # on the final chunk. Defensive accumulation here is the
                # hook point for future per-chunk streaming -- if Ollama
                # adopts incremental tool-call emission, partial-call
                # assembly logic plugs in around this list.
                tool_calls_seen.extend(chunk.tool_calls)
            if chunk.done:
                break

        # Visibility for live tool-decision debugging: every LLM-chosen
        # tool gets printed alongside the transcription that prompted it.
        # The pattern path doesn't show up here — those are router-level
        # decisions, not LLM ones. print (not log.info) keeps the trace
        # visible without bumping the global level, matching the rest of
        # the [router] / [boot] diagnostics.
        for tc in tool_calls_seen:
            fn = tc.get("function") or {}
            tool_name = fn.get("name") or "<missing>"
            print(
                f"[router] llm-chose-tool={tool_name} for={transcription!r}"
            )
        if len(tool_calls_seen) > 1:
            log.info(
                "[router] ToolIntent count=%d in one response",
                len(tool_calls_seen),
            )
        seen_tool_keys: set[str] = set()
        for tc in tool_calls_seen:
            intent = self._tool_intent_from(tc)
            if intent is None:
                continue
            key = json.dumps(
                {"name": intent.tool_name, "args": intent.args}, sort_keys=True
            )
            if key in seen_tool_keys:
                log.info(
                    "[router] skipped duplicate tool call: %s(%r)",
                    intent.tool_name,
                    intent.args,
                )
                continue
            seen_tool_keys.add(key)
            yield intent

        if not tool_calls_seen:
            for part in full_text_parts:
                if part:
                    yield SpeakIntent(text=part)

        # Only LLM-GENERATED text (chunk.content from the model) gets
        # stored as an assistant turn. Tool result strings — which are
        # produced downstream by execute_intent in the producer adapter
        # and spoken to the user — never reach this method and never
        # land in conversation history. That separation is the explicit
        # fix for: "tool result strings stored as assistant turns cause
        # the LLM to repeat the same tool on unrelated follow-ups."
        #
        # Mixed case (LLM narration + tool call in the same response):
        # store the narration portion only. The router never sees the
        # tool result string here, so this guard alone keeps tool output
        # out of history.
        #
        # CancelledError raised from inside the for loop above skips
        # this line entirely, so partial assistant text never pollutes
        # history either.
        if full_text_parts and not tool_calls_seen:
            self._conv.add_assistant_turn("".join(full_text_parts))

    # -- internal --

    def _try_pattern(self, transcription: str) -> Intent | None:
        normalized = _normalize(transcription)
        if not normalized:
            return None
        if _STOP_PATTERN.match(normalized):
            return StopIntent()
        for pattern, builder in self._patterns:
            m = pattern.match(normalized)
            if m is None:
                continue
            intent = builder(m)
            # If the pattern produced a ToolIntent whose tool isn't
            # registered (config disabled it, or the pattern table is
            # ahead of the registry), fall through to the LLM rather
            # than dispatch an "unknown tool" error. The LLM may have
            # a different path to satisfy the request, or it'll explain
            # the limitation in-character.
            if (
                self._registry is not None
                and isinstance(intent, ToolIntent)
                and self._registry.get(intent.tool_name) is None
            ):
                log.info(
                    "pattern matched tool %r but it is not registered; "
                    "falling through to LLM",
                    intent.tool_name,
                )
                return None
            return intent
        return None

    def _tool_intent_from(self, tool_call: dict) -> ToolIntent | None:
        """Convert an Ollama tool_call dict into a ToolIntent. Tolerant
        of the schema variations across Ollama versions: function.name +
        function.arguments where arguments may be a dict or a JSON
        string."""
        try:
            fn = tool_call.get("function") or {}
            name = fn.get("name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    log.warning(
                        "tool_call arguments not valid JSON: %r", args[:200]
                    )
                    return None
            if not name:
                log.warning("tool_call missing name: %r", tool_call)
                return None
            return ToolIntent(tool_name=name, args=args or {})
        except Exception:
            log.exception("malformed tool_call: %r", tool_call)
            return None


# --- intent → speech adapter ----------------------------------------


# Spoken fallback when a tool succeeds but returns no human-facing
# output. Kept short and in-persona.
_GENERIC_OK = "Done, sir."
# Spoken fallback when a tool returns no error string either. Most
# tools do populate .error; this guards the rare "False and silent" case
# rather than spitting an empty string at the user.
_GENERIC_FAIL = "I couldn't do that, sir."

# Characters that PiperTTS's sentence-boundary regex treats as terminators.
# Without one of these the speak_stream buffer waits up to max_wait_seconds
# before flushing, so two back-to-back tool outputs run together.
_SENTENCE_TERMINATORS = (".", "!", "?")

# Markdown that the LLM sometimes emits in SpeakIntent content (link
# syntax, emphasis, code spans). TTS reads it literally — "[Google]
# (https://google.com)" became "open bracket Google close bracket open
# paren h-t-t-p-s..." in live testing. Strip the markdown punctuation
# while keeping the readable inner text.
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_MD_CODE = re.compile(r"`([^`]+)`")


def _scrub_for_speech(text: str) -> str:
    """Strip common Markdown punctuation from text destined for TTS.
    Applied in execute_intent so every Intent-yielded chunk benefits."""
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_CODE.sub(r"\1", text)
    return text


def _ensure_sentence_terminator(text: str) -> str:
    """Append '. ' to text that doesn't already end on a sentence
    terminator, so consecutive ToolIntent outputs become separate
    sentences in PiperTTS's segmenter. Trailing whitespace is normalised
    first because outputs sometimes carry a trailing newline from
    psutil / pathlib that defeats the suffix check."""
    stripped = text.rstrip()
    if not stripped:
        return text
    if stripped[-1] in _SENTENCE_TERMINATORS:
        return stripped + " "
    return stripped + ". "


async def execute_intent(
    intent: Intent, registry: ToolRegistry | None
) -> AsyncIterator[str]:
    """Convert one Intent into spoken text chunks.

    SpeakIntent yields its text verbatim. ToolIntent dispatches through
    the registry and speaks the result (output on success, error on
    failure, persona fallback if both are absent). If
    intent.spoken_response is set, that line is spoken *before* the
    tool runs — useful for actions where the LLM wants to narrate the
    action before any output appears ("Locking the screen now, sir.").

    A None registry on a ToolIntent yields an error message — the
    bridge between Intent and execution should always be wired in the
    composition root, so a None here is a configuration bug, not a
    user-visible failure mode. CompoundIntent recurses across its
    children in order."""
    if isinstance(intent, StopIntent):
        return  # silence — no TTS, pipeline transitions to IDLE via zero-yield path
    if isinstance(intent, SpeakIntent):
        yield _scrub_for_speech(intent.text)
        return
    if isinstance(intent, ToolIntent):
        if intent.spoken_response:
            yield _ensure_sentence_terminator(
                _scrub_for_speech(intent.spoken_response)
            )
        if registry is None:
            yield _GENERIC_FAIL
            log.error("execute_intent: no registry wired for ToolIntent")
            return
        result = await registry.execute(intent.tool_name, intent.args)
        if result.success:
            if isinstance(result.output, str) and result.output:
                yield _ensure_sentence_terminator(
                    _scrub_for_speech(result.output)
                )
            elif result.output is None:
                yield _GENERIC_OK
            else:
                # dict output: the LLM/UI consumes structured data; spoken
                # layer just confirms the action happened.
                yield _GENERIC_OK
        else:
            yield _ensure_sentence_terminator(
                _scrub_for_speech(result.error or _GENERIC_FAIL)
            )
        return
    if isinstance(intent, CompoundIntent):
        for sub in intent.intents:
            async for chunk in execute_intent(sub, registry):
                yield chunk
        return
