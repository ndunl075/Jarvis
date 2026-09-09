"""Hybrid intent router: pattern layer for snap commands, LLM for the rest.

Two layers
----------
1. Pattern layer (deterministic, fast, no LLM call). First-match-wins
   table of (regex, intent_builder) tuples applied to the user's
   transcription after filler-word stripping. Misses fall through to
   the LLM. SPEC: this is what makes Jarvis feel snappy -- "open
   spotify" should never wait for a 7B-parameter model.

2. LLM layer. The transcription is added to the conversation history,
   OllamaClient.stream_chat() is invoked, and the model's text is
   buffered for the duration of one round (see "Double-speaking"
   below). Tool calls arrive batched on the final chunk; the
   accumulation point below is defensive against Ollama later
   streaming them per-chunk.

Tool-result feedback loop
-------------------------
The standard OpenAI/Ollama round trip, so the model can act on what a
tool returned instead of dispatching blind:

  1. Model returns tool_calls.
  2. The router executes them through the registry.
  3. Conversation records ONE assistant message carrying those
     tool_calls plus one role:"tool" message per call, correlated by
     tool_call_id (Conversation.add_tool_round_trip).
  4. The model is re-invoked with that history.
  5. Repeat until it returns no tool calls, bounded by
     max_tool_iterations (default 3 -- latency on a local 7B is the
     binding constraint, not correctness).

Hitting the bound is not an error path that hangs or loops: the final
round's tool results are spoken directly, so the user always hears
something.

The loop needs a registry (you cannot observe a result you cannot
execute). Without one -- or with max_tool_iterations == 1 -- the
router falls back to the legacy one-shot behaviour: yield ToolIntents
and let the caller execute and speak them. That fallback is also the
config escape hatch for anyone who prefers a tool's own polished
output over the model's paraphrase of it.

Double-speaking
---------------
Text is buffered for the whole of each round rather than yielded per
chunk, because tool calls arrive last: speaking narration as it
streams and only then discovering a tool call would produce
"Opening Chrome..." followed by the tool's own "Opening Chrome". So a
round that ends in tool calls speaks nothing -- neither the narration
nor the raw tool result -- and the model's next round, which has seen
the results, is the single utterance the user hears. Only the
bound-exhausted path speaks a raw tool result, and only because there
is no further model round to summarise it.

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

The feedback loop preserves that property by construction. Every await
inside a round -- the stream, and each registry.execute() -- happens
BEFORE anything is written to history; the assistant tool_calls
message and its role:"tool" results are appended together in one
synchronous call with no await between them, so a cancel can land
before the round trip or after it, never inside it.

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
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

from jarvis.llm.conversation import ToolExchange

if TYPE_CHECKING:
    from jarvis.llm.conversation import Conversation
    from jarvis.llm.ollama_client import OllamaClient
    from jarvis.tools.registry import ToolRegistry, ToolResult

log = logging.getLogger(__name__)

# Maximum LLM invocations per user turn. 3 buys the two-step
# interactions that matter ("check the weather, then act on it") while
# capping worst-case latency on a local 7B at three inferences. 1
# disables the feedback loop entirely and restores one-shot dispatch.
DEFAULT_MAX_TOOL_ITERATIONS = 3

# Cap on a single tool result fed back to the model. A directory
# listing or a web fetch can be tens of KB; a 7B's context is small and
# every extra token is latency on the NEXT inference too. Truncation is
# announced in-band so the model knows it is seeing a prefix rather
# than silently reasoning over half a listing.
MAX_TOOL_RESULT_CHARS = 4000
_TRUNCATION_NOTE = "\n... (result truncated)"


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
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ) -> None:
        # `registry` is the live source of the tool schemas — querying it
        # fresh on every route() call means a settings-driven enable/disable
        # takes effect on the very next turn (no rebuild of the router).
        # It is ALSO the executor for the tool-result feedback loop; a
        # router without one cannot observe results and so degrades to
        # one-shot dispatch (yield ToolIntents, caller executes).
        # `tools` is the legacy static-list path retained for the existing
        # test suite and any caller that wants to inject a curated schema
        # set; if both are provided, registry wins.
        self._llm = llm
        self._conv = conversation
        self._registry = registry
        self._static_tools = tools or []
        self._patterns = _build_patterns(time_provider)
        self._max_tool_iterations = max(1, int(max_tool_iterations))
        # Ollama does not attach ids to the tool calls it emits, so the
        # router mints them. Monotonic over the router's lifetime rather
        # than per-turn: conversation history outlives a turn, and two
        # round trips sharing "call_1" would let the trimming/orphan
        # filter pair a tool result with the wrong assistant message.
        self._tool_call_seq = 0

    def _current_tools(self) -> list[dict]:
        if self._registry is not None:
            return self._registry.as_openai_functions()
        return self._static_tools

    def _feedback_enabled(self) -> bool:
        """True when the router can run the tool-result loop itself."""
        return self._registry is not None and self._max_tool_iterations > 1

    def _next_call_id(self) -> str:
        self._tool_call_seq += 1
        return f"call_{self._tool_call_seq}"

    async def route(self, transcription: str) -> AsyncGenerator[Intent, None]:
        """Route a transcription into a stream of Intents.

        Typed as AsyncGenerator rather than AsyncIterator (it satisfies
        IntentProducer either way) so callers can aclose() it
        explicitly. A consumer cancelled mid-turn leaves this generator
        suspended inside stream_chat or a tool; closing it promptly is
        cheaper and more predictable than waiting for the GC.

        Pattern matches yield exactly one Intent and return — no LLM
        call, no added latency, unchanged by the feedback loop.

        LLM responses run the tool-result loop: up to
        max_tool_iterations model invocations, executing tool calls and
        feeding their results back between rounds. The Intents yielded
        are the SpeakIntents of whichever round finally answered
        without calling a tool. When the loop is disabled (no registry,
        or max_tool_iterations == 1) tool calls are yielded as
        ToolIntents for the caller to execute instead."""
        pattern_intent = self._try_pattern(transcription)
        if pattern_intent is not None:
            yield pattern_intent
            return

        messages = self._conv.add_user_turn(transcription)
        for iteration in range(1, self._max_tool_iterations + 1):
            text_parts, raw_tool_calls = await self._stream_round(messages)

            if not raw_tool_calls:
                # The model answered in prose. This is the only path
                # that speaks model text, and the only one that records
                # a plain assistant turn.
                for part in text_parts:
                    if part:
                        yield SpeakIntent(text=part)
                full_text = "".join(text_parts)
                if full_text:
                    self._conv.add_assistant_turn(full_text)
                return

            intents = self._tool_intents_from(raw_tool_calls, transcription)
            if not intents:
                # The model tried to call tools but every call was
                # malformed. Narration is still suppressed (it was
                # written to accompany an action that never happened)
                # and no assistant turn is recorded.
                return

            if not self._feedback_enabled():
                # Legacy one-shot dispatch: the caller executes and
                # speaks. Nothing is stored — the tool result never
                # passes through this method, which is what kept it out
                # of history before the loop existed.
                for intent in intents:
                    yield intent
                return

            # Execute first, record second. Every await lives in
            # _run_tools, so a cancel lands before the round trip is
            # written rather than halfway through it.
            exchanges, spoken = await self._run_tools(intents)
            messages = self._conv.add_tool_round_trip(
                content="".join(text_parts), exchanges=exchanges
            )

            if iteration >= self._max_tool_iterations:
                # Bound reached. Degrade by speaking the results we
                # already have rather than burning another inference or
                # going silent. The results are in history as role:"tool"
                # messages, so they are NOT also stored as assistant
                # text — that duplicate is exactly the shape that used
                # to make the model re-fire tools on follow-ups.
                log.warning(
                    "[router] tool feedback loop hit the %d-iteration "
                    "bound for %r; speaking raw tool output",
                    self._max_tool_iterations,
                    transcription,
                )
                if spoken:
                    yield SpeakIntent(text=spoken)
                return

    # -- internal --

    async def _stream_round(
        self, messages: list[dict]
    ) -> tuple[list[str], list[dict]]:
        """One LLM invocation. Returns (content chunks, raw tool calls).

        Text is accumulated rather than yielded because tool calls
        arrive on the final chunk — see the module's "Double-speaking"
        note. Chunk granularity is preserved in the returned list so
        the caller can hand PiperTTS the same token-sized pieces it
        would have got streaming.

        CancelledError propagates out untouched: no try/except here,
        and the caller has written nothing to history yet."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        async for chunk in self._llm.stream_chat(
            messages, tools=self._current_tools()
        ):
            if chunk.content:
                text_parts.append(chunk.content)
            if chunk.tool_calls:
                # Per current Ollama behavior, tool_calls arrive complete
                # on the final chunk. Defensive accumulation here is the
                # hook point for future per-chunk streaming -- if Ollama
                # adopts incremental tool-call emission, partial-call
                # assembly logic plugs in around this list.
                tool_calls.extend(chunk.tool_calls)
            if chunk.done:
                break
        return text_parts, tool_calls

    def _tool_intents_from(
        self, raw_tool_calls: list[dict], transcription: str
    ) -> list[ToolIntent]:
        """Parse, log and de-duplicate one round's tool calls."""
        # Visibility for live tool-decision debugging: every LLM-chosen
        # tool gets printed alongside the transcription that prompted it.
        # The pattern path doesn't show up here — those are router-level
        # decisions, not LLM ones. print (not log.info) keeps the trace
        # visible without bumping the global level, matching the rest of
        # the [router] / [boot] diagnostics.
        for tc in raw_tool_calls:
            fn = tc.get("function") or {}
            tool_name = fn.get("name") or "<missing>"
            print(
                f"[router] llm-chose-tool={tool_name} for={transcription!r}"
            )
        if len(raw_tool_calls) > 1:
            log.info(
                "[router] ToolIntent count=%d in one response",
                len(raw_tool_calls),
            )
        # De-duplication is per response, not across iterations: a model
        # that legitimately re-reads a file after writing it must be
        # allowed to. Cross-iteration repetition is what the iteration
        # bound is for.
        seen_tool_keys: set[str] = set()
        intents: list[ToolIntent] = []
        for tc in raw_tool_calls:
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
            intents.append(intent)
        return intents

    async def _run_tools(
        self, intents: list[ToolIntent]
    ) -> tuple[list[ToolExchange], str]:
        """Execute each ToolIntent; return the exchanges to record and
        the spoken form of the results (used only if the loop then hits
        its iteration bound).

        Writes nothing to conversation history — the caller appends the
        whole round trip atomically once every tool has finished, so a
        cancel mid-execution leaves history exactly as it was.
        registry.execute() never raises for tool-level failures (it
        returns ToolResult(success=False)), but does re-raise
        CancelledError, which is the behaviour this relies on."""
        registry = self._registry
        assert registry is not None  # guarded by _feedback_enabled()
        exchanges: list[ToolExchange] = []
        spoken: list[str] = []
        for intent in intents:
            result = await registry.execute(intent.tool_name, intent.args)
            exchanges.append(
                ToolExchange(
                    call_id=self._next_call_id(),
                    name=intent.tool_name,
                    arguments=intent.args,
                    result=_result_for_model(result),
                )
            )
            spoken.append(_result_for_speech(result))
        return exchanges, "".join(spoken)

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


def _result_for_model(result: ToolResult) -> str:
    """Stringify a ToolResult for the role:"tool" message the model reads.

    Deliberately NOT the same rendering as the spoken path. Speech
    collapses a dict result to "Done, sir." because a listener cannot
    consume structured data; the model can, and that structure is the
    whole point of "list my Downloads folder and tell me which one is
    the invoice" — so dicts are serialised as JSON rather than dropped.

    Failures are surfaced verbatim too. A model told "Error: unknown
    tool 'weathr'" can pick a different tool on the next iteration; one
    told nothing just calls the same broken tool again."""
    if not result.success:
        text = f"Error: {result.error or 'the tool failed'}"
    elif result.output is None:
        text = "Success (no output)."
    elif isinstance(result.output, str):
        text = result.output or "Success (no output)."
    else:
        try:
            text = json.dumps(result.output, default=str)
        except (TypeError, ValueError):
            text = str(result.output)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        return text[:MAX_TOOL_RESULT_CHARS] + _TRUNCATION_NOTE
    return text


def _result_for_speech(result: ToolResult) -> str:
    """Spoken rendering of a ToolResult, matching execute_intent's
    ToolIntent branch. Used only on the bound-exhausted path, where
    there is no further model round to summarise the result."""
    if not result.success:
        return _ensure_sentence_terminator(result.error or _GENERIC_FAIL)
    if isinstance(result.output, str) and result.output:
        return _ensure_sentence_terminator(result.output)
    # Terminated (unlike execute_intent's bare _GENERIC_OK) because the
    # bound-exhausted path may concatenate several results into one
    # SpeakIntent and PiperTTS segments on sentence boundaries.
    return _ensure_sentence_terminator(_GENERIC_OK)


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
