"""Tests for jarvis.llm.intent_router.IntentRouter.

Coverage:
- Pattern layer: every SPEC pattern, filler stripping, partial misses.
- LLM layer: streaming SpeakIntents per chunk, tool calls -> ToolIntents,
  conversation history updated only on successful completion.
- Cancellation: stream cancel propagates to OllamaClient (its own tests
  verify the httpx-level contract); router skips add_assistant_turn but
  preserves the user turn.
"""

from __future__ import annotations

import asyncio
import datetime
from urllib.parse import quote_plus

import pytest

from jarvis.core.config import ToolsConfig
from jarvis.llm.conversation import Conversation
from jarvis.llm.intent_router import (
    _STOP_PATTERN,
    CompoundIntent,
    IntentRouter,
    SpeakIntent,
    StopIntent,
    ToolIntent,
    _normalize,
    execute_intent,
)
from jarvis.llm.ollama_client import ChatChunk
from jarvis.tools.registry import EmptyArgs, ToolRegistry, ToolResult

# --- helpers ---------------------------------------------------------


class FakeOllama:
    """A scriptable fake of OllamaClient. stream_chat() returns the
    canned ChatChunk sequence. Tracks which messages and tools were
    passed in."""

    def __init__(self, chunks: list[ChatChunk] | None = None) -> None:
        self._chunks = chunks or []
        self.last_messages: list[dict] | None = None
        self.last_tools: list[dict] | None = None
        self.stream_calls = 0

    async def stream_chat(self, messages, *, tools=None):
        self.stream_calls += 1
        self.last_messages = messages
        self.last_tools = tools
        for c in self._chunks:
            yield c


class _Conv:
    """A tracking Conversation stand-in. Records calls so tests can
    verify the order and content."""

    def __init__(self, system_prompt: str = "sys") -> None:
        self.user_turns: list[str] = []
        self.assistant_turns: list[str] = []
        self._messages_for_user: dict[str, list[dict]] = {}
        self._sp = system_prompt

    def add_user_turn(self, text: str) -> list[dict]:
        self.user_turns.append(text)
        msgs = [
            {"role": "system", "content": self._sp},
            {"role": "user", "content": text},
        ]
        self._messages_for_user[text] = msgs
        return msgs

    def add_assistant_turn(self, text: str) -> None:
        self.assistant_turns.append(text)

    def current_messages(self):
        """Return [system, *interleaved user/assistant turns]. Used by
        the end-to-end contract test that asserts tool result strings
        never end up in conversation history."""
        msgs = [{"role": "system", "content": self._sp}]
        # Interleave in append order: each user_turn was appended via
        # add_user_turn, each assistant via add_assistant_turn; the
        # tests that use this rely on the recorded order matching the
        # actual call sequence.
        for u, a in zip(self.user_turns, self.assistant_turns, strict=False):
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        # Trailing user turn with no assistant response yet.
        if len(self.user_turns) > len(self.assistant_turns):
            msgs.append(
                {"role": "user", "content": self.user_turns[-1]}
            )
        return msgs


def _content_chunk(text: str, *, done: bool = False) -> ChatChunk:
    return ChatChunk(content=text, done=done)


def _tool_chunk(tool_calls: list[dict]) -> ChatChunk:
    return ChatChunk(content="", tool_calls=tuple(tool_calls), done=True)


def _make_router(
    *,
    chunks: list[ChatChunk] | None = None,
    tools: list[dict] | None = None,
    fixed_time: datetime.datetime | None = None,
) -> tuple[IntentRouter, FakeOllama, _Conv]:
    llm = FakeOllama(chunks=chunks)
    conv = _Conv()
    time_provider = (lambda: fixed_time) if fixed_time else datetime.datetime.now
    r = IntentRouter(
        llm=llm, conversation=conv, tools=tools, time_provider=time_provider,
    )
    return r, llm, conv


async def _collect(gen) -> list:
    return [item async for item in gen]


# --- normalization / filler stripping -----------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Open Spotify", "open spotify"),
        ("OPEN SPOTIFY!", "open spotify"),
        ("  open  spotify  ", "open  spotify"),
        ("hey jarvis open spotify", "open spotify"),
        ("hey jarvis, open spotify", "open spotify"),
        ("can you open spotify", "open spotify"),
        ("can you please open spotify", "open spotify"),
        ("could you please open spotify", "open spotify"),
        ("would you open spotify", "open spotify"),
        ("please open spotify", "open spotify"),
        # Iterative stripping: hey jarvis + can you please + open
        ("hey jarvis can you please open spotify", "open spotify"),
        # Trailing punctuation
        ("open spotify.", "open spotify"),
        ("open spotify?", "open spotify"),
        # Empty / whitespace
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize(raw: str, expected: str):
    assert _normalize(raw) == expected


# --- pattern layer parametrized ----------------------------------


@pytest.mark.parametrize(
    "transcription,expected_tool,expected_args",
    [
        # open <app>
        ("open spotify", "open_app", {"name": "spotify"}),
        ("Open Chrome", "open_app", {"name": "chrome"}),
        ("hey jarvis can you please open spotify", "open_app", {"name": "spotify"}),
        ("open my browser", "open_app", {"name": "my browser"}),
        # search / google
        ("search dogs", "open_url",
         {"url": f"https://www.google.com/search?q={quote_plus('dogs')}"}),
        ("google jarvis ai", "open_url",
         {"url": f"https://www.google.com/search?q={quote_plus('jarvis ai')}"}),
        ("search for the weather", "open_url",
         {"url": f"https://www.google.com/search?q={quote_plus('the weather')}"}),
        ("google for cats", "open_url",
         {"url": f"https://www.google.com/search?q={quote_plus('cats')}"}),
        # "search up X" — colloquial; "up" should be stripped before the
        # query, not included in it.
        ("search up pasta", "open_url",
         {"url": f"https://www.google.com/search?q={quote_plus('pasta')}"}),
        # Bare query (no "up"/"for") — the query is whatever follows.
        ("search recipes", "open_url",
         {"url": f"https://www.google.com/search?q={quote_plus('recipes')}"}),
        # Word boundary: "forty" must NOT be misread as "for" + "ty".
        ("google forty winks", "open_url",
         {"url": f"https://www.google.com/search?q={quote_plus('forty winks')}"}),
        # volume
        ("volume up", "volume", {"action": "up"}),
        ("volume down", "volume", {"action": "down"}),
        ("volume mute", "volume", {"action": "mute"}),
        ("Volume Unmute", "volume", {"action": "unmute"}),
        # screenshot
        ("screenshot", "screenshot", {}),
        ("take a screenshot", "screenshot", {}),
        ("take screenshot", "screenshot", {}),
        # lock
        ("lock the screen", "lock_screen", {}),
        ("lock my pc", "lock_screen", {}),
        ("lock pc", "lock_screen", {}),
        ("lock screen", "lock_screen", {}),
        # deep research (before quick research)
        ("deep research quantum computing", "deep_research", {"query": "quantum computing"}),
        ("do deep research on climate change", "deep_research", {"query": "climate change"}),
        ("jarvis deep research black holes", "deep_research", {"query": "black holes"}),
        ("enable deep research ultra", "enable_deep_research_ultra", {}),
        ("turn on ultra research", "enable_deep_research_ultra", {}),
        ("disable deep research ultra", "disable_deep_research_ultra", {}),
        ("use normal deep research", "disable_deep_research_ultra", {}),
        ("pause deep research", "pause_deep_research", {}),
        ("resume deep research", "resume_deep_research", {}),
        ("delete deep research on solar power", "delete_deep_research", {"query": "solar power"}),
        ("remove deep research about ai safety", "delete_deep_research", {"query": "ai safety"}),
        ("delete the deep research", "delete_deep_research", {"query": ""}),
        ("delete all deep research", "delete_all_deep_research", {}),
        ("clear all deep research history", "delete_all_deep_research", {}),
        # dashboard
        ("show dashboard", "show_dashboard", {}),
        ("open the dashboard", "show_dashboard", {}),
        # regression: "open my dashboard" used to be stolen by the
        # catch-all `open <X>` pattern and routed to open_app.
        ("open my dashboard", "show_dashboard", {}),
        ("show my dashboard", "show_dashboard", {}),
        ("bring up dashboard", "show_dashboard", {}),
        ("show system stats", "show_dashboard", {}),
        ("show me my system stats", "show_dashboard", {}),
        ("close dashboard", "close_dashboard", {}),
        ("close the dashboard", "close_dashboard", {}),
        ("close my dashboard", "close_dashboard", {}),
        # help
        ("what can you do", "open_help", {}),
        ("what can i say", "open_help", {}),
        ("show help", "open_help", {}),
        ("open my help", "open_help", {}),
        ("show my capabilities", "open_help", {}),
        # notes
        ("open my notes", "open_notes", {}),
        ("show notes", "open_notes", {}),
        ("bring up my notes", "open_notes", {}),
        ("close notes", "close_notes", {}),
        ("close my notes", "close_notes", {}),
        (
            "take a note about the meeting being moved",
            "take_note",
            {"content": "the meeting being moved"},
        ),
        ("jot down buy milk", "take_note", {"content": "buy milk"}),
        (
            "write this down: project Phoenix kicks off Monday",
            "take_note",
            {"content": "project phoenix kicks off monday"},
        ),
        ("read this note", "read_note", {"title": ""}),
        ("delete this note", "delete_note", {"title": ""}),
        ("delete the groceries note", "delete_note", {"title": "groceries"}),
        (
            "add this to my meeting note: agenda finalized",
            "append_to_note",
            {"title": "meeting", "content": "agenda finalized"},
        ),
        # clipboard history
        ("show clipboard history", "show_clipboard_history", {}),
        ("show my clipboard history", "show_clipboard_history", {}),
        ("open my clipboard history", "show_clipboard_history", {}),
        ("bring up clipboard history", "show_clipboard_history", {}),
        ("what have i copied", "show_clipboard_history", {}),
        ("close clipboard history", "close_clipboard_history", {}),
        ("close my clipboard history", "close_clipboard_history", {}),
        ("clear my clipboard history", "clear_clipboard_history", {}),
        ("clear the clipboard history", "clear_clipboard_history", {}),
        ("paste my last copy", "paste_clipboard_item", {"index": 1}),
        ("paste item 3", "paste_clipboard_item", {"index": 3}),
        # logs
        ("show logs", "show_logs", {}),
        ("open the log", "show_logs", {}),
        ("open my logs", "show_logs", {}),
        ("show me the logs", "show_logs", {}),
        ("show errors", "show_logs", {}),
        ("close logs", "close_logs", {}),
        ("close the log", "close_logs", {}),
        # research panel
        ("research quantum computing", "research", {"query": "quantum computing"}),
        ("look up black holes", "research", {"query": "black holes"}),
        ("close research", "close_research", {}),
        ("read more", "read_more", {}),
        ("copy that", "copy_research", {}),
        # workspace
        ("open my workspace", "launch_workspace", {}),
        ("launch workspace", "launch_workspace", {}),
        ("start my workspace", "launch_workspace", {}),
        ("jarvis open my workspace", "launch_workspace", {}),
        # see screen (vision)
        ("look at my screen", "see_screen", {}),
        ("look at the screen", "see_screen", {}),
        ("see my screen", "see_screen", {}),
        ("can you see my screen", "see_screen", {}),
        ("read my screen", "see_screen", {}),
        ("describe my screen", "see_screen", {}),
        ("describe the display", "see_screen", {}),
        ("what's on my screen", "see_screen", {}),
        ("what is on the screen", "see_screen", {}),
        ("what do you see", "see_screen", {}),
        ("what can you see", "see_screen", {}),
        ("what do you see on my screen", "see_screen", {}),
        ("what can you see on the screen", "see_screen", {}),
        # filler-stripped: "hey jarvis, what's on my screen"
        ("hey jarvis what's on my screen", "see_screen", {}),
        # filler + can-you: "can you see my screen" -> "see my screen"
        ("hey jarvis can you see my screen", "see_screen", {}),
    ],
)
async def test_pattern_yields_correct_tool_intent(
    transcription, expected_tool, expected_args
):
    router, llm, conv = _make_router()
    intents = await _collect(router.route(transcription))
    assert len(intents) == 1
    assert isinstance(intents[0], ToolIntent)
    assert intents[0].tool_name == expected_tool
    assert intents[0].args == expected_args
    # Pattern hit means LLM and conversation untouched.
    assert llm.stream_calls == 0
    assert conv.user_turns == []


async def test_google_search_pattern_constructs_search_url_not_homepage():
    """Live bug: 'Google how to make pasta' was opening the homepage.
    Asserts the URL is the search endpoint with the query, quote_plus-
    encoded so Google sees the canonical '+'-for-spaces form."""
    router, _, _ = _make_router()
    intents = await _collect(router.route("google how to make pasta"))
    assert len(intents) == 1
    assert isinstance(intents[0], ToolIntent)
    assert intents[0].tool_name == "open_url"
    assert (
        intents[0].args["url"]
        == "https://www.google.com/search?q=how+to+make+pasta"
    )


async def test_what_time_yields_speak_intent_with_formatted_time():
    fixed = datetime.datetime(2026, 5, 9, 15, 47, 0)
    router, llm, conv = _make_router(fixed_time=fixed)
    intents = await _collect(router.route("what time is it"))
    assert len(intents) == 1
    assert isinstance(intents[0], SpeakIntent)
    assert "3:47 PM" in intents[0].text
    assert "sir" in intents[0].text
    assert llm.stream_calls == 0


@pytest.mark.parametrize(
    "phrasing",
    ["what's the time", "what is the time", "what time is it", "what's time"],
)
async def test_time_pattern_phrasings(phrasing: str):
    fixed = datetime.datetime(2026, 5, 9, 9, 5, 0)
    router, _, _ = _make_router(fixed_time=fixed)
    intents = await _collect(router.route(phrasing))
    assert len(intents) == 1
    assert isinstance(intents[0], SpeakIntent)


# --- pattern layer: misses fall through to LLM ----------------


@pytest.mark.parametrize(
    "transcription",
    [
        "what is the capital of france",
        "tell me a joke",
        "open",  # bare "open" with no app -- shouldn't match
        "",
        "   ",
        "remind me to walk the dog",
    ],
)
async def test_misses_fall_through_to_llm(transcription: str):
    router, llm, conv = _make_router(
        chunks=[
            _content_chunk("LLM response"),
            _content_chunk("", done=True),
        ],
    )
    intents = await _collect(router.route(transcription))
    assert llm.stream_calls == 1
    assert any(isinstance(i, SpeakIntent) for i in intents)


# --- LLM layer: streaming chunks --------------------------------


async def test_llm_yields_one_speak_intent_per_content_chunk():
    chunks = [
        _content_chunk("Hello"),
        _content_chunk(" "),
        _content_chunk("sir"),
        _content_chunk(".", done=True),
    ]
    router, llm, conv = _make_router(chunks=chunks)
    intents = await _collect(router.route("are you there"))
    assert all(isinstance(i, SpeakIntent) for i in intents)
    assert [i.text for i in intents] == ["Hello", " ", "sir", "."]


async def test_llm_skips_empty_content_chunks():
    """An empty-content chunk shouldn't yield an empty SpeakIntent --
    that would push an empty string into TTS for no reason."""
    chunks = [
        _content_chunk(""),
        _content_chunk("Hello"),
        _content_chunk("", done=True),
    ]
    router, _, _ = _make_router(chunks=chunks)
    intents = await _collect(router.route("hi"))
    speaks = [i for i in intents if isinstance(i, SpeakIntent)]
    assert [i.text for i in speaks] == ["Hello"]


async def test_llm_passes_messages_from_conversation():
    chunks = [_content_chunk("ok", done=True)]
    router, llm, conv = _make_router(chunks=chunks)
    await _collect(router.route("tell me about birds"))
    assert llm.last_messages is not None
    # User turn was added to conversation and forwarded to LLM.
    assert conv.user_turns == ["tell me about birds"]
    user_msg = next(
        (m for m in llm.last_messages if m["role"] == "user"), None
    )
    assert user_msg is not None
    assert user_msg["content"] == "tell me about birds"


async def test_llm_passes_tools_through():
    tools = [{"type": "function", "function": {"name": "screenshot"}}]
    chunks = [_content_chunk("ok", done=True)]
    router, llm, _ = _make_router(chunks=chunks, tools=tools)
    await _collect(router.route("anything"))
    assert llm.last_tools == tools


async def test_assistant_turn_recorded_with_assembled_full_text():
    chunks = [
        _content_chunk("Hello "),
        _content_chunk("there, "),
        _content_chunk("sir."),
        _content_chunk("", done=True),
    ]
    router, _, conv = _make_router(chunks=chunks)
    await _collect(router.route("hi"))
    assert conv.assistant_turns == ["Hello there, sir."]


async def test_no_assistant_turn_when_llm_yields_only_empty_content():
    chunks = [_content_chunk("", done=True)]
    router, _, conv = _make_router(chunks=chunks)
    await _collect(router.route("hi"))
    assert conv.assistant_turns == []
    # User turn still added.
    assert conv.user_turns == ["hi"]


# --- LLM layer: tool calls --------------------------------------


async def test_tool_call_in_final_chunk_yields_tool_intent():
    tool_call = {
        "function": {"name": "screenshot", "arguments": {}},
    }
    chunks = [_tool_chunk([tool_call])]
    router, _, _ = _make_router(chunks=chunks)
    intents = await _collect(router.route("take a picture"))
    assert len(intents) == 1
    assert isinstance(intents[0], ToolIntent)
    assert intents[0].tool_name == "screenshot"
    assert intents[0].args == {}


async def test_play_and_launch_phrases_use_llm_not_pattern():
    """Music and game requests are not pattern-matched; the LLM chooses
    play_youtube_music vs launch_steam_game from context."""
    router, llm, conv = _make_router()
    for phrase in (
        "play bohemian rhapsody",
        "launch elden ring on steam",
        "play some jazz",
    ):
        llm.stream_calls = 0
        conv.user_turns.clear()
        await _collect(router.route(phrase))
        assert llm.stream_calls == 1, phrase


async def test_multiple_tool_calls_yield_multiple_tool_intents():
    # NOTE: input deliberately doesn't match any pattern -- compound
    # commands like "open X and do Y" can hit the greedy "open <app>"
    # pattern (which would capture the whole tail as the app name).
    # The LLM-emits-multiple-tool_calls path is what we're testing here.
    tool_calls = [
        {"function": {"name": "open_app", "arguments": {"name": "spotify"}}},
        {"function": {"name": "weather", "arguments": {}}},
    ]
    chunks = [_tool_chunk(tool_calls)]
    router, _, _ = _make_router(chunks=chunks)
    intents = await _collect(
        router.route("play some music and tell me the weather")
    )
    tools = [i for i in intents if isinstance(i, ToolIntent)]
    assert [t.tool_name for t in tools] == ["open_app", "weather"]
    assert tools[0].args == {"name": "spotify"}


async def test_duplicate_tool_calls_in_one_response_are_deduped(caplog):
    """LLM sometimes returns the same tool call twice. The router must
    yield only the first and log the skip — the pipeline must never
    execute duplicate tool calls in a single response."""
    import logging
    tool_call = {"function": {"name": "screenshot", "arguments": {}}}
    chunks = [_tool_chunk([tool_call, tool_call])]
    router, _, _ = _make_router(chunks=chunks)
    with caplog.at_level(logging.INFO, logger="jarvis.llm.intent_router"):
        intents = await _collect(router.route("take two screenshots"))
    tools = [i for i in intents if isinstance(i, ToolIntent)]
    assert len(tools) == 1
    assert tools[0].tool_name == "screenshot"
    assert any("skipped duplicate" in r.message for r in caplog.records)


async def test_mixed_text_and_tool_calls_suppresses_narration():
    """When the LLM emits tool calls, narration is not spoken — only the
    tool confirmation (once), avoiding duplicate TTS."""
    tool_call = {"function": {"name": "screenshot", "arguments": {}}}
    chunks = [
        _content_chunk("Taking that "),
        _content_chunk("now."),
        ChatChunk(content="", tool_calls=(tool_call,), done=True),
    ]
    router, _, conv = _make_router(chunks=chunks)
    intents = await _collect(router.route("snap"))
    speaks = [i for i in intents if isinstance(i, SpeakIntent)]
    tools = [i for i in intents if isinstance(i, ToolIntent)]
    assert speaks == []
    assert len(tools) == 1
    assert tools[0].tool_name == "screenshot"
    assert conv.assistant_turns == []


async def test_tool_call_args_as_json_string_is_decoded():
    """Some Ollama versions/models emit arguments as a JSON-encoded
    string instead of a dict. The router decodes."""
    tool_call = {
        "function": {"name": "open_app", "arguments": '{"name": "spotify"}'},
    }
    chunks = [_tool_chunk([tool_call])]
    router, _, _ = _make_router(chunks=chunks)
    intents = await _collect(router.route("anything"))
    assert intents[0].tool_name == "open_app"
    assert intents[0].args == {"name": "spotify"}


async def test_tool_call_with_invalid_json_args_dropped(caplog):
    import logging
    tool_call = {"function": {"name": "open_app", "arguments": "{invalid"}}
    chunks = [_tool_chunk([tool_call])]
    router, _, _ = _make_router(chunks=chunks)
    with caplog.at_level(logging.WARNING, logger="jarvis.llm.intent_router"):
        intents = await _collect(router.route("anything"))
    tools = [i for i in intents if isinstance(i, ToolIntent)]
    assert tools == []
    assert any("not valid JSON" in r.message for r in caplog.records)


async def test_tool_call_missing_name_dropped(caplog):
    import logging
    tool_call = {"function": {"arguments": {}}}
    chunks = [_tool_chunk([tool_call])]
    router, _, _ = _make_router(chunks=chunks)
    with caplog.at_level(logging.WARNING, logger="jarvis.llm.intent_router"):
        intents = await _collect(router.route("anything"))
    tools = [i for i in intents if isinstance(i, ToolIntent)]
    assert tools == []
    assert any("missing name" in r.message for r in caplog.records)


# --- Conversation injection ---------------------------------------


async def test_conversation_is_injected_state_persists_across_routes():
    """Two route() calls should both see the same Conversation -- the
    second call's user_turns list should have both entries."""
    conv = Conversation(system_prompt_provider=lambda: "sys")
    llm = FakeOllama(chunks=[_content_chunk("ok", done=True)])
    router = IntentRouter(llm=llm, conversation=conv)

    await _collect(router.route("first question"))
    await _collect(router.route("second question"))

    msgs = conv.current_messages()
    user_contents = [m["content"] for m in msgs if m["role"] == "user"]
    assert "first question" in user_contents
    assert "second question" in user_contents


# --- cancellation ---------------------------------------------------


async def test_cancellation_skips_assistant_turn_keeps_user_turn():
    """Tripwire for the documented tradeoff: cancelled streams DO NOT
    record an assistant turn (would pollute history with partial text)
    but DO keep the user turn (the user said it; that's history)."""
    started = asyncio.Event()
    release = asyncio.Event()

    class StallingLLM:
        last_messages = None
        last_tools = None
        stream_calls = 0

        async def stream_chat(self, messages, *, tools=None):
            self.stream_calls += 1
            self.last_messages = messages
            self.last_tools = tools
            yield _content_chunk("First ")
            started.set()
            # Block here until the test releases us; if cancelled, the
            # CancelledError propagates out of this generator.
            await release.wait()
            yield _content_chunk("rest", done=True)

    llm = StallingLLM()
    conv = _Conv()
    router = IntentRouter(llm=llm, conversation=conv)

    received = []

    async def consume():
        async for intent in router.route("are you there"):
            received.append(intent)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # User turn IS preserved (the user said the thing).
    assert conv.user_turns == ["are you there"]
    # Assistant turn is NOT recorded (partial response would pollute history).
    assert conv.assistant_turns == []
    # Text is buffered until the stream completes; cancellation drops it.
    assert received == []


# --- type re-exports / sanity -------------------------------------


# --- registry awareness ---------------------------------------------


def _make_router_with_registry(
    *,
    registry: ToolRegistry,
    chunks: list[ChatChunk] | None = None,
) -> tuple[IntentRouter, FakeOllama, _Conv]:
    llm = FakeOllama(chunks=chunks)
    conv = _Conv()
    r = IntentRouter(llm=llm, conversation=conv, registry=registry)
    return r, llm, conv


class _PingTool:
    name: str = "ping"
    description: str = "Returns pong."
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    async def execute(self, args: EmptyArgs) -> ToolResult:
        return ToolResult(success=True, output="pong, sir.")


async def test_registry_provides_tools_to_llm_fresh_per_call():
    """The router queries registry.as_openai_functions per route() call
    so a registry edit between turns lands on the next turn."""
    reg = ToolRegistry(ToolsConfig())
    reg.register(_PingTool())
    chunks = [_content_chunk("ok", done=True)]
    router, llm, _ = _make_router_with_registry(registry=reg, chunks=chunks)
    await _collect(router.route("tell me a joke"))
    assert llm.last_tools is not None
    assert any(t["function"]["name"] == "ping" for t in llm.last_tools)


async def test_pattern_falls_through_when_tool_not_registered():
    """`open_app` is in the pattern table. If the registry has no
    `open_app` tool registered (e.g. config disabled it before reach),
    the pattern must NOT produce a ToolIntent we'd then fail to execute;
    instead fall through to the LLM. Defends against a confusing
    "I tried to do that but the tool doesn't exist" error."""
    reg = ToolRegistry(ToolsConfig())  # no tools registered
    chunks = [_content_chunk("LLM path used", done=True)]
    router, llm, _ = _make_router_with_registry(registry=reg, chunks=chunks)
    intents = await _collect(router.route("open spotify"))
    # LLM was consulted (it would NOT have been if the pattern matched).
    assert llm.stream_calls == 1
    assert [type(i).__name__ for i in intents] == ["SpeakIntent"]


async def test_pattern_used_when_tool_is_registered():
    """Sanity check on the inverse: with the tool registered, the
    pattern wins and the LLM is not consulted."""
    class _OpenApp:
        name = "open_app"
        description = "Open an app."
        args_schema = EmptyArgs
        requires_confirmation = False

        async def execute(self, args):  # pragma: no cover
            return ToolResult(success=True)

    reg = ToolRegistry(ToolsConfig())
    reg.register(_OpenApp())
    router, llm, _ = _make_router_with_registry(registry=reg)
    intents = await _collect(router.route("open spotify"))
    assert llm.stream_calls == 0
    assert isinstance(intents[0], ToolIntent)
    assert intents[0].tool_name == "open_app"


# --- execute_intent --------------------------------------------------


async def _collect_strings(gen) -> list[str]:
    return [c async for c in gen]


class _OkTool:
    name: str = "ok_tool"
    description: str = "Always returns a string."
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, output: str | None = "tool said hi") -> None:
        self._output = output

    async def execute(self, args: EmptyArgs) -> ToolResult:
        return ToolResult(success=True, output=self._output)


class _FailTool:
    name: str = "fail_tool"
    description: str = "Always fails."
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self, error: str | None = "things went wrong") -> None:
        self._error = error

    async def execute(self, args: EmptyArgs) -> ToolResult:
        return ToolResult(success=False, error=self._error)


async def test_execute_intent_speak_yields_text():
    chunks = await _collect_strings(execute_intent(SpeakIntent("hello"), None))
    assert chunks == ["hello"]


async def test_execute_intent_speak_strips_markdown_link_syntax():
    """LLM-emitted markdown was being read literally by TTS
    ("open bracket Google close bracket open paren h-t-t-p-s ..."). The
    scrubber unwraps [label](url) to just label."""
    chunks = await _collect_strings(
        execute_intent(
            SpeakIntent("Visit [Google](https://google.com) for that."),
            None,
        )
    )
    assert chunks == ["Visit Google for that."]


async def test_execute_intent_speak_strips_bold_italic_code():
    chunks = await _collect_strings(
        execute_intent(
            SpeakIntent("It is **important** to *try* the `print` command."),
            None,
        )
    )
    assert chunks == ["It is important to try the print command."]


async def test_execute_intent_tool_yields_output_string():
    reg = ToolRegistry(ToolsConfig())
    reg.register(_OkTool(output="screenshot saved, sir."))
    chunks = await _collect_strings(
        execute_intent(ToolIntent("ok_tool", {}), reg)
    )
    # Sentence-terminator normalisation appends a trailing space so the
    # PiperTTS speak_stream regex sees a sentence boundary between this
    # and the next ToolIntent output in a multi-tool response.
    assert chunks == ["screenshot saved, sir. "]


async def test_execute_intent_tool_with_none_output_yields_generic_ok():
    """A tool that succeeded but produced no string output must still
    speak SOMETHING so the user knows it ran."""
    reg = ToolRegistry(ToolsConfig())
    reg.register(_OkTool(output=None))
    chunks = await _collect_strings(
        execute_intent(ToolIntent("ok_tool", {}), reg)
    )
    assert chunks == ["Done, sir."]


async def test_execute_intent_tool_with_dict_output_yields_generic_ok():
    """Structured (dict) tool outputs are not spoken verbatim; the
    persona fallback runs so the user hears a confirmation."""
    class _StructTool:
        name = "struct_tool"
        description = "Returns a dict."
        args_schema = EmptyArgs
        requires_confirmation = False

        async def execute(self, args):
            return ToolResult(success=True, output={"k": "v"})

    reg = ToolRegistry(ToolsConfig())
    reg.register(_StructTool())
    chunks = await _collect_strings(
        execute_intent(ToolIntent("struct_tool", {}), reg)
    )
    assert chunks == ["Done, sir."]


async def test_execute_intent_tool_failure_yields_error_string():
    reg = ToolRegistry(ToolsConfig())
    reg.register(_FailTool(error="the printer is on fire"))
    chunks = await _collect_strings(
        execute_intent(ToolIntent("fail_tool", {}), reg)
    )
    assert chunks == ["the printer is on fire. "]


async def test_execute_intent_tool_failure_without_error_yields_persona_fallback():
    reg = ToolRegistry(ToolsConfig())
    reg.register(_FailTool(error=None))
    chunks = await _collect_strings(
        execute_intent(ToolIntent("fail_tool", {}), reg)
    )
    assert chunks == ["I couldn't do that, sir. "]


async def test_execute_intent_disabled_tool_surfaces_error_not_crash():
    """The user can disable a tool between as_openai_functions() and
    execute(); the spoken path must say something, not error silently."""
    reg = ToolRegistry(ToolsConfig(enabled={"ok_tool": False}))
    reg.register(_OkTool())
    chunks = await _collect_strings(
        execute_intent(ToolIntent("ok_tool", {}), reg)
    )
    assert len(chunks) == 1
    assert "disabled" in chunks[0]


async def test_execute_intent_unknown_tool_surfaces_error():
    reg = ToolRegistry(ToolsConfig())  # no tools registered
    chunks = await _collect_strings(
        execute_intent(ToolIntent("nonesuch", {}), reg)
    )
    assert len(chunks) == 1
    assert "unknown" in chunks[0]


async def test_execute_intent_spoken_response_precedes_tool_output():
    """ToolIntent.spoken_response is the LLM's narration of the action
    before any tool output. Both should be spoken, in order."""
    reg = ToolRegistry(ToolsConfig())
    reg.register(_OkTool(output="here's the data, sir."))
    intent = ToolIntent(
        "ok_tool", {}, spoken_response="One moment, sir."
    )
    chunks = await _collect_strings(execute_intent(intent, reg))
    assert chunks == ["One moment, sir. ", "here's the data, sir. "]


async def test_execute_intent_no_registry_on_tool_yields_fail():
    """Passing None registry on a ToolIntent is a composition-root bug
    rather than a user-facing failure; we still yield SOMETHING audible
    rather than break the speak stream silently."""
    chunks = await _collect_strings(
        execute_intent(ToolIntent("anything", {}), None)
    )
    assert len(chunks) == 1
    assert "couldn't" in chunks[0]


async def test_execute_intent_compound_recurses_across_children():
    reg = ToolRegistry(ToolsConfig())
    reg.register(_OkTool(output="part two."))
    compound = CompoundIntent(
        intents=(SpeakIntent("part one."), ToolIntent("ok_tool", {})),
    )
    chunks = await _collect_strings(execute_intent(compound, reg))
    # ToolIntent outputs are sentence-terminator-normalised so PiperTTS
    # pauses between back-to-back tool results (live multi-tool bug).
    assert chunks == ["part one.", "part two. "]


async def test_execute_intent_tool_output_gets_sentence_terminator():
    """Outputs that already end on a terminator are kept (with a trailing
    space); outputs without one get '. ' appended. PiperTTS's
    speak_stream segments on sentence boundaries — without this, two
    back-to-back tool results in one response run together."""
    reg = ToolRegistry(ToolsConfig())
    reg.register(_OkTool(output="screenshot saved"))  # no terminator
    chunks = await _collect_strings(
        execute_intent(ToolIntent("ok_tool", {}), reg)
    )
    assert chunks == ["screenshot saved. "]


async def test_execute_intent_tool_terminated_output_stays_terminated():
    reg = ToolRegistry(ToolsConfig())
    reg.register(_OkTool(output="Done, sir!"))
    chunks = await _collect_strings(
        execute_intent(ToolIntent("ok_tool", {}), reg)
    )
    # Existing terminator preserved; trailing space added for the
    # sentence-boundary regex.
    assert chunks == ["Done, sir! "]


async def test_tool_only_llm_response_stores_no_assistant_text():
    """When the LLM emits ONLY a tool call (no narration content), there
    is no LLM-generated text to record, so the assistant turn stays
    empty. The tool RESULT string ('CPU at 13 percent...') is produced
    downstream by execute_intent and never reaches the router — that
    separation is what keeps tool output out of conversation history."""
    tool_call = {"function": {"name": "system_stats", "arguments": {}}}
    chunks = [ChatChunk(content="", tool_calls=(tool_call,), done=True)]
    router, _, conv = _make_router(chunks=chunks)
    await _collect(router.route("what's CPU?"))
    assert conv.user_turns == ["what's CPU?"]
    assert conv.assistant_turns == []


async def test_mixed_text_and_tool_call_does_not_store_narration():
    """When the LLM emits BOTH narration and a tool call, narration is
    not stored — the tool result (via execute_intent) is the single
    spoken/history response."""
    tool_call = {"function": {"name": "system_stats", "arguments": {}}}
    chunks = [
        _content_chunk("Let me check, sir. "),
        ChatChunk(content="", tool_calls=(tool_call,), done=True),
    ]
    router, _, conv = _make_router(chunks=chunks)
    await _collect(router.route("what's CPU?"))
    assert conv.assistant_turns == []


async def test_tool_result_string_never_lands_in_conversation_history():
    """End-to-end contract: two consecutive turns where the first ends
    in a tool call must leave conversation history as
    [system, user1, user2] — no tool result string ('CPU at 13 percent
    ...') sneaks in as an assistant turn between them. This was the
    pattern that caused the LLM to re-fire system_stats on unrelated
    follow-ups in live testing."""
    # Turn 1: pure tool call (no narration content), simulating the
    # LLM choosing to act rather than chat.
    tool_call = {"function": {"name": "system_stats", "arguments": {}}}
    turn_1_chunks = [
        ChatChunk(content="", tool_calls=(tool_call,), done=True),
    ]
    # Turn 2: pure conversational answer (no tool call), the
    # next-turn "hello" case from the bug report.
    turn_2_chunks = [_content_chunk("Hello, sir.", done=True)]

    llm = FakeOllama(chunks=turn_1_chunks)
    conv = _Conv()
    router = IntentRouter(llm=llm, conversation=conv)

    await _collect(router.route("cpu"))
    # Mid-test: rebind the LLM's scripted chunks for turn 2.
    llm._chunks = turn_2_chunks
    await _collect(router.route("hello"))

    msgs = conv.current_messages()
    # Conversation should be: system, user1, user2, assistant("Hello, sir.").
    # Critically, NO 'CPU at 13 percent...' or any tool result string.
    contents = [m["content"] for m in msgs]
    assert all("CPU" not in c for c in contents), (
        f"tool result string leaked into conversation history: {contents}"
    )
    assert all("percent" not in c for c in contents), contents
    # Sanity: turn 2's LLM text WAS stored (proves the storage path
    # itself is alive; we're not silently dropping everything).
    assert "Hello, sir." in contents


async def test_tool_only_response_does_not_pollute_assistant_history():
    """The other side of the same coin: a tool-only response (no LLM
    narration) also must not leave any assistant turn behind."""
    tool_call = {"function": {"name": "screenshot", "arguments": {}}}
    chunks = [ChatChunk(content="", tool_calls=(tool_call,), done=True)]
    router, _, conv = _make_router(chunks=chunks)
    await _collect(router.route("take a shot"))
    assert conv.assistant_turns == []


async def test_text_only_response_still_recorded_as_assistant_history():
    """Inverse guard: pure conversational answers DO get stored so
    follow-up questions can reference them. Without this we'd lose
    'who wrote it' working after 'tell me about Hamlet'."""
    chunks = [
        _content_chunk("Hamlet is by Shakespeare."),
        _content_chunk("", done=True),
    ]
    router, _, conv = _make_router(chunks=chunks)
    await _collect(router.route("tell me about Hamlet"))
    assert conv.assistant_turns == ["Hamlet is by Shakespeare."]


async def test_router_logs_multi_tool_count(caplog):
    """The router logs when an LLM emits >1 tool_call in a single
    response so the operator can correlate live-test 'three tools fired
    at once' observations with the model output."""
    import logging
    chunks = [
        _tool_chunk([
            {"function": {"name": "system_stats", "arguments": {}}},
            {"function": {"name": "list_directory",
                          "arguments": {"path": "~"}}},
        ]),
    ]
    router, _, _ = _make_router(chunks=chunks)
    with caplog.at_level(logging.INFO, logger="jarvis.llm.intent_router"):
        await _collect(router.route("anything"))
    assert any(
        "ToolIntent count=2" in r.message for r in caplog.records
    )


def test_compound_intent_constructable():
    """Reserved type still works for callers that want to wrap intents
    explicitly. Router itself doesn't emit it in Phase 3."""
    inner = (SpeakIntent(text="a"), ToolIntent(tool_name="x", args={}))
    c = CompoundIntent(intents=inner)
    assert c.intents == inner


# --- StopIntent / stop pattern ----------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "stop",
        "Stop",
        "STOP",
        "shut up",
        "shut  up",
        "Shut Up",
        "quiet",
        "Quiet",
        "be quiet",
        "Be Quiet",
        "nevermind",
        "never mind",
        "never  mind",
        "cancel",
        "Cancel",
        "that's enough",
        "thats enough",
        "enough",
        "Enough",
    ],
)
def test_stop_pattern_matches_standalone_stop_commands(phrase: str):
    """Every listed stop phrase must match when normalized.
    _normalize lowercases and strips trailing punctuation."""
    from jarvis.llm.intent_router import _normalize
    assert _STOP_PATTERN.match(_normalize(phrase)) is not None, (
        f"{phrase!r} should have matched _STOP_PATTERN"
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "stop the music",
        "stop playing",
        "stop everything",
        "cancel that download",
        "be quiet for a minute",
        "never mind the weather",
        "enough already",
        "shut up and listen",
    ],
)
def test_stop_pattern_does_not_match_stop_with_object(phrase: str):
    """Stop phrases followed by an object ('stop the music') must NOT
    match — those are commands the LLM should handle."""
    from jarvis.llm.intent_router import _normalize
    assert _STOP_PATTERN.match(_normalize(phrase)) is None, (
        f"{phrase!r} should NOT have matched _STOP_PATTERN"
    )


@pytest.mark.parametrize(
    "transcription",
    [
        "stop",
        "hey jarvis, stop",
        "hey jarvis shut up",
        "quiet",
        "nevermind",
        "cancel",
        "enough",
    ],
)
async def test_stop_transcription_yields_stop_intent(transcription: str):
    """Router must yield exactly one StopIntent for stop phrases.
    LLM is never called; no user turn added to conversation history."""
    router, llm, conv = _make_router(
        chunks=[_content_chunk("should not appear", done=True)],
    )
    intents = await _collect(router.route(transcription))
    assert len(intents) == 1
    assert isinstance(intents[0], StopIntent)
    assert llm.stream_calls == 0
    assert conv.user_turns == []


async def test_execute_intent_stop_yields_nothing():
    """StopIntent must produce zero chunks from execute_intent.
    The pipeline's first_chunk_seen guard then transitions CS to IDLE."""
    chunks = await _collect_strings(execute_intent(StopIntent(), None))
    assert chunks == []


async def test_stop_does_not_pollute_conversation_history():
    """A stop command must leave conversation history untouched.
    Specifically: no user turn, no assistant turn."""
    conv = Conversation(system_prompt_provider=lambda: "sys")
    conv.add_user_turn("tell me about the weather")
    conv.add_assistant_turn("It's sunny, sir.")
    messages_before = list(conv.current_messages())

    llm = FakeOllama()
    router = IntentRouter(llm=llm, conversation=conv)
    await _collect(router.route("stop"))

    assert conv.current_messages() == messages_before


async def test_stop_takes_precedence_over_pattern_table():
    """'stop' must hit the stop handler, not fall through to the open_app
    pattern ('open' is not 'stop', but this guards against any future
    pattern that might greedily eat 'stop')."""
    router, llm, _ = _make_router()
    intents = await _collect(router.route("stop"))
    assert len(intents) == 1
    assert isinstance(intents[0], StopIntent)
    assert llm.stream_calls == 0
