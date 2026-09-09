"""Tests for the tool-result feedback loop in jarvis.llm.intent_router.

The loop is the fix for "Jarvis cannot do anything that takes two
steps": the model calls a tool, the router executes it, records the
result as a role:"tool" message keyed by tool_call_id, and re-invokes
the model so it can act on what came back.

Coverage:
- Two-step interaction: the final answer depends on the tool result.
- Protocol shape: role:"tool" carries the id declared by the assistant
  message that requested it.
- Iteration bound: a model that calls tools forever stops and still
  speaks.
- The original bug stays fixed: a tool result must not read as
  something the assistant SAID, and must not make the model re-fire
  the tool on an unrelated follow-up.
- Cancellation mid-loop leaves conversation history clean.
- The pattern layer still short-circuits: no LLM call, no router-side
  tool execution.
- No double-speaking: a round that ends in tool calls speaks neither
  its narration nor the raw tool result.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from typing import Any

import pytest

from jarvis.core.config import ToolsConfig
from jarvis.llm.conversation import Conversation
from jarvis.llm.intent_router import (
    MAX_TOOL_RESULT_CHARS,
    IntentRouter,
    SpeakIntent,
    ToolIntent,
)
from jarvis.llm.ollama_client import ChatChunk
from jarvis.tools.registry import EmptyArgs, ToolRegistry, ToolResult

# --- helpers ---------------------------------------------------------


class ScriptedOllama:
    """Fake OllamaClient driven by a per-call script.

    `rounds` is a list of chunk lists: the Nth stream_chat call replays
    rounds[N]. Running past the end replays the last entry, which is how
    the "model calls a tool forever" cases are expressed. Every call's
    `messages` argument is deep-copied so tests can assert on exactly
    what the model saw at each iteration."""

    def __init__(self, rounds: list[list[ChatChunk]]) -> None:
        self.rounds = rounds
        self.seen_messages: list[list[dict]] = []
        self.last_tools: list[dict] | None = None

    @property
    def stream_calls(self) -> int:
        return len(self.seen_messages)

    async def stream_chat(self, messages, *, tools=None):
        index = min(len(self.seen_messages), len(self.rounds) - 1)
        self.seen_messages.append(copy.deepcopy(messages))
        self.last_tools = tools
        for chunk in self.rounds[index]:
            yield chunk


def _text(text: str, *, done: bool = False) -> ChatChunk:
    return ChatChunk(content=text, done=done)


def _calls(*names_and_args: tuple[str, dict], content: str = "") -> ChatChunk:
    """Final chunk carrying tool calls, the way Ollama batches them."""
    return ChatChunk(
        content=content,
        tool_calls=tuple(
            {"function": {"name": name, "arguments": args}}
            for name, args in names_and_args
        ),
        done=True,
    )


class _ScriptedTool:
    """Tool returning a canned ToolResult, counting its invocations."""

    args_schema = EmptyArgs
    requires_confirmation = False

    def __init__(self, name: str, result: ToolResult) -> None:
        self.name = name
        self.description = f"Scripted {name}."
        self._result = result
        self.calls = 0

    async def execute(self, args) -> ToolResult:
        self.calls += 1
        return self._result


def _registry(*tools) -> ToolRegistry:
    reg = ToolRegistry(ToolsConfig())
    for tool in tools:
        reg.register(tool)
    return reg


def _conversation() -> Conversation:
    return Conversation(system_prompt_provider=lambda: "sys")


def _router(
    llm: Any,  # any object with a stream_chat(messages, *, tools) generator
    registry: ToolRegistry | None,
    conv: Conversation | None = None,
    *,
    max_tool_iterations: int = 3,
) -> tuple[IntentRouter, Conversation]:
    conv = conv if conv is not None else _conversation()
    return (
        IntentRouter(
            llm=llm,
            conversation=conv,
            registry=registry,
            max_tool_iterations=max_tool_iterations,
        ),
        conv,
    )


async def _collect(gen) -> list:
    return [item async for item in gen]


def _spoken(intents: list) -> str:
    return "".join(i.text for i in intents if isinstance(i, SpeakIntent))


def _by_role(messages: list[dict], role: str) -> list[dict]:
    return [m for m in messages if m.get("role") == role]


# --- the two-step interaction ----------------------------------------


async def test_model_sees_tool_result_and_answers_from_it():
    """The headline case. Round 1 calls a tool; round 2 must be able to
    read the result out of history and answer with it."""
    weather = _ScriptedTool(
        "weather", ToolResult(success=True, output="4 degrees and raining")
    )
    llm = ScriptedOllama([
        [_calls(("weather", {}))],
        [_text("It's 4 degrees, sir. Take a coat."), _text("", done=True)],
    ])
    router, conv = _router(llm, _registry(weather))

    intents = await _collect(router.route("what's the weather like"))

    assert llm.stream_calls == 2
    assert weather.calls == 1
    # The second invocation carried the tool result.
    second = llm.seen_messages[1]
    tool_msgs = _by_role(second, "tool")
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "4 degrees and raining"
    # The answer the user hears is the model's, and it depends on the result.
    assert _spoken(intents) == "It's 4 degrees, sir. Take a coat."
    assert conv.current_messages()[-1] == {
        "role": "assistant",
        "content": "It's 4 degrees, sir. Take a coat.",
    }


async def test_second_tool_round_trip_chains():
    """Three rounds: tool, tool, prose. Round 3 must see BOTH results —
    this is the 'check the weather, and if it's cold open my coat app'
    shape."""
    weather = _ScriptedTool(
        "weather", ToolResult(success=True, output="4 degrees")
    )
    open_app = _ScriptedTool(
        "open_app", ToolResult(success=True, output="Opened Coats")
    )
    llm = ScriptedOllama([
        [_calls(("weather", {}))],
        [_calls(("open_app", {}))],
        [_text("It's cold, so I opened your coat app, sir.", done=True)],
    ])
    router, _ = _router(llm, _registry(weather, open_app))

    intents = await _collect(router.route("weather, and open my coat app if cold"))

    assert llm.stream_calls == 3
    third = llm.seen_messages[2]
    results = [m["content"] for m in _by_role(third, "tool")]
    assert results == ["4 degrees", "Opened Coats"]
    assert _spoken(intents) == "It's cold, so I opened your coat app, sir."


async def test_final_answer_preserves_chunk_granularity():
    """The pipeline pumps SpeakIntents into PiperTTS.speak_stream, which
    sentence-segments internally. The final round's chunks must arrive as
    separate SpeakIntents, not glued into one blob."""
    tool = _ScriptedTool("ping", ToolResult(success=True, output="pong"))
    llm = ScriptedOllama([
        [_calls(("ping", {}))],
        [_text("It said "), _text("pong, "), _text("sir."), _text("", done=True)],
    ])
    router, _ = _router(llm, _registry(tool))

    intents = await _collect(router.route("ping the thing"))

    assert [i.text for i in intents] == ["It said ", "pong, ", "sir."]


# --- protocol shape ---------------------------------------------------


async def test_tool_message_id_matches_assistant_tool_call_id():
    tool = _ScriptedTool("ping", ToolResult(success=True, output="pong"))
    llm = ScriptedOllama([
        [_calls(("ping", {"n": 1}))],
        [_text("done", done=True)],
    ])
    router, _ = _router(llm, _registry(tool))
    await _collect(router.route("ping the thing"))

    second = llm.seen_messages[1]
    assistant = next(m for m in second if m.get("tool_calls"))
    tool_msg = next(m for m in second if m.get("role") == "tool")

    assert len(assistant["tool_calls"]) == 1
    call = assistant["tool_calls"][0]
    assert call["id"]
    assert call["function"] == {"name": "ping", "arguments": {"n": 1}}
    assert tool_msg["tool_call_id"] == call["id"]
    assert tool_msg["name"] == "ping"
    # Order matters: the assistant message must precede its results.
    assert second.index(assistant) < second.index(tool_msg)


async def test_multiple_calls_in_one_round_get_distinct_ids():
    a = _ScriptedTool("alpha", ToolResult(success=True, output="A"))
    b = _ScriptedTool("beta", ToolResult(success=True, output="B"))
    llm = ScriptedOllama([
        [_calls(("alpha", {}), ("beta", {}))],
        [_text("both done", done=True)],
    ])
    router, _ = _router(llm, _registry(a, b))
    await _collect(router.route("do both things please"))

    second = llm.seen_messages[1]
    assistant = next(m for m in second if m.get("tool_calls"))
    ids = [c["id"] for c in assistant["tool_calls"]]
    tool_msgs = _by_role(second, "tool")

    assert len(set(ids)) == 2
    assert [m["tool_call_id"] for m in tool_msgs] == ids
    assert [m["content"] for m in tool_msgs] == ["A", "B"]


async def test_call_ids_stay_unique_across_turns():
    """Ids are the pairing key inside history, so two turns must never
    mint the same one — a collision would let the orphan filter match a
    result to the wrong assistant message."""
    tool = _ScriptedTool("ping", ToolResult(success=True, output="pong"))
    llm = ScriptedOllama([
        [_calls(("ping", {}))],
        [_text("first", done=True)],
        [_calls(("ping", {}))],
        [_text("second", done=True)],
    ])
    router, _ = _router(llm, _registry(tool))
    await _collect(router.route("ping once"))
    await _collect(router.route("ping again"))

    ids = []
    for msgs in llm.seen_messages:
        for m in msgs:
            if m.get("role") == "tool":
                ids.append(m["tool_call_id"])
    assert len(set(ids)) == 2, ids


async def test_dict_tool_output_reaches_model_as_json():
    """The spoken path collapses dict output to 'Done, sir.' because a
    listener can't consume structure. The model can — and 'tell me which
    one is the invoice' needs it."""
    listing = _ScriptedTool(
        "list_directory",
        ToolResult(success=True, output={"files": ["invoice.pdf", "cat.png"]}),
    )
    llm = ScriptedOllama([
        [_calls(("list_directory", {"path": "~/Downloads"}))],
        [_text("invoice.pdf, sir.", done=True)],
    ])
    router, _ = _router(llm, _registry(listing))
    await _collect(router.route("list my downloads and find the invoice"))

    tool_msg = next(m for m in llm.seen_messages[1] if m.get("role") == "tool")
    assert json.loads(tool_msg["content"]) == {
        "files": ["invoice.pdf", "cat.png"]
    }


async def test_tool_failure_is_reported_to_the_model():
    """A model told the tool failed can try something else; a model told
    nothing calls the same broken tool again."""
    broken = _ScriptedTool(
        "weather", ToolResult(success=False, error="no network, sir")
    )
    llm = ScriptedOllama([
        [_calls(("weather", {}))],
        [_text("I can't reach the forecast, sir.", done=True)],
    ])
    router, _ = _router(llm, _registry(broken))
    await _collect(router.route("what's the weather like"))

    tool_msg = next(m for m in llm.seen_messages[1] if m.get("role") == "tool")
    assert "no network, sir" in tool_msg["content"]


async def test_oversized_tool_result_is_truncated_for_the_model():
    huge = _ScriptedTool(
        "dump", ToolResult(success=True, output="x" * (MAX_TOOL_RESULT_CHARS * 2))
    )
    llm = ScriptedOllama([
        [_calls(("dump", {}))],
        [_text("that's a lot", done=True)],
    ])
    router, _ = _router(llm, _registry(huge))
    await _collect(router.route("dump everything you have"))

    tool_msg = next(m for m in llm.seen_messages[1] if m.get("role") == "tool")
    assert len(tool_msg["content"]) < MAX_TOOL_RESULT_CHARS * 2
    assert "truncated" in tool_msg["content"]


# --- the iteration bound ----------------------------------------------


async def test_iteration_bound_stops_a_model_that_calls_tools_forever(caplog):
    tool = _ScriptedTool("ping", ToolResult(success=True, output="pong"))
    # Only one round scripted, and ScriptedOllama repeats the last one:
    # the model asks for the tool again every single time.
    llm = ScriptedOllama([[_calls(("ping", {}))]])
    router, _ = _router(llm, _registry(tool), max_tool_iterations=3)

    with caplog.at_level(logging.WARNING, logger="jarvis.llm.intent_router"):
        intents = await _collect(router.route("ping the thing"))

    assert llm.stream_calls == 3
    assert tool.calls == 3
    # Degrades by speaking what it has rather than going silent.
    assert _spoken(intents) == "pong. "
    assert any("bound" in r.message for r in caplog.records)


async def test_iteration_bound_is_configurable():
    tool = _ScriptedTool("ping", ToolResult(success=True, output="pong"))
    llm = ScriptedOllama([[_calls(("ping", {}))]])
    router, _ = _router(llm, _registry(tool), max_tool_iterations=2)

    await _collect(router.route("ping the thing"))

    assert llm.stream_calls == 2


async def test_max_tool_iterations_one_restores_one_shot_dispatch():
    """The escape hatch: the router yields a ToolIntent for the caller to
    execute and speak, exactly as before the loop existed. It must not
    execute the tool itself."""
    tool = _ScriptedTool("ping", ToolResult(success=True, output="pong"))
    llm = ScriptedOllama([[_calls(("ping", {}))]])
    router, conv = _router(llm, _registry(tool), max_tool_iterations=1)

    intents = await _collect(router.route("ping the thing"))

    assert llm.stream_calls == 1
    assert tool.calls == 0
    assert [type(i).__name__ for i in intents] == ["ToolIntent"]
    assert not [m for m in conv.current_messages() if m["role"] == "tool"]


async def test_loop_disabled_without_a_registry():
    """No registry means no way to execute, so no way to observe a
    result: fall back to one-shot dispatch rather than pretend."""
    llm = ScriptedOllama([[_calls(("ping", {}))]])
    router, _ = _router(llm, None)

    intents = await _collect(router.route("ping the thing"))

    assert llm.stream_calls == 1
    assert [type(i).__name__ for i in intents] == ["ToolIntent"]


async def test_bound_hit_does_not_store_tool_output_as_assistant_text():
    """The degraded path speaks a raw tool result. It must not ALSO
    record it as assistant text — that duplicate is the exact shape the
    original bug had."""
    tool = _ScriptedTool("ping", ToolResult(success=True, output="pong"))
    llm = ScriptedOllama([[_calls(("ping", {}))]])
    router, conv = _router(llm, _registry(tool), max_tool_iterations=2)

    await _collect(router.route("ping the thing"))

    assistants = _by_role(conv.current_messages(), "assistant")
    assert all("pong" not in (m.get("content") or "") for m in assistants)


# --- no double-speaking -----------------------------------------------


async def test_narration_and_raw_result_are_both_suppressed_mid_loop():
    """A round that ends in tool calls speaks nothing: not the model's
    narration ('Checking that now') and not the tool's own output. The
    user hears exactly one utterance — the final answer."""
    tool = _ScriptedTool(
        "weather", ToolResult(success=True, output="4 degrees and raining")
    )
    llm = ScriptedOllama([
        [_text("Checking that now, sir. "), _calls(("weather", {}))],
        [_text("It's 4 degrees, sir.", done=True)],
    ])
    router, _ = _router(llm, _registry(tool))

    spoken = _spoken(await _collect(router.route("what's the weather like")))

    assert spoken == "It's 4 degrees, sir."
    assert "Checking that now" not in spoken
    assert "4 degrees and raining" not in spoken


async def test_router_yields_no_tool_intents_when_the_loop_is_on():
    """The producer adapter speaks every ToolIntent it receives. If the
    router both executed a tool AND yielded a ToolIntent for it, the
    adapter would run it a second time and speak the result on top of
    the model's answer."""
    tool = _ScriptedTool("ping", ToolResult(success=True, output="pong"))
    llm = ScriptedOllama([
        [_calls(("ping", {}))],
        [_text("It said pong, sir.", done=True)],
    ])
    router, _ = _router(llm, _registry(tool))

    intents = await _collect(router.route("ping the thing"))

    assert not [i for i in intents if isinstance(i, ToolIntent)]
    assert tool.calls == 1


# --- the original bug stays fixed -------------------------------------


async def test_tool_result_is_never_stored_as_assistant_text():
    """The regression the loop had to avoid re-introducing: a tool result
    stored as an ASSISTANT turn reads to the model as something it said,
    and it starts repeating the tool. Results live under role:"tool"
    with the id of the call that produced them, and nowhere else."""
    stats = _ScriptedTool(
        "system_stats", ToolResult(success=True, output="CPU at 13 percent")
    )
    llm = ScriptedOllama([
        [_calls(("system_stats", {}))],
        [_text("You're at 13 percent, sir.", done=True)],
    ])
    router, conv = _router(llm, _registry(stats))
    await _collect(router.route("what's my cpu doing"))

    msgs = conv.current_messages()
    assistants = _by_role(msgs, "assistant")
    assert all("CPU at 13 percent" not in (m.get("content") or "")
               for m in assistants)
    tool_msgs = _by_role(msgs, "tool")
    assert [m["content"] for m in tool_msgs] == ["CPU at 13 percent"]


class _RepeatIfItLooksLikeSomethingISaid:
    """A model that reproduces the ORIGINAL bug's failure mode.

    The author's live observation was that tool output stored as
    assistant text made the model re-fire the same tool on the next,
    unrelated turn — it read its own apparent words as a standing
    instruction to keep reporting stats. This fake encodes exactly that
    policy: call system_stats whenever the history shows the assistant
    having said something stats-shaped. With the feedback loop the
    result lives under role:"tool" instead, so the policy never trips
    and the follow-up gets a plain answer.

    It is a model of the bug, not the bug itself — no local 7B is going
    to behave identically. What it pins down is the structural claim:
    nothing in the history the model receives presents a tool result as
    assistant speech."""

    def __init__(self) -> None:
        self.seen_messages: list[list[dict]] = []
        self.refired = False

    @property
    def stream_calls(self) -> int:
        return len(self.seen_messages)

    async def stream_chat(self, messages, *, tools=None):
        self.seen_messages.append(copy.deepcopy(messages))
        said_by_assistant = " ".join(
            m.get("content") or ""
            for m in messages
            if m.get("role") == "assistant"
        )
        if "CPU at" in said_by_assistant:
            self.refired = True
            yield _calls(("system_stats", {}))
            return
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=-1,
        )
        last_user = (
            messages[last_user_idx].get("content", "")
            if last_user_idx >= 0
            else ""
        )
        this_turn = messages[last_user_idx + 1 :]
        if any(m.get("role") == "tool" for m in this_turn):
            # The result is in hand; answer from it.
            yield _text("13 percent, sir.", done=True)
        elif "cpu" in last_user.lower():
            yield _calls(("system_stats", {}))
        else:
            yield _text("Good evening, sir.", done=True)


async def test_tool_result_does_not_refire_the_tool_on_a_followup():
    stats = _ScriptedTool(
        "system_stats", ToolResult(success=True, output="CPU at 13 percent")
    )
    llm = _RepeatIfItLooksLikeSomethingISaid()
    router, _ = _router(llm, _registry(stats))

    await _collect(router.route("what's my cpu doing"))
    calls_after_first_turn = stats.calls
    intents = await _collect(router.route("good evening"))

    assert calls_after_first_turn == 1
    assert not llm.refired
    assert stats.calls == 1, "the tool re-fired on an unrelated follow-up"
    assert _spoken(intents) == "Good evening, sir."


async def test_the_bug_mimic_is_not_vacuous():
    """Guard on the guard. Hand the same fake the OLD history shape —
    the tool result recorded as an assistant turn — and it re-fires, as
    the live bug report described. So the test above passing means the
    shape genuinely changed, not that the fake never triggers."""
    stats = _ScriptedTool(
        "system_stats", ToolResult(success=True, output="CPU at 13 percent")
    )
    llm = _RepeatIfItLooksLikeSomethingISaid()
    conv = _conversation()
    conv.add_user_turn("what's my cpu doing")
    conv.add_assistant_turn("CPU at 13 percent")  # the pre-fix shape
    router, _ = _router(llm, _registry(stats), conv)

    await _collect(router.route("good evening"))

    assert llm.refired
    assert stats.calls >= 1


async def test_followup_turn_still_carries_the_tool_history():
    """The other half of the trade: not re-firing must not mean amnesia.
    The follow-up turn still shows the model the round trip, so
    'and the memory?' has something to refer back to."""
    stats = _ScriptedTool(
        "system_stats", ToolResult(success=True, output="CPU at 13 percent")
    )
    llm = ScriptedOllama([
        [_calls(("system_stats", {}))],
        [_text("13 percent, sir.", done=True)],
        [_text("Steady, sir.", done=True)],
    ])
    router, _ = _router(llm, _registry(stats))
    await _collect(router.route("what's my cpu doing"))
    await _collect(router.route("is that normal"))

    third = llm.seen_messages[2]
    assert [m["content"] for m in _by_role(third, "tool")] == [
        "CPU at 13 percent"
    ]
    assert any(m.get("tool_calls") for m in third)


# --- cancellation ------------------------------------------------------


async def test_cancel_during_tool_execution_leaves_history_clean():
    """Cancel between 'the model asked for a tool' and 'the tool
    answered'. History must not keep an assistant tool_calls message
    whose results never arrived — chat APIs reject that shape."""
    started = asyncio.Event()
    release = asyncio.Event()

    class _StallingTool:
        name = "slow"
        description = "Blocks until released."
        args_schema = EmptyArgs
        requires_confirmation = False

        async def execute(self, args):
            started.set()
            await release.wait()
            return ToolResult(success=True, output="never gets here")

    llm = ScriptedOllama([
        [_calls(("slow", {}))],
        [_text("done", done=True)],
    ])
    router, conv = _router(llm, _registry(_StallingTool()))

    route = router.route("do the slow thing")
    task = asyncio.create_task(_collect(route))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await route.aclose()

    msgs = conv.current_messages()
    assert not [m for m in msgs if m.get("role") == "tool"]
    assert not [m for m in msgs if m.get("tool_calls")]
    # The user turn survives (the documented tradeoff); nothing else does.
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the slow thing"},
    ]


async def test_cancel_during_the_second_round_leaves_history_clean():
    """Cancel after a completed round trip, mid-way through the model's
    follow-up stream. The round trip is legitimate history and stays;
    the partial assistant text does not."""
    started = asyncio.Event()
    release = asyncio.Event()

    class _StallingSecondRound:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_chat(self, messages, *, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield _calls(("ping", {}))
                return
            yield _text("Half a sen")
            started.set()
            await release.wait()
            yield _text("tence.", done=True)

    tool = _ScriptedTool("ping", ToolResult(success=True, output="pong"))
    router, conv = _router(_StallingSecondRound(), _registry(tool))

    route = router.route("ping the thing")
    task = asyncio.create_task(_collect(route))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Close the suspended generator here rather than leaving it for the
    # GC: an un-finalised async generator holding a stalled await is
    # cleaned up at interpreter whim and has been observed to disturb
    # the event loop later tests in the same session run on.
    await route.aclose()

    msgs = conv.current_messages()
    # The completed round trip is intact and well-formed.
    assistant = next(m for m in msgs if m.get("tool_calls"))
    tool_msg = next(m for m in msgs if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == assistant["tool_calls"][0]["id"]
    # No partial assistant text anywhere.
    assert all(
        "Half a sen" not in (m.get("content") or "") for m in msgs
    )


# --- the pattern layer is untouched ------------------------------------


async def test_pattern_match_never_reaches_the_llm_or_the_loop():
    """Snap commands must stay snap commands: no inference, and no
    router-side tool execution either — the adapter runs them and speaks
    the tool's own output, which is what makes 'open spotify' instant."""
    open_app = _ScriptedTool(
        "open_app", ToolResult(success=True, output="Opened Spotify")
    )
    llm = ScriptedOllama([[_text("should never run", done=True)]])
    router, conv = _router(llm, _registry(open_app))

    intents = await _collect(router.route("open spotify"))

    assert llm.stream_calls == 0
    assert open_app.calls == 0
    assert intents == [ToolIntent("open_app", {"name": "spotify"})]
    # Pattern hits stay out of conversation history entirely.
    assert conv.current_messages() == [{"role": "system", "content": "sys"}]


async def test_stop_command_still_short_circuits_with_the_loop_on():
    llm = ScriptedOllama([[_text("should never run", done=True)]])
    router, _ = _router(llm, _registry())

    intents = await _collect(router.route("nevermind"))

    assert llm.stream_calls == 0
    assert [type(i).__name__ for i in intents] == ["StopIntent"]


# --- malformed tool calls ---------------------------------------------


async def test_all_calls_malformed_ends_the_turn_without_speaking():
    """Narration written to accompany an action that never happened is
    not worth speaking, and there is nothing to feed back."""
    llm = ScriptedOllama([
        [
            ChatChunk(
                content="Working on it. ",
                tool_calls=({"function": {"arguments": {}}},),
                done=True,
            )
        ],
    ])
    router, conv = _router(llm, _registry())

    intents = await _collect(router.route("do the thing for me"))

    assert intents == []
    assert llm.stream_calls == 1
    assert not _by_role(conv.current_messages(), "assistant")


async def test_duplicate_calls_in_one_round_execute_once():
    tool = _ScriptedTool("ping", ToolResult(success=True, output="pong"))
    llm = ScriptedOllama([
        [_calls(("ping", {}), ("ping", {}))],
        [_text("done", done=True)],
    ])
    router, _ = _router(llm, _registry(tool))
    await _collect(router.route("ping the thing"))

    assert tool.calls == 1
    assert len(_by_role(llm.seen_messages[1], "tool")) == 1
