"""Tests for jarvis.llm.conversation.Conversation."""

from __future__ import annotations

import pytest

from jarvis.llm.conversation import (
    Conversation,
    ToolExchange,
    _filter_turns_for_llm,
)

# --- helpers ---


def _provider(prompt: str = "system here"):
    return lambda: prompt


class _Clock:
    """Deterministic time provider for inactivity-timeout tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# --- construction validation ---


def test_invalid_max_turns_rejected():
    with pytest.raises(ValueError):
        Conversation(system_prompt_provider=_provider(), max_turns=0)
    with pytest.raises(ValueError):
        Conversation(system_prompt_provider=_provider(), max_turns=-1)


def test_invalid_inactivity_timeout_rejected():
    with pytest.raises(ValueError):
        Conversation(
            system_prompt_provider=_provider(),
            inactivity_timeout_seconds=0,
        )
    with pytest.raises(ValueError):
        Conversation(
            system_prompt_provider=_provider(),
            inactivity_timeout_seconds=-1,
        )


# --- basic flow ---


def test_add_user_turn_returns_messages_with_system_prompt():
    c = Conversation(system_prompt_provider=_provider("You are Jarvis."))
    msgs = c.add_user_turn("hello")
    assert msgs == [
        {"role": "system", "content": "You are Jarvis."},
        {"role": "user", "content": "hello"},
    ]


def test_add_assistant_turn_appends():
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.add_user_turn("hi")
    c.add_assistant_turn("hello")
    assert c.current_messages() == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_alternating_turns_preserved_in_order():
    c = Conversation(system_prompt_provider=_provider("sys"))
    for i in range(3):
        c.add_user_turn(f"u{i}")
        c.add_assistant_turn(f"a{i}")
    msgs = c.current_messages()
    # 1 system + 6 turn messages
    assert len(msgs) == 7
    assert msgs[0]["role"] == "system"
    contents = [m["content"] for m in msgs[1:]]
    assert contents == ["u0", "a0", "u1", "a1", "u2", "a2"]


# --- max_turns truncation ---


def test_max_turns_drops_oldest_preserves_system_prompt():
    # Trim runs after every add. With max_turns=4 and 6 user/assistant
    # pairs, the tail settles to the last 4 messages: [u4, a4, u5, a5].
    c = Conversation(system_prompt_provider=_provider("sys"), max_turns=4)
    for i in range(6):
        c.add_user_turn(f"u{i}")
        c.add_assistant_turn(f"a{i}")
    msgs = c.current_messages()
    assert msgs[0]["role"] == "system"
    assert len(msgs) == 5  # system + 4 turns
    contents = [m["content"] for m in msgs[1:]]
    assert contents == ["u4", "a4", "u5", "a5"]
    # All earlier turns dropped.
    assert "u0" not in contents
    assert "a0" not in contents
    assert "u3" not in contents
    assert "a3" not in contents


def test_max_turns_one_keeps_only_latest_message():
    c = Conversation(system_prompt_provider=_provider("sys"), max_turns=1)
    c.add_user_turn("first")
    c.add_assistant_turn("response")
    c.add_user_turn("second")
    msgs = c.current_messages()
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "second"},
    ]


def test_trim_happens_after_each_add():
    c = Conversation(system_prompt_provider=_provider("sys"), max_turns=2)
    c.add_user_turn("a")
    c.add_assistant_turn("b")
    c.add_user_turn("c")  # triggers trim of "a"
    msgs = c.current_messages()
    contents = [m["content"] for m in msgs[1:]]
    assert contents == ["b", "c"]


# --- inactivity timeout ---


def test_inactivity_timeout_does_not_fire_before_first_message():
    clock = _Clock(t=10_000)
    c = Conversation(
        system_prompt_provider=_provider("sys"),
        inactivity_timeout_seconds=10,
        time_provider=clock,
    )
    msgs = c.add_user_turn("first")
    # No prior activity to trigger a clear from.
    assert len(msgs) == 2  # system + user


def test_inactivity_timeout_clears_on_next_user_turn_after_threshold():
    clock = _Clock()
    c = Conversation(
        system_prompt_provider=_provider("sys"),
        inactivity_timeout_seconds=300,
        time_provider=clock,
    )
    c.add_user_turn("hello")
    c.add_assistant_turn("hi")

    # Just under the timeout: history preserved.
    clock.advance(299)
    msgs = c.add_user_turn("are you there?")
    contents = [m["content"] for m in msgs[1:]]
    assert "hello" in contents

    # Push past the timeout: next user turn should clear.
    clock.advance(301)
    msgs = c.add_user_turn("anyone?")
    contents = [m["content"] for m in msgs[1:]]
    assert contents == ["anyone?"]


def test_assistant_turn_resets_inactivity_timer():
    """Tripwire: a slow LLM response (e.g., 30 s on first inference)
    must NOT cause the next user turn to wipe history. Both user and
    assistant turns reset the activity clock."""
    clock = _Clock()
    c = Conversation(
        system_prompt_provider=_provider("sys"),
        inactivity_timeout_seconds=10,
        time_provider=clock,
    )
    c.add_user_turn("hi")  # last_activity = 0
    clock.advance(8)
    c.add_assistant_turn("hello")  # last_activity should reset to 8
    clock.advance(8)
    # 16 s since user turn but only 8 s since assistant. Must not clear.
    msgs = c.add_user_turn("more")
    contents = [m["content"] for m in msgs[1:]]
    assert "hi" in contents


def test_inactivity_threshold_exclusive_at_exact_value():
    """The condition is `> timeout`, not `>= timeout`. Exact-equals
    should NOT clear (avoids flakiness around a tightly-timed user)."""
    clock = _Clock()
    c = Conversation(
        system_prompt_provider=_provider("sys"),
        inactivity_timeout_seconds=10,
        time_provider=clock,
    )
    c.add_user_turn("hi")
    clock.advance(10)  # exactly at threshold
    c.add_user_turn("more")
    # "hi" must still be in raw _turns (inactivity timer did not fire a clear).
    # current_messages() filters the orphaned "hi" turn, so check _turns directly.
    raw_contents = [m["content"] for m in c._turns]
    assert "hi" in raw_contents  # not cleared


# --- clear() ---


def test_clear_wipes_history_keeps_system_prompt():
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.add_user_turn("a")
    c.add_assistant_turn("b")
    c.clear()
    assert c.current_messages() == [{"role": "system", "content": "sys"}]


def test_clear_when_already_empty_is_safe():
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.clear()  # must not raise
    assert c.current_messages() == [{"role": "system", "content": "sys"}]


# --- system prompt sourcing (live) ---


def test_system_prompt_updates_on_each_call():
    """The provider is consulted on every current_messages -- a
    settings-UI edit to the prompt flows through without restart."""
    prompt = ["initial"]
    c = Conversation(system_prompt_provider=lambda: prompt[0])
    c.add_user_turn("hi")
    assert c.current_messages()[0]["content"] == "initial"

    prompt[0] = "updated"
    assert c.current_messages()[0]["content"] == "updated"


def test_empty_system_prompt_omits_system_message_entirely():
    c = Conversation(system_prompt_provider=lambda: "")
    c.add_user_turn("hi")
    assert c.current_messages() == [{"role": "user", "content": "hi"}]


# --- current_messages safety ---


def test_current_messages_returns_fresh_list():
    """Mutating the returned list MUST NOT affect internal state."""
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.add_user_turn("hi")
    msgs = c.current_messages()
    msgs.append({"role": "user", "content": "INJECTED"})
    msgs.clear()
    contents = [m["content"] for m in c.current_messages()[1:]]
    assert contents == ["hi"]
    assert "INJECTED" not in contents


# --- maybe_clear() --------------------------------------------------------


def test_maybe_clear_within_window_keeps_history():
    """Wake within continuity window: history preserved."""
    clock = _Clock()
    c = Conversation(system_prompt_provider=_provider("sys"), time_provider=clock)
    c.add_user_turn("tell me a story")
    c.add_assistant_turn("Once upon a time...")
    clock.advance(30)  # 30s < 60s default-ish; we pass 60 explicitly
    c.maybe_clear(continuity_seconds=60)
    contents = [m["content"] for m in c.current_messages()[1:]]
    assert "tell me a story" in contents
    assert "Once upon a time..." in contents


def test_maybe_clear_outside_window_clears_history():
    """Wake beyond continuity window: history wiped."""
    clock = _Clock()
    c = Conversation(system_prompt_provider=_provider("sys"), time_provider=clock)
    c.add_user_turn("hello")
    c.add_assistant_turn("hi")
    clock.advance(90)  # 90s > 60s
    c.maybe_clear(continuity_seconds=60)
    assert c.current_messages() == [{"role": "system", "content": "sys"}]


def test_maybe_clear_at_exact_threshold_keeps_history():
    """Boundary: exactly at the threshold is NOT cleared (condition is >)."""
    clock = _Clock()
    c = Conversation(system_prompt_provider=_provider("sys"), time_provider=clock)
    c.add_user_turn("hi")
    c.add_assistant_turn("hello")
    clock.advance(60)  # exactly at threshold
    c.maybe_clear(continuity_seconds=60)
    contents = [m["content"] for m in c.current_messages()[1:]]
    assert "hi" in contents


def test_maybe_clear_with_no_history_is_safe():
    """Calling maybe_clear with no prior turns must not raise."""
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.maybe_clear(continuity_seconds=60)  # must not raise
    assert c.current_messages() == [{"role": "system", "content": "sys"}]


def test_maybe_clear_configurable_threshold():
    """A short threshold (10s) causes clear after 15s; longer (30s) keeps it."""
    clock = _Clock()
    c = Conversation(system_prompt_provider=_provider("sys"), time_provider=clock)
    c.add_user_turn("a")
    c.add_assistant_turn("b")
    clock.advance(15)

    c.maybe_clear(continuity_seconds=10)   # 15 > 10: should clear
    assert c.current_messages() == [{"role": "system", "content": "sys"}]

    # Rebuild and use a longer threshold.
    c2 = Conversation(system_prompt_provider=_provider("sys"), time_provider=clock)
    c2.add_user_turn("a")
    c2.add_assistant_turn("b")
    clock.advance(15)  # total 30s from first add, 15s from second

    c2.maybe_clear(continuity_seconds=30)  # 15 <= 30: keep
    assert len(c2.current_messages()) > 1


# --- _filter_turns_for_llm ---


def test_filter_removes_tool_role_messages_with_no_parent_call_id():
    """Tool messages that cannot be paired with an assistant tool_calls
    entry are dropped — the chat API rejects that shape. Well-formed
    pairs are kept; see the tool round-trip section below."""
    turns = [
        {"role": "user", "content": "what's the weather?"},
        {"role": "assistant", "tool_calls": [{"name": "get_weather"}]},
        {"role": "tool", "content": "72 degrees"},
    ]
    filtered = _filter_turns_for_llm(turns)
    roles = [m["role"] for m in filtered]
    assert "tool" not in roles


def test_filter_removes_assistant_with_only_tool_calls_no_content():
    turns = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [{"name": "get_weather"}]},
    ]
    filtered = _filter_turns_for_llm(turns)
    assert all(m.get("role") != "assistant" for m in filtered)


def test_filter_strips_tool_calls_key_from_assistant_with_content():
    turns = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "It's sunny", "tool_calls": [{"name": "x"}]},
    ]
    filtered = _filter_turns_for_llm(turns)
    assistant = next(m for m in filtered if m.get("role") == "assistant")
    assert "tool_calls" not in assistant
    assert assistant["content"] == "It's sunny"


def test_filter_removes_orphaned_user_turn_not_last():
    """A user message with no following assistant (not the last message) is dropped."""
    turns = [
        {"role": "user", "content": "orphan"},
        {"role": "user", "content": "current"},
    ]
    filtered = _filter_turns_for_llm(turns)
    contents = [m["content"] for m in filtered]
    assert "orphan" not in contents
    assert "current" in contents


def test_filter_preserves_complete_user_assistant_pairs():
    turns = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
        {"role": "assistant", "content": "goodbye"},
    ]
    filtered = _filter_turns_for_llm(turns)
    assert filtered == turns


def test_filter_keeps_last_user_turn_even_without_assistant():
    """The last message being a user turn (in-flight query) must not be dropped."""
    turns = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "current question"},
    ]
    filtered = _filter_turns_for_llm(turns)
    contents = [m["content"] for m in filtered]
    assert "current question" in contents


def test_filter_empty_turns_returns_empty():
    assert _filter_turns_for_llm([]) == []


# --- has_unanswered_user_turn ---


def test_has_unanswered_user_turn_true_when_last_is_user():
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.add_user_turn("hello")
    assert c.has_unanswered_user_turn() is True


def test_has_unanswered_user_turn_false_after_assistant_reply():
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.add_user_turn("hello")
    c.add_assistant_turn("hi there")
    assert c.has_unanswered_user_turn() is False


def test_has_unanswered_user_turn_false_when_empty():
    c = Conversation(system_prompt_provider=_provider("sys"))
    assert c.has_unanswered_user_turn() is False


# --- current_messages filters tool artifacts ---


def test_current_messages_filters_tool_role():
    """role:tool entries must not appear in current_messages output."""
    c = Conversation(system_prompt_provider=lambda: "")
    c._turns = [  # inject directly to test filter in isolation
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [{"name": "get_weather"}]},
        {"role": "tool", "content": "It's 72 degrees"},
    ]
    msgs = c.current_messages()
    roles = [m["role"] for m in msgs]
    assert "tool" not in roles


def test_current_messages_filters_orphaned_user_then_new_user():
    """Two consecutive user turns: the earlier orphan is dropped."""
    c = Conversation(system_prompt_provider=lambda: "")
    c._turns = [
        {"role": "user", "content": "stale orphan"},
        {"role": "user", "content": "new question"},
    ]
    msgs = c.current_messages()
    contents = [m["content"] for m in msgs]
    assert "stale orphan" not in contents
    assert "new question" in contents




# --- tool round trips (the feedback loop) ----------------------------


def _exchange(call_id: str = "call_1", name: str = "weather",
              result: str = "4 degrees") -> ToolExchange:
    return ToolExchange(call_id=call_id, name=name, result=result)


def test_add_tool_round_trip_writes_the_protocol_shape():
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.add_user_turn("what's the weather?")
    msgs = c.add_tool_round_trip(
        content="",
        exchanges=[ToolExchange("call_1", "weather", {"unit": "c"}, "4 degrees")],
    )
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": {"unit": "c"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "weather",
            "content": "4 degrees",
        },
    ]


def test_add_tool_round_trip_keeps_narration_on_the_assistant_message():
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.add_user_turn("weather?")
    msgs = c.add_tool_round_trip(
        content="Checking, sir. ", exchanges=[_exchange()]
    )
    assistant = next(m for m in msgs if m.get("tool_calls"))
    assert assistant["content"] == "Checking, sir. "


def test_add_tool_round_trip_one_tool_message_per_exchange():
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.add_user_turn("two things")
    msgs = c.add_tool_round_trip(
        content="",
        exchanges=[
            _exchange("call_1", "alpha", "A"),
            _exchange("call_2", "beta", "B"),
        ],
    )
    tools = [m for m in msgs if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tools] == [
        ("call_1", "A"),
        ("call_2", "B"),
    ]


def test_add_tool_round_trip_resets_the_inactivity_timer():
    """A slow tool must not let the NEXT user turn look like a stale
    session and wipe the round trip the model is about to read."""
    clock = _Clock()
    c = Conversation(
        system_prompt_provider=_provider("sys"),
        inactivity_timeout_seconds=100.0,
        time_provider=clock,
    )
    c.add_user_turn("weather?")
    clock.advance(90.0)  # a very slow tool
    c.add_tool_round_trip(content="", exchanges=[_exchange()])
    clock.advance(90.0)
    c.add_user_turn("and tomorrow?")
    contents = [m.get("content") for m in c.current_messages()]
    assert "4 degrees" in contents


def test_round_trip_survives_into_current_messages():
    """The whole point: what the tool returned reaches the model."""
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.add_user_turn("weather?")
    c.add_tool_round_trip(content="", exchanges=[_exchange()])
    c.add_assistant_turn("It's 4 degrees, sir.")
    c.add_user_turn("and tomorrow?")
    msgs = c.current_messages()
    assert [m["role"] for m in msgs] == [
        "system", "user", "assistant", "tool", "assistant", "user",
    ]


# --- trimming must never orphan a tool message -----------------------


def test_trim_drops_tool_messages_along_with_their_assistant_parent():
    """max_turns is small enough that the cut lands on the assistant
    message carrying the tool_calls. Its results must go with it — a
    role:"tool" message with no parent tool_call_id is rejected by
    Ollama and the OpenAI-compatible APIs alike."""
    c = Conversation(system_prompt_provider=_provider("sys"), max_turns=3)
    c.add_user_turn("weather?")
    c.add_tool_round_trip(
        content="",
        exchanges=[_exchange("call_1", "weather", "4 degrees")],
    )
    # 3 turns stored: user, assistant(tool_calls), tool.
    c.add_assistant_turn("It's 4 degrees, sir.")
    # Trimming to 3 leaves [assistant(tool_calls), tool, assistant]; one
    # more turn pushes the cut onto the assistant tool_calls message.
    c.add_user_turn("and tomorrow?")

    stored = c._turns
    assert not any(m.get("role") == "tool" for m in stored), stored
    assert len(stored) <= 3


def test_trim_keeps_a_round_trip_whole_when_it_survives():
    c = Conversation(system_prompt_provider=_provider("sys"), max_turns=3)
    c.add_user_turn("weather?")
    c.add_tool_round_trip(
        content="",
        exchanges=[_exchange("call_1", "weather", "4 degrees")],
    )
    stored = c._turns
    assert [m["role"] for m in stored] == ["user", "assistant", "tool"]
    assert stored[2]["tool_call_id"] == stored[1]["tool_calls"][0]["id"]


def test_trim_never_leaves_an_orphan_for_any_max_turns():
    """Property check across every cut position: after any sequence of
    adds, each stored tool message has an assistant message ahead of it
    declaring its id."""
    for max_turns in range(1, 9):
        c = Conversation(
            system_prompt_provider=_provider("sys"), max_turns=max_turns
        )
        c.add_user_turn("weather?")
        c.add_tool_round_trip(
            content="",
            exchanges=[
                ToolExchange("call_1", "weather", {}, "4 degrees"),
                ToolExchange("call_2", "clock", {}, "9pm"),
            ],
        )
        c.add_assistant_turn("Cold and late, sir.")
        c.add_user_turn("thanks")
        for i, msg in enumerate(c._turns):
            if msg.get("role") != "tool":
                continue
            declared = {
                call["id"]
                for m in c._turns[:i]
                for call in (m.get("tool_calls") or [])
            }
            assert msg["tool_call_id"] in declared, (
                f"orphaned tool message at max_turns={max_turns}: {c._turns}"
            )


def test_filter_drops_a_tool_message_orphaned_by_a_hand_built_history():
    turns = [
        {"role": "tool", "tool_call_id": "call_1", "content": "4 degrees"},
        {"role": "user", "content": "weather?"},
    ]
    filtered = _filter_turns_for_llm(turns)
    assert all(m["role"] != "tool" for m in filtered)


def test_filter_drops_an_assistant_tool_call_whose_result_never_arrived():
    """The mid-flight state a cancelled turn would leave if the two
    halves were ever written separately."""
    turns = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "weather", "arguments": {}}}
            ],
        },
    ]
    filtered = _filter_turns_for_llm(turns)
    assert filtered == [{"role": "user", "content": "weather?"}]


def test_filter_drops_tool_calls_when_only_some_results_arrived():
    """Partial results are as rejectable as none: every declared id
    needs an answer."""
    turns = [
        {"role": "user", "content": "two things"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "alpha", "arguments": {}}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "beta", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "A"},
    ]
    filtered = _filter_turns_for_llm(turns)
    assert [m["role"] for m in filtered] == ["user"]


def test_filter_keeps_narration_when_dropping_an_unanswered_tool_call():
    turns = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "Checking, sir.",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "weather", "arguments": {}}}
            ],
        },
    ]
    filtered = _filter_turns_for_llm(turns)
    assistant = next(m for m in filtered if m["role"] == "assistant")
    assert assistant == {"role": "assistant", "content": "Checking, sir."}


def test_filter_keeps_a_well_formed_round_trip_intact():
    turns = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "weather", "arguments": {}}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "4 degrees"},
        {"role": "assistant", "content": "It's 4 degrees, sir."},
    ]
    assert _filter_turns_for_llm(turns) == turns


def test_filter_pairs_results_only_with_a_preceding_assistant_message():
    """A tool message sitting BEFORE the assistant that declares its id
    is not a round trip, whatever the ids say."""
    turns = [
        {"role": "tool", "tool_call_id": "call_1", "content": "early"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "weather", "arguments": {}}}
            ],
        },
    ]
    filtered = _filter_turns_for_llm(turns)
    assert filtered == []


def test_clear_wipes_tool_round_trips_too():
    c = Conversation(system_prompt_provider=_provider("sys"))
    c.add_user_turn("weather?")
    c.add_tool_round_trip(content="", exchanges=[_exchange()])
    c.clear()
    assert c.current_messages() == [{"role": "system", "content": "sys"}]
