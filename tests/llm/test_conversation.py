"""Tests for jarvis.llm.conversation.Conversation."""

from __future__ import annotations

import pytest

from jarvis.llm.conversation import Conversation, _filter_turns_for_llm

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


def test_filter_removes_tool_role_messages():
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


