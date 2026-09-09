"""In-memory conversation state for the Jarvis chat loop.

Maintains turn history as message dicts in OpenAI/Ollama format
([{"role": "user"|"assistant", "content": str}, ...]). Trims to
max_turns (system prompt always preserved at index 0). Clears after
inactivity_timeout_seconds of idle on the next user turn.

Not a Loadable -- pure in-memory state, no resources.

Wake-word activation policy
---------------------------
maybe_clear(continuity_seconds) is wired by the composition root to
AudioPipeline's on_wake hook. It uses a time-windowed rule:

- If elapsed_since_last_assistant_turn <= continuity_seconds: keep
  history — the user is following up ("yes", "the sorcerer part", etc.)
- If elapsed_since_last_assistant_turn > continuity_seconds: wipe —
  a new session has begun and stale tool-call context would bias the LLM.

Default continuity_seconds = 60 (cfg.llm.conversation_continuity_seconds).
Trade-offs: 10 s — aggressive follow-ups don't link; 300 s — cross-task
contamination risk returns; 60 s — matches typical conversational tempo.

System prompt sourcing
----------------------
The constructor takes a `system_prompt_provider: Callable[[], str]`
called on every current_messages() (and therefore every add_user_turn,
which returns the message list). This means edits to LLMConfig's
system_prompt take effect on the very next user message without a
restart and without any ConfigChanged subscription -- the conversation
just always reads the live value.

Picked over the alternative (subscribe to ConfigChanged and update
a stored prompt) because:
  - Zero synchronization risk; no stale-cache window.
  - One function call per turn cost; negligible.
  - Conversation stays event-system-free (matches the Phase 1 layering
    decision that core.config also has no events.py dependency).

The composition root passes `lambda: cfg.llm.system_prompt`.

Inactivity policy
-----------------
last_activity is updated by BOTH add_user_turn and add_assistant_turn.
If only the user turn updated it, a slow LLM response (e.g., 30 s on
a fresh model load) followed by a follow-up user turn would falsely
clear the history. Resetting on assistant turn too keeps the
conversation alive across the full request/response cycle.

Tool round trips (the feedback loop)
------------------------------------
add_tool_round_trip() records the OpenAI/Ollama tool-calling shape:
one assistant message carrying `tool_calls`, immediately followed by
one `role: "tool"` message per call, each keyed by `tool_call_id`.
That is what lets the model SEE what a tool returned and act on it
("what's the weather, and if it's below 10 open my coat app").

Why role:"tool" and not an assistant turn. An earlier revision stored
tool result strings as ASSISTANT turns; the model read them as things
it had said and re-fired the same tool on unrelated follow-ups. The
fix at the time was to drop tool results entirely (see the removed
role:"tool" filter). Storing them under their own role with the call
id that produced them is the shape the protocol actually defines --
the model reads it as "a tool I already called and already have an
answer for", which is precisely the framing the old bug lacked.

Orphaning is the hazard this file has to handle. A round trip adds
1 + N messages at once, so trimming to max_turns can decapitate one:
cut the assistant message and the role:"tool" messages that referenced
it are left dangling. Chat APIs (Ollama included) reject a tool
message with no parent tool_call_id, and reject an assistant tool_calls
message whose results never arrived. Two layers defend against it:

  1. _trim() extends its cut forward over any role:"tool" message that
     would be left without its parent (write time).
  2. _filter_turns_for_llm() drops both halves of a broken pair
     (read time) -- covering hand-built history and the mid-flight
     state left by a turn cancelled between the assistant tool_calls
     message and its results.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# A single message in OpenAI/Ollama chat format.
Message = dict


@dataclass(frozen=True, slots=True)
class ToolExchange:
    """One executed tool call plus the result the model should see.

    `call_id` is the correlation key: it is written into the assistant
    message's tool_calls[].id AND into the matching role:"tool"
    message's tool_call_id. `result` is already stringified -- the
    conversation stores text, not ToolResult objects, so this module
    stays free of any tools/ dependency."""

    call_id: str
    name: str
    arguments: dict = field(default_factory=dict)
    result: str = ""


def _tool_call_ids(msg: Message) -> list[str]:
    """The declared call ids of an assistant tool_calls message.
    Non-dict / id-less entries yield an empty string, which can never
    match a stored tool_call_id -- so a malformed call always reads as
    unanswered rather than silently passing the pairing check."""
    calls = msg.get("tool_calls") or []
    if not isinstance(calls, (list, tuple)):
        return []
    return [
        (c.get("id") or "") if isinstance(c, dict) else "" for c in calls
    ]


def _filter_turns_for_llm(turns: list[Message]) -> list[Message]:
    """Normalise stored turns into a shape the chat API will accept.

    Kept:
    - assistant messages carrying tool_calls, but only when every call
      id they declare has a matching role:"tool" result *later* in the
      list. This is the feedback loop's payload -- dropping it would
      put us back to one-shot dispatch.
    - role:"tool" messages whose tool_call_id belongs to such an
      assistant message.

    Removed:
    - assistant tool_calls with no matching results (a turn cancelled
      between dispatch and completion). The tool_calls key is stripped;
      if the message had no spoken content it is dropped entirely.
    - role:"tool" messages orphaned from their parent assistant message
      (trimming decapitated the round trip, or history was hand-built).
    - orphaned user turns: user messages with no following assistant
      turn, excluding the last message (the current turn being sent).
    """
    result: list[Message] = []
    live_call_ids: set[str] = set()
    n = len(turns)
    for i, msg in enumerate(turns):
        role = msg.get("role")
        is_last = i == n - 1
        if role == "tool":
            # Only survives alongside the assistant message that
            # declared its id; that message was appended to `result`
            # earlier in this same pass, so live_call_ids is authoritative.
            if msg.get("tool_call_id") in live_call_ids:
                result.append(msg)
            continue
        if role == "assistant" and "tool_calls" in msg:
            call_ids = _tool_call_ids(msg)
            answered_after = {
                t.get("tool_call_id")
                for t in turns[i + 1 :]
                if t.get("role") == "tool"
            }
            if call_ids and all(cid in answered_after for cid in call_ids):
                live_call_ids.update(call_ids)
                result.append(msg)
                continue
            cleaned = {k: v for k, v in msg.items() if k != "tool_calls"}
            if not cleaned.get("content"):
                continue
            result.append(cleaned)
            continue
        if role == "user" and not is_last:
            has_assistant_after = any(
                turns[j].get("role") == "assistant" for j in range(i + 1, n)
            )
            if not has_assistant_after:
                continue
        result.append(msg)
    return result


class Conversation:
    def __init__(
        self,
        *,
        system_prompt_provider: Callable[[], str],
        max_turns: int = 10,
        inactivity_timeout_seconds: float = 300.0,
        time_provider: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        if inactivity_timeout_seconds <= 0:
            raise ValueError(
                f"inactivity_timeout_seconds must be positive, got "
                f"{inactivity_timeout_seconds}"
            )
        self._system_prompt_provider = system_prompt_provider
        self.max_turns = max_turns
        self.inactivity_timeout_seconds = inactivity_timeout_seconds
        self._time = time_provider
        self._turns: list[Message] = []
        self._last_activity: float | None = None

    # -- API --

    def add_user_turn(self, text: str) -> list[Message]:
        """Append a user message and return the full message list ready
        for OllamaClient.stream_chat. Clears history first if the
        inactivity timeout has elapsed."""
        if self._inactivity_elapsed():
            self._turns.clear()
        self._turns.append({"role": "user", "content": text})
        self._trim()
        self._last_activity = self._time()
        return self.current_messages()

    def add_assistant_turn(self, text: str) -> None:
        """Append an assistant message. Called after the LLM stream
        completes. Updates the inactivity timer so a slow response
        followed by a follow-up question doesn't fire a stale clear."""
        self._turns.append({"role": "assistant", "content": text})
        self._trim()
        self._last_activity = self._time()

    def add_tool_round_trip(
        self,
        *,
        content: str,
        exchanges: Sequence[ToolExchange],
    ) -> list[Message]:
        """Record one full tool round trip and return the message list
        ready for the next OllamaClient.stream_chat call.

        Appends, in protocol order:
          {"role": "assistant", "content": content, "tool_calls": [...]}
          {"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}
          ... one tool message per exchange ...

        `content` is whatever the model narrated alongside its tool
        calls; it is stored (empty string included) because the
        assistant message is the parent of the tool results and cannot
        be omitted, unlike a plain narration turn.

        One method rather than an add-assistant + add-tool pair on
        purpose: the two halves must land together or not at all, and a
        single append-then-trim can never observe the half-written state
        (nor can a trim run between them and orphan the results).

        Returns current_messages() so the caller can feed the model
        immediately, mirroring add_user_turn's contract.
        """
        self._turns.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": ex.call_id,
                        "type": "function",
                        "function": {
                            "name": ex.name,
                            "arguments": ex.arguments,
                        },
                    }
                    for ex in exchanges
                ],
            }
        )
        for ex in exchanges:
            self._turns.append(
                {
                    "role": "tool",
                    "tool_call_id": ex.call_id,
                    "name": ex.name,
                    "content": ex.result,
                }
            )
        self._trim()
        self._last_activity = self._time()
        return self.current_messages()

    def clear(self) -> None:
        """Wipe the turn history unconditionally. The system prompt is
        re-supplied on next current_messages() call."""
        had_turns = bool(self._turns)
        self._turns.clear()
        if had_turns:
            print("[conversation] cleared on wake")

    def maybe_clear(self, continuity_seconds: float) -> None:
        """Clear history on wake-word activation using the time-windowed rule.

        If the last assistant turn completed within continuity_seconds, the
        user is likely following up — history is kept. If more time has
        elapsed (or there is no prior history), this is a fresh session and
        history is wiped.

        Called by AudioPipeline's on_wake hook via the composition root.
        """
        if self._last_activity is None or not self._turns:
            return
        elapsed = self._time() - self._last_activity
        if elapsed > continuity_seconds:
            self.clear()
        else:
            print(
                f"[conversation] kept on wake "
                f"({elapsed:.0f}s < {continuity_seconds:.0f}s continuity window)"
            )

    def has_unanswered_user_turn(self) -> bool:
        """True when the last stored turn is a user message with no following
        assistant reply. Used by the router adapter to decide whether to store
        a tool-only spoken result as the assistant turn."""
        return bool(self._turns) and self._turns[-1].get("role") == "user"

    def current_messages(self) -> list[Message]:
        """Read-only view: [system, *filtered_history]. Returns a fresh list
        each call so callers can safely pass to mutating consumers."""
        prompt = self._system_prompt_provider()
        filtered = _filter_turns_for_llm(self._turns)
        if prompt:
            return [{"role": "system", "content": prompt}, *filtered]
        return list(filtered)

    # -- internal --

    def _inactivity_elapsed(self) -> bool:
        if self._last_activity is None:
            return False
        return (self._time() - self._last_activity) > self.inactivity_timeout_seconds

    def _trim(self) -> None:
        """Drop oldest turns down to max_turns without splitting a tool
        round trip.

        A role:"tool" message is only legal while the assistant message
        declaring its tool_call_id is still ahead of it, and tool
        messages are always appended directly after that assistant
        message. So the only way front-trimming can break the shape is
        by cutting the assistant and leaving its results behind --
        extend the cut forward over them. That can take the history
        BELOW max_turns, which is the correct trade: a short history is
        fine, a history the chat API rejects is not.

        At max_turns=1 a round trip therefore trims to nothing (the
        user turn, the assistant call and its result are three
        messages, and none can stand alone). That is a degenerate
        setting -- the default is 10, and even 3 holds a whole
        single-call round trip -- but it is bounded and well-formed
        rather than broken.
        """
        excess = len(self._turns) - self.max_turns
        if excess <= 0:
            return
        cut = excess
        while cut < len(self._turns) and self._turns[cut].get("role") == "tool":
            cut += 1
        del self._turns[:cut]
