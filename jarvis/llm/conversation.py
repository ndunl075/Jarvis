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
"""

from __future__ import annotations

import time
from collections.abc import Callable

# A single message in OpenAI/Ollama chat format.
Message = dict


def _filter_turns_for_llm(turns: list[Message]) -> list[Message]:
    """Strip tool-call artifacts before sending history to the LLM.

    Removes:
    - role:"tool" messages (tool result payloads the LLM should not re-see)
    - assistant messages that contain only tool_calls with no spoken content
    - tool_calls key from assistant messages that do have spoken content
    - orphaned user turns: user messages with no following assistant turn,
      excluding the last message (which is the current turn being sent)
    """
    result: list[Message] = []
    n = len(turns)
    for i, msg in enumerate(turns):
        role = msg.get("role")
        is_last = i == n - 1
        if role == "tool":
            continue
        if role == "assistant" and "tool_calls" in msg:
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
        excess = len(self._turns) - self.max_turns
        if excess > 0:
            del self._turns[:excess]
