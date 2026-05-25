"""Per-request context for the current user utterance.

The audio pipeline sets `current_user_transcription` around each
router/execute pass so tools can recover the full sentence when the LLM
passes a thin or empty tool argument.
"""

from __future__ import annotations

from contextvars import ContextVar

current_user_transcription: ContextVar[str | None] = ContextVar(
    "current_user_transcription",
    default=None,
)
