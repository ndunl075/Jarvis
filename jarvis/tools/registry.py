"""Tool protocol and registry — the single namespace through which the LLM
discovers callable tools and through which the router executes them.

Local tools (jarvis/tools/local/*.py) register at composition time; MCP-
adapted tools register lazily via jarvis/tools/mcp_client.py as remote
servers come online. All tools satisfy the same Tool protocol; the
registry does not distinguish local from remote.

Design rationale (Phase 4 Task 1 design note, approved):

- Validation lives at the registry boundary. Tools see an already-
  validated pydantic BaseModel via execute(). Bad raw args surface as
  ToolResult(success=False, error=...), never as exceptions, so the
  router can speak the failure back without try/except plumbing.

- Name collisions raise ToolNameCollisionError. Built-in tools register first
  at startup so they always win; MCP wrappers catch the exception and
  drop the colliding tool from the offending server only. No silent
  overwrites — a clobber on a tool the LLM trusts is a footgun.

- list_enabled() is the source of truth for visibility. as_openai_
  functions() filters through it; execute() re-checks it fresh per call.
  A config edit between the LLM choosing a tool and execute() dispatching
  it must drop the call cleanly rather than race-execute the disabled
  tool. Tested explicitly.

- requires_confirmation is in the protocol but is NOT yet wired to any
  UX. Deferred to Phase 6+ when hotkey-based cancellation lands: the
  cancel-window approach needs barge-in (currently disabled by speaker
  → mic feedback) and the voice-confirmation alternative has the same
  STT-during-TTS issue. All Phase 4 tools ship with the flag False; no
  destructive tools (files.delete, etc.) are included.

- MCP tool name sanitisation lives in mcp_client.py (Task 3). The
  shared TOOL_NAME_REGEX below is the validity contract: lowercase
  alphanumerics, underscore, hyphen; 1–64 chars. Dots get replaced with
  underscores at adapter time (fs.read_file -> fs_read_file) because the
  Ollama/OpenAI function-name surface forbids them.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from jarvis.core.config import ToolsConfig

log = logging.getLogger(__name__)

# Function-name regex enforced by OpenAI/Ollama tool calling. Shared so the
# constraint lives in one place; MCPTool sanitisation (Task 3) checks
# against it and raises if the resulting name is empty or invalid.
TOOL_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class EmptyArgs(BaseModel):
    """Sentinel pydantic model for tools that take no arguments. Shared
    so no-args tools don't each declare an empty BaseModel subclass."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Return value of Tool.execute.

    `output` is the human-facing result — str preferred (the router
    speaks it back); dict allowed for structured returns the UI might
    consume. `error` carries the failure message; the two fields are
    kept separate so the SpeakIntent layer can format them independently
    rather than reparse a unified payload."""

    success: bool
    output: str | dict | None = None
    error: str | None = None


@runtime_checkable
class Tool(Protocol):
    """The contract every callable tool — local or MCP-adapted — must
    satisfy.

    Protocol (not ABC) so MCP wrappers fulfil it structurally without
    inheriting our base. @runtime_checkable lets registry / tests do
    isinstance(x, Tool) for protocol-conformance smoke tests."""

    name: str
    description: str
    args_schema: type[BaseModel]
    requires_confirmation: bool

    async def execute(self, args: BaseModel) -> ToolResult: ...


# --- voice patterns (optional tool capability) ------------------------

# Priority ordering for the router's pattern layer.
#
# The layer is FIRST-MATCH-WINS, so priority *is* precedence: a lower
# number is tried earlier and therefore wins an utterance both patterns
# could match. Declaration order and registration order deliberately do
# not decide anything — a tool declaring a pattern cannot accidentally
# jump the queue by being imported or registered sooner.
#
# Numbering convention: the built-in table is spaced by 10 so a new
# pattern can be slotted between any two existing ones without renumbering.
# The two constants below name the only two bands that carry meaning
# beyond "earlier than the next one":
#
#   PRIORITY_DEFAULT   a tightly-anchored pattern with no known overlap.
#                      Safe for a new tool that matches a distinctive
#                      phrase ("show the fridge inventory").
#   PRIORITY_CATCH_ALL a permissive "<verb> <anything>" pattern that must
#                      lose to every more specific phrasing. `open_app`'s
#                      ^open\s+(.+)$ lives here: it is the reason
#                      "open my notes" opens notes rather than launching
#                      an app named "my notes".
#
# Anything that can be shadowed by, or can shadow, another tool's pattern
# needs an explicit number and a comment saying what it must beat.
PRIORITY_DEFAULT = 500
PRIORITY_CATCH_ALL = 900


def _no_args(match: re.Match[str]) -> dict:  # noqa: ARG001
    """Default arg builder: the pattern is a bare command with no captures."""
    return {}


@dataclass(frozen=True, slots=True)
class VoicePattern:
    """A regex that routes an utterance straight to the owning tool,
    skipping the LLM entirely.

    Declared as a `voice_patterns` class attribute on the tool so the
    phrasing lives next to the implementation it dispatches to, rather
    than in a hand-maintained table in the router.

    `regex` is matched (``re.Pattern.match``) against the *normalized*
    transcription — lowercased, trailing punctuation stripped, leading
    fillers ("hey jarvis", "could you please") peeled off. Write patterns
    lowercase and anchored at both ends unless a trailing capture is
    intended.

    `args` maps the successful match to the tool's argument dict; the
    default produces ``{}`` for no-argument commands.

    `priority` is the router's ordering key — see PRIORITY_DEFAULT /
    PRIORITY_CATCH_ALL above. It is required precisely because it must be
    a deliberate decision."""

    regex: str
    priority: int
    args: Callable[[re.Match[str]], dict] = field(default=_no_args)


@runtime_checkable
class VoiceRoutable(Protocol):
    """Optional extension to `Tool`: a tool that can be reached by the
    router's deterministic pattern layer as well as by LLM tool-calling.

    Deliberately a SEPARATE protocol rather than an optional member on
    `Tool`. `Tool` is @runtime_checkable, and isinstance() against a
    runtime_checkable Protocol is a hasattr() check over every declared
    member — adding `voice_patterns` to `Tool` would make
    ``isinstance(mcp_tool, Tool)`` start returning False for every
    MCP-adapted tool, which structurally satisfies `Tool` today and has
    no voice patterns by construction. Keeping it separate means MCP
    tools stay valid `Tool`s and are simply not `VoiceRoutable`."""

    name: str
    voice_patterns: ClassVar[tuple[VoicePattern, ...]]


class ToolNameCollisionError(ValueError):
    """Raised by ToolRegistry.register when a tool with the same name is
    already registered. Caller decides policy: built-ins (which register
    first) treat this as a programming bug; MCP wrappers catch + log +
    drop the single colliding tool from the offending server."""


class ToolRegistry:
    """Flat namespace of registered tools. Constructed with a ToolsConfig
    so it can consult the enable/disable map fresh on every call."""

    def __init__(self, tools_config: ToolsConfig) -> None:
        self._tools_config = tools_config
        self._tools: dict[str, Tool] = {}

    # -- registration -------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Add a tool. Raises ToolNameCollisionError if `tool.name` is taken.
        See module docstring on the no-silent-overwrite policy."""
        if tool.name in self._tools:
            existing = type(self._tools[tool.name]).__name__
            raise ToolNameCollisionError(
                f"tool name {tool.name!r} already registered "
                f"(existing: {existing})"
            )
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    # -- visibility ---------------------------------------------------

    def _is_enabled(self, name: str) -> bool:
        # Config convention (ToolsConfig.enabled): absence == enabled.
        return self._tools_config.enabled.get(name, True)

    def list_enabled(self) -> list[Tool]:
        return [t for n, t in self._tools.items() if self._is_enabled(n)]

    def as_openai_functions(self) -> list[dict]:
        """The shape OllamaClient.stream_chat expects as `tools=`. Ollama
        follows the OpenAI tool-calling schema:

            [{"type": "function",
              "function": {"name": ..., "description": ..., "parameters": <JSONSchema>}}, ...]

        Pydantic emits JSONSchema with $defs for nested models; Ollama
        accepts those. Filters through list_enabled, so disabled tools
        are invisible to the LLM."""
        out: list[dict] = []
        for tool in self.list_enabled():
            out.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_schema.model_json_schema(),
                },
            })
        return out

    # -- execution ----------------------------------------------------

    async def execute(self, name: str, raw_args: dict) -> ToolResult:
        """Validate `raw_args` against the tool's schema and dispatch.

        Re-checks list_enabled at call time, not at as_openai_functions
        time, so a config edit between LLM tool-choice and dispatch
        reliably drops the call rather than racing it through. All error
        modes — unknown, disabled, invalid args, tool crash — return
        ToolResult(success=False, error=...) so the SpeakIntent layer
        has a single shape to format."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"unknown tool {name!r}")
        if not self._is_enabled(name):
            return ToolResult(success=False, error=f"tool {name!r} is disabled")
        try:
            args = tool.args_schema.model_validate(raw_args)
        except ValidationError as e:
            return ToolResult(
                success=False,
                error=f"invalid args for {name!r}: "
                      f"{e.errors(include_url=False)}",
            )
        try:
            return await tool.execute(args)
        except asyncio.CancelledError:
            # Cancellation propagates — the caller (router) needs to see
            # it to unwind its own coroutine, not get a synthesised
            # ToolResult that masks the cancel.
            raise
        except Exception as e:
            log.exception("tool %r raised", name)
            return ToolResult(
                success=False, error=f"tool {name!r} crashed: {e}"
            )
