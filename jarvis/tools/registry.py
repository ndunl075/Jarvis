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

- requires_confirmation IS wired, and execute() is where it is
  enforced. Every tool call — pattern-routed, LLM-chosen via
  IntentRouter._run_tools, or command-palette — funnels through
  execute(), so gating it there gates all of them; there is no second
  dispatch path a caller could reach around.

  The UX is a modal Qt dialog (jarvis/ui/tool_confirm.py), NOT the
  cancel-window or voice-confirmation designs this was originally
  deferred for. Both of those needed the audio path: the cancel window
  needs barge-in (still disabled by speaker → mic feedback) and voice
  confirmation needs STT during TTS. A dialog on the Qt main thread
  needs neither, so the blocker does not apply to it. The audio thread
  awaits the verdict; the bridge is described in ui/tool_confirm.py.

  The gate FAILS CLOSED. A registry built without a confirmer — every
  headless test, and the window before the Qt UI exists — denies any
  tool that asks for confirmation rather than running it unguarded. A
  denial is a ToolResult(success=False, error=...) like every other
  failure mode here, never an exception: the router speaks it back and
  the model can read it and choose differently.

  Tools opt in individually. Today that is type_into_active_window
  (synthesises arbitrary keystrokes into whatever window has focus —
  a terminal, an address bar, a password field) and any MCP tool whose
  server config sets requires_confirmation. See the note on
  lock_screen for a tool that was considered and deliberately left
  ungated.

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


# --- confirmation (the requires_confirmation gate) --------------------


@dataclass(frozen=True, slots=True)
class ConfirmationRequest:
    """What the user is being asked to approve.

    A plain value object so the registry never imports Qt: the audio
    thread builds one of these and hands it to whatever ToolConfirmer is
    wired, which in the desktop app is jarvis.ui.tool_confirm. Tests wire
    a two-line fake instead.

    `summary` is the sentence the user reads. It comes from the tool's
    optional confirmation_summary(args) hook, falling back to the tool
    description — the description is written for the LLM, so a tool whose
    risk depends on its arguments ("type <this> into <that>") should
    implement the hook. `arguments` is shown verbatim underneath: the
    summary says what kind of thing is about to happen, the arguments say
    exactly what."""

    tool_name: str
    summary: str
    arguments: dict


@runtime_checkable
class ToolConfirmer(Protocol):
    """Asks the user to approve one tool call and returns their verdict.

    Awaited on the audio loop, so an implementation that has to reach
    another thread must do so without blocking this one. Contract:

      - returns True only on an explicit approval; anything else —
        denial, timeout, dismissal, a wedged UI — is False,
      - never raises for a routine outcome (the registry treats a raise
        as a denial anyway), and
      - propagates CancelledError, so a cancelled interaction unwinds
        instead of stalling behind a prompt nobody is looking at."""

    async def confirm(self, request: ConfirmationRequest) -> bool: ...


def _confirmation_summary(tool: Tool, args: BaseModel) -> str:
    """The sentence shown to the user for `tool`.

    The optional confirmation_summary(args) hook wins; a tool that does
    not define one (every MCP tool, by construction) falls back to its
    description. A hook that raises is not allowed to take the gate down
    with it — the fallback is used and the prompt still appears."""
    build = getattr(tool, "confirmation_summary", None)
    if callable(build):
        try:
            text = build(args)
        except Exception:
            log.exception(
                "confirmation_summary raised for %r; using description",
                getattr(tool, "name", "<unnamed>"),
            )
        else:
            if isinstance(text, str) and text.strip():
                return text.strip()
    return tool.description


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

    def __init__(
        self,
        tools_config: ToolsConfig,
        *,
        confirmer: ToolConfirmer | None = None,
    ) -> None:
        self._tools_config = tools_config
        self._tools: dict[str, Tool] = {}
        self._confirmer = confirmer

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

    # -- confirmation -------------------------------------------------

    @property
    def confirmer(self) -> ToolConfirmer | None:
        return self._confirmer

    def set_confirmer(self, confirmer: ToolConfirmer | None) -> None:
        """Install (or clear) the approval UI.

        A setter and not a constructor-only argument because of the build
        order in jarvis/app.py: the registry is part of the audio stack,
        which is composed before the QApplication the dialog needs
        exists. The composition root installs the confirmer as soon as
        Qt is up and before the audio thread starts, so the fail-closed
        window is never one a spoken command can land in."""
        self._confirmer = confirmer

    async def _confirm(self, name: str, tool: Tool, args: BaseModel) -> str | None:
        """Ask the user to approve `name`.

        Returns None on approval, or the reason for the refusal — which
        is what the caller puts in the ToolResult, so "nothing asked you"
        and "you said no" read differently in the log and in the model's
        tool result.

        Fails closed on every path. No confirmer wired, a confirmer that
        raises, a confirmer that returns something other than True — all
        deny. The one thing that does not deny is CancelledError, which
        belongs to the caller."""
        confirmer = self._confirmer
        if confirmer is None:
            log.warning(
                "tool %r requires confirmation but no confirmer is wired; "
                "refusing", name,
            )
            return "no confirmation prompt is available"
        try:
            arguments = args.model_dump(mode="json")
        except Exception:
            # A tool whose args model will not serialise still gets a
            # prompt — just without the argument detail. Denying here
            # would be fail-closed but would also make the tool
            # permanently unusable for a purely cosmetic reason.
            log.debug("could not serialise args for %r", name, exc_info=True)
            arguments = {}
        request = ConfirmationRequest(
            tool_name=name,
            summary=_confirmation_summary(tool, args),
            arguments=arguments,
        )
        try:
            approved = await confirmer.confirm(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("confirmer raised for %r; treating as denied", name)
            return "the confirmation prompt failed"
        if approved is True:
            return None
        return "it was not approved"

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
        modes — unknown, disabled, invalid args, tool crash, refused at
        the confirmation prompt — return ToolResult(success=False,
        error=...) so the SpeakIntent layer has a single shape to format.

        The confirmation gate sits after validation and before dispatch:
        after, so a malformed call is rejected without interrupting the
        user for a tool that could never have run; before, so approval is
        a precondition of the tool executing at all."""
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
        # getattr, not attribute access: `requires_confirmation` is part
        # of the Tool protocol, but a duck-typed object that omits it
        # would otherwise crash the dispatch path rather than be gated.
        # Absent means "did not ask to be gated", which is the same
        # answer as False and keeps every ungated tool on its old path —
        # no confirmer consulted, no work done, no added latency.
        if getattr(tool, "requires_confirmation", False):
            refusal = await self._confirm(name, tool, args)
            if refusal is not None:
                log.info("tool %r did not run: %s", name, refusal)
                return ToolResult(
                    success=False,
                    error=f"tool {name!r} did not run: {refusal}",
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
