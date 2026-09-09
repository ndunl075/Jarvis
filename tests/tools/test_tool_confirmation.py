"""Tests for the requires_confirmation gate in ToolRegistry.execute().

The gate's whole value is that it sits at the single dispatch choke
point, so these tests are about execute() and nothing else — no Qt, no
threads. The Qt half (the dialog, and the audio → Qt → audio hop) is
tested in tests/ui/test_tool_confirm.py.

What is pinned here:
  - an approval runs the tool, with the validated args it was shown,
  - a denial does NOT run it and comes back as ToolResult(success=False),
    never as an exception,
  - no confirmer wired denies (fail-closed), and says so distinctly,
  - a confirmer that raises denies,
  - CancelledError from a confirmer propagates rather than becoming a
    denial,
  - a tool with the flag False never touches the confirmer at all,
  - the gate runs after validation and after the enabled check, so a
    call that could never have succeeded does not interrupt the user.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

from jarvis.core.config import ToolsConfig
from jarvis.tools.registry import (
    ConfirmationRequest,
    EmptyArgs,
    ToolConfirmer,
    ToolRegistry,
    ToolResult,
)

# --- fakes -------------------------------------------------------------


class _TypeArgs(BaseModel):
    text: str
    times: int = Field(default=1, ge=1)


class _GatedTool:
    """A tool that asks for confirmation and records whether it ran."""

    name: str = "gated"
    description: str = "Does something the user should be asked about."
    args_schema = _TypeArgs
    requires_confirmation: bool = True

    def __init__(self) -> None:
        self.ran_with: list[_TypeArgs] = []

    async def execute(self, args: _TypeArgs) -> ToolResult:
        self.ran_with.append(args)
        return ToolResult(success=True, output="done")


class _GatedWithSummary(_GatedTool):
    name: str = "gated_summary"

    def confirmation_summary(self, args: _TypeArgs) -> str:
        return f"Type {len(args.text)} characters."


class _GatedWithBrokenSummary(_GatedTool):
    name: str = "gated_broken_summary"

    def confirmation_summary(self, args: _TypeArgs) -> str:
        raise RuntimeError("intentional test boom")


class _UngatedTool:
    name: str = "ungated"
    description: str = "Ordinary tool."
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    def __init__(self) -> None:
        self.ran = 0

    async def execute(self, args: EmptyArgs) -> ToolResult:
        self.ran += 1
        return ToolResult(success=True, output="ok")


class _RecordingConfirmer:
    """Answers with a fixed verdict and remembers what it was asked."""

    def __init__(self, verdict: bool = True) -> None:
        self.verdict = verdict
        self.requests: list[ConfirmationRequest] = []

    async def confirm(self, request: ConfirmationRequest) -> bool:
        self.requests.append(request)
        return self.verdict


class _RaisingConfirmer:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    async def confirm(self, request: ConfirmationRequest) -> bool:
        self.calls += 1
        raise self.exc


def _registry(confirmer=None, **kwargs) -> ToolRegistry:
    return ToolRegistry(ToolsConfig(**kwargs), confirmer=confirmer)


def _register(reg: ToolRegistry, tool) -> None:
    """register() is typed for `Tool`, and pyright rejects every duck-typed
    fake in this file over `args_schema` invariance. One ignore here beats
    one on every call site."""
    reg.register(tool)  # type: ignore[arg-type]

# --- protocol surface ---------------------------------------------------


def test_recording_confirmer_satisfies_the_protocol():
    """ToolConfirmer is runtime_checkable so a fake this small is a
    legitimate implementation, not a mock that happens to work."""
    assert isinstance(_RecordingConfirmer(), ToolConfirmer)


def test_registry_defaults_to_no_confirmer():
    """Every pre-existing test in the suite constructs ToolRegistry with
    one positional argument; that must keep meaning "no approval UI"."""
    assert ToolRegistry(ToolsConfig()).confirmer is None


def test_set_confirmer_installs_and_clears():
    reg = _registry()
    confirmer = _RecordingConfirmer()
    reg.set_confirmer(confirmer)
    assert reg.confirmer is confirmer
    reg.set_confirmer(None)
    assert reg.confirmer is None


# --- approval / denial --------------------------------------------------


@pytest.mark.asyncio
async def test_approval_runs_the_tool():
    confirmer = _RecordingConfirmer(verdict=True)
    reg = _registry(confirmer)
    tool = _GatedTool()
    _register(reg, tool)

    result = await reg.execute("gated", {"text": "hello"})

    assert result == ToolResult(success=True, output="done")
    assert [a.text for a in tool.ran_with] == ["hello"]
    assert len(confirmer.requests) == 1


@pytest.mark.asyncio
async def test_denial_does_not_run_the_tool_and_returns_a_tool_result():
    """A denial is a normal failure, not an exception. The router speaks
    ToolResult.error back and the model can read it and try something
    else; a raise would unwind the whole round instead."""
    confirmer = _RecordingConfirmer(verdict=False)
    reg = _registry(confirmer)
    tool = _GatedTool()
    _register(reg, tool)

    result = await reg.execute("gated", {"text": "hello"})

    assert tool.ran_with == []
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "gated" in (result.error or "")
    assert "not approved" in (result.error or "")


@pytest.mark.asyncio
async def test_no_confirmer_denies():
    """Fail closed. A registry with no approval UI — every headless test,
    and the app before Qt exists — must refuse a confirmable tool rather
    than run it unguarded. The alternative default (allow when nothing is
    watching) makes the flag worthless in exactly the configuration an
    attacker would prefer."""
    reg = _registry()
    tool = _GatedTool()
    _register(reg, tool)

    result = await reg.execute("gated", {"text": "hello"})

    assert tool.ran_with == []
    assert result.success is False
    assert "no confirmation prompt is available" in (result.error or "")


@pytest.mark.asyncio
async def test_no_confirmer_denial_reads_differently_from_a_user_denial():
    """The two refusals reach the model as text, so they have to be
    distinguishable: "you said no" is a decision, "nothing asked you" is
    a misconfiguration."""
    tool_a, tool_b = _GatedTool(), _GatedTool()
    tool_b.name = "gated_b"

    silent = _registry()
    _register(silent, tool_a)
    denied = _registry(_RecordingConfirmer(verdict=False))
    _register(denied, tool_b)

    a = await silent.execute("gated", {"text": "x"})
    b = await denied.execute("gated_b", {"text": "x"})

    assert a.error != b.error


@pytest.mark.asyncio
async def test_confirmer_that_raises_denies():
    confirmer = _RaisingConfirmer(RuntimeError("dialog exploded"))
    reg = _registry(confirmer)
    tool = _GatedTool()
    _register(reg, tool)

    result = await reg.execute("gated", {"text": "hello"})

    assert confirmer.calls == 1
    assert tool.ran_with == []
    assert result.success is False


@pytest.mark.asyncio
async def test_confirmer_returning_a_truthy_non_true_denies():
    """`approved is True` and not `if approved`. A confirmer that returns
    a coroutine, a sentinel, or anything else it did not mean as an
    approval is not an approval."""

    class _Sloppy:
        async def confirm(self, request: ConfirmationRequest):
            return "yes"

    reg = _registry(_Sloppy())
    tool = _GatedTool()
    _register(reg, tool)

    result = await reg.execute("gated", {"text": "hello"})

    assert tool.ran_with == []
    assert result.success is False


@pytest.mark.asyncio
async def test_cancellation_during_confirmation_propagates():
    """CancelledError is the one thing the gate does not swallow — same
    contract execute() already had for the tool call itself. The router
    needs to see it to unwind its own coroutine."""
    reg = _registry(_RaisingConfirmer(asyncio.CancelledError()))
    tool = _GatedTool()
    _register(reg, tool)

    with pytest.raises(asyncio.CancelledError):
        await reg.execute("gated", {"text": "hello"})
    assert tool.ran_with == []


# --- the ungated path is untouched --------------------------------------


@pytest.mark.asyncio
async def test_ungated_tool_never_consults_the_confirmer():
    """The flag being False must cost nothing: no prompt, no await of a
    cross-thread hop, no added latency on "volume up"."""
    confirmer = _RecordingConfirmer(verdict=False)
    reg = _registry(confirmer)
    tool = _UngatedTool()
    _register(reg, tool)

    result = await reg.execute("ungated", {})

    assert result.success is True
    assert tool.ran == 1
    assert confirmer.requests == []


@pytest.mark.asyncio
async def test_tool_without_the_attribute_at_all_is_not_gated():
    """getattr(..., False) in execute(): a duck-typed object that omits
    the protocol member behaves as it always did rather than crashing the
    dispatch path."""

    class _Legacy:
        name = "legacy"
        description = "no flag declared"
        args_schema = EmptyArgs

        async def execute(self, args: EmptyArgs) -> ToolResult:
            return ToolResult(success=True, output="ran")

    reg = _registry(_RecordingConfirmer(verdict=False))
    _register(reg, _Legacy())

    assert (await reg.execute("legacy", {})).success is True


# --- gate ordering ------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_args_are_rejected_without_prompting():
    """Validation first. Interrupting the user to approve a call that
    could never have run is pure noise — and worse, it trains them to
    approve without reading."""
    confirmer = _RecordingConfirmer(verdict=True)
    reg = _registry(confirmer)
    tool = _GatedTool()
    _register(reg, tool)

    result = await reg.execute("gated", {"text": "hi", "times": 0})

    assert result.success is False
    assert "invalid args" in (result.error or "")
    assert confirmer.requests == []
    assert tool.ran_with == []


@pytest.mark.asyncio
async def test_disabled_tool_is_rejected_without_prompting():
    confirmer = _RecordingConfirmer(verdict=True)
    reg = _registry(confirmer, enabled={"gated": False})
    _register(reg, _GatedTool())

    result = await reg.execute("gated", {"text": "hi"})

    assert result.success is False
    assert "disabled" in (result.error or "")
    assert confirmer.requests == []


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected_without_prompting():
    confirmer = _RecordingConfirmer(verdict=True)
    reg = _registry(confirmer)

    result = await reg.execute("nope", {})

    assert result.success is False
    assert confirmer.requests == []


# --- what the user is shown ---------------------------------------------


@pytest.mark.asyncio
async def test_request_carries_the_name_summary_and_validated_args():
    """The prompt shows the *validated* arguments — defaults filled in —
    because that is what the tool will actually receive."""
    confirmer = _RecordingConfirmer(verdict=True)
    reg = _registry(confirmer)
    _register(reg, _GatedWithSummary())

    await reg.execute("gated_summary", {"text": "abcd"})

    request = confirmer.requests[0]
    assert request.tool_name == "gated_summary"
    assert request.summary == "Type 4 characters."
    assert request.arguments == {"text": "abcd", "times": 1}


@pytest.mark.asyncio
async def test_summary_falls_back_to_the_description():
    confirmer = _RecordingConfirmer(verdict=True)
    reg = _registry(confirmer)
    tool = _GatedTool()
    _register(reg, tool)

    await reg.execute("gated", {"text": "abcd"})

    assert confirmer.requests[0].summary == tool.description


@pytest.mark.asyncio
async def test_a_summary_hook_that_raises_still_prompts():
    """A cosmetic failure must not take the gate down with it — in
    either direction. It neither denies nor skips the prompt."""
    confirmer = _RecordingConfirmer(verdict=True)
    reg = _registry(confirmer)
    tool = _GatedWithBrokenSummary()
    _register(reg, tool)

    result = await reg.execute("gated_broken_summary", {"text": "abcd"})

    assert result.success is True
    assert confirmer.requests[0].summary == tool.description
