"""Tests for jarvis.tools.registry.

Covers protocol satisfaction, registration semantics (including the
no-silent-overwrite collision policy), the config-driven enable filter
with fresh-per-execute re-check, args validation at the registry
boundary, the OpenAI/Ollama function schema shape, and exception
isolation on tool crashes."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from jarvis.core.config import ToolsConfig
from jarvis.tools.registry import (
    TOOL_NAME_REGEX,
    EmptyArgs,
    Tool,
    ToolNameCollisionError,
    ToolRegistry,
    ToolResult,
)

# --- fakes -------------------------------------------------------------


class _EchoArgs(BaseModel):
    text: str
    times: int = Field(default=1, ge=1)


class _EchoTool:
    name: str = "echo"
    description: str = "Echo the text some number of times."
    args_schema = _EchoArgs
    requires_confirmation: bool = False

    def __init__(self) -> None:
        self.executed_with: list[_EchoArgs] = []

    async def execute(self, args: _EchoArgs) -> ToolResult:
        self.executed_with.append(args)
        return ToolResult(
            success=True, output=" ".join([args.text] * args.times)
        )


class _NoArgsTool:
    name: str = "ping"
    description: str = "Returns pong."
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    async def execute(self, args: EmptyArgs) -> ToolResult:
        return ToolResult(success=True, output="pong")


class _CrashingTool:
    name: str = "crash"
    description: str = "Always crashes."
    args_schema = EmptyArgs
    requires_confirmation: bool = False

    async def execute(self, args: EmptyArgs) -> ToolResult:
        raise RuntimeError("intentional test boom")


# --- module surface ----------------------------------------------------


def test_tool_name_regex_accepts_valid_names():
    for name in ("echo", "fs_read_file", "open-url", "x", "a" * 64):
        assert TOOL_NAME_REGEX.match(name), name


def test_tool_name_regex_rejects_invalid_names():
    for name in ("", " ", "with.dot", "with space", "a" * 65, "中文"):
        assert TOOL_NAME_REGEX.match(name) is None, name


# --- protocol smoke ----------------------------------------------------


def test_concrete_tools_satisfy_protocol():
    """Tool is @runtime_checkable so isinstance works structurally."""
    assert isinstance(_EchoTool(), Tool)
    assert isinstance(_NoArgsTool(), Tool)


def test_phase_4_tools_default_requires_confirmation_false():
    """Per design note: requires_confirmation is in the protocol but is
    not yet wired to any UX (deferred to Phase 6+). All Phase 4 tools
    must ship with the flag False; this test guards the contract on the
    in-tree fakes that stand in for them at this layer."""
    assert _EchoTool().requires_confirmation is False
    assert _NoArgsTool().requires_confirmation is False


# --- registration ------------------------------------------------------


def test_register_and_get():
    reg = ToolRegistry(ToolsConfig())
    tool = _EchoTool()
    reg.register(tool)
    assert reg.get("echo") is tool


def test_get_unknown_returns_none():
    reg = ToolRegistry(ToolsConfig())
    assert reg.get("nonesuch") is None


def test_register_duplicate_name_raises_collision():
    """ToolNameCollisionError lets the caller decide policy. Built-ins
    register first at startup; MCP wrappers catch and drop the single
    colliding tool from the offending server (no silent overwrite)."""
    reg = ToolRegistry(ToolsConfig())
    reg.register(_EchoTool())
    with pytest.raises(ToolNameCollisionError):
        reg.register(_EchoTool())


def test_register_built_in_then_mcp_clobber_is_blocked():
    """Concrete walk-through of the policy: built-in registers first
    and wins; a second tool with the same name (simulating an MCP
    wrapper) is refused so the caller can log + drop just that one."""
    reg = ToolRegistry(ToolsConfig())
    builtin = _EchoTool()
    reg.register(builtin)

    class _MCPEcho:
        name: str = "echo"
        description: str = "MCP echo from a remote server."
        args_schema = _EchoArgs
        requires_confirmation: bool = False

        async def execute(self, args: _EchoArgs) -> ToolResult:
            return ToolResult(success=True, output="from-mcp")

    with pytest.raises(ToolNameCollisionError):
        reg.register(_MCPEcho())
    # Built-in is intact.
    assert reg.get("echo") is builtin


def test_unregister_then_reregister_ok():
    reg = ToolRegistry(ToolsConfig())
    reg.register(_EchoTool())
    reg.unregister("echo")
    reg.register(_EchoTool())  # must not raise
    assert reg.get("echo") is not None


def test_unregister_unknown_is_noop():
    reg = ToolRegistry(ToolsConfig())
    reg.unregister("nonesuch")  # must not raise


# --- enabled filter ----------------------------------------------------


def test_list_enabled_includes_unconfigured_tools():
    """Convention: a tool absent from config.tools.enabled is enabled."""
    reg = ToolRegistry(ToolsConfig())
    reg.register(_EchoTool())
    assert [t.name for t in reg.list_enabled()] == ["echo"]


def test_list_enabled_skips_disabled_tools():
    reg = ToolRegistry(ToolsConfig(enabled={"echo": False}))
    reg.register(_EchoTool())
    assert reg.list_enabled() == []


def test_list_enabled_keeps_explicitly_enabled_tools():
    reg = ToolRegistry(ToolsConfig(enabled={"echo": True}))
    reg.register(_EchoTool())
    assert [t.name for t in reg.list_enabled()] == ["echo"]


# --- as_openai_functions ----------------------------------------------


def test_as_openai_functions_emits_ollama_compatible_schema():
    reg = ToolRegistry(ToolsConfig())
    reg.register(_EchoTool())
    funcs = reg.as_openai_functions()
    assert len(funcs) == 1
    entry = funcs[0]
    assert entry["type"] == "function"
    fn = entry["function"]
    assert fn["name"] == "echo"
    assert "Echo" in fn["description"]
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "text" in params["properties"]
    assert "times" in params["properties"]
    assert "text" in params["required"]


def test_as_openai_functions_excludes_disabled_tools():
    reg = ToolRegistry(ToolsConfig(enabled={"echo": False}))
    reg.register(_EchoTool())
    reg.register(_NoArgsTool())
    names = [e["function"]["name"] for e in reg.as_openai_functions()]
    assert names == ["ping"]


def test_as_openai_functions_handles_empty_args():
    reg = ToolRegistry(ToolsConfig())
    reg.register(_NoArgsTool())
    [entry] = reg.as_openai_functions()
    params = entry["function"]["parameters"]
    assert params["type"] == "object"
    assert params.get("properties", {}) == {}


# --- execute -----------------------------------------------------------


async def test_execute_validates_and_dispatches():
    reg = ToolRegistry(ToolsConfig())
    tool = _EchoTool()
    reg.register(tool)
    result = await reg.execute("echo", {"text": "hi", "times": 2})
    assert result == ToolResult(success=True, output="hi hi")
    assert len(tool.executed_with) == 1
    assert tool.executed_with[0].text == "hi"
    assert tool.executed_with[0].times == 2


async def test_execute_unknown_tool_returns_error():
    reg = ToolRegistry(ToolsConfig())
    result = await reg.execute("nonesuch", {})
    assert not result.success
    assert "nonesuch" in (result.error or "")


async def test_execute_invalid_args_returns_error_not_raise():
    """Bad raw args from the LLM must surface as ToolResult.error so
    the SpeakIntent layer can speak the error. Tools must never see
    invalid args reaching their execute()."""
    reg = ToolRegistry(ToolsConfig())
    tool = _EchoTool()
    reg.register(tool)
    result = await reg.execute("echo", {"text": 42, "times": -1})
    assert not result.success
    assert result.error is not None
    assert tool.executed_with == []  # never reached the tool


async def test_execute_disabled_tool_returns_error():
    reg = ToolRegistry(ToolsConfig(enabled={"echo": False}))
    tool = _EchoTool()
    reg.register(tool)
    result = await reg.execute("echo", {"text": "hi"})
    assert not result.success
    assert "disabled" in (result.error or "")
    assert tool.executed_with == []


async def test_execute_isolates_tool_exception():
    reg = ToolRegistry(ToolsConfig())
    reg.register(_CrashingTool())
    result = await reg.execute("crash", {})
    assert not result.success
    assert "crashed" in (result.error or "")


async def test_execute_rechecks_enabled_per_call():
    """Defends the documented bug: a user can disable a tool between
    when the LLM was given the function list and when execute() is
    dispatched. The enable check must run fresh per call, not be cached
    from as_openai_functions() time. Regression of this test means a
    disabled tool can be invoked mid-stream."""
    cfg = ToolsConfig()
    reg = ToolRegistry(cfg)
    tool = _EchoTool()
    reg.register(tool)
    # 1. Enabled at function-list time.
    assert [e["function"]["name"] for e in reg.as_openai_functions()] == ["echo"]
    # 2. User flips it off mid-stream (e.g. via settings UI).
    cfg.enabled["echo"] = False
    # 3. The pending dispatch must be rejected, not race-executed.
    result = await reg.execute("echo", {"text": "hi"})
    assert not result.success
    assert "disabled" in (result.error or "")
    assert tool.executed_with == []


async def test_execute_no_args_tool():
    reg = ToolRegistry(ToolsConfig())
    reg.register(_NoArgsTool())
    result = await reg.execute("ping", {})
    assert result == ToolResult(success=True, output="pong")


# --- setup_local_tools wiring -----------------------------------------


def test_setup_local_tools_registers_the_phase_4_set():
    """The built-in tool set is pinned so adding/removing a tool is a
    deliberate change reviewers can see."""
    from jarvis.tools import setup_local_tools
    reg = ToolRegistry(ToolsConfig())
    setup_local_tools(reg)
    names = sorted(t.name for t in reg.list_enabled())
    assert names == [
        "clipboard",
        "close_app",
        "get_weather",
        "launch_steam_game",
        "launch_workspace",
        "list_directory",
        "lock_screen",
        "open_app",
        "open_url",
        "play_youtube_music",
        # Renamed from "system_stats" because the short name read as a
        # generic "tell me about the system" probe and the LLM was
        # firing it for ambiguous transcriptions ("jot", "in chat",
        # "capital of france"). The verbose action-specific name keeps
        # it out of the model's fallback set.
        "report_cpu_and_memory_percentages",
        "screenshot",
        # type_into_active_window is omitted: disabled by default in ToolsConfig
        "volume",
    ]


def test_setup_local_tools_idempotent_into_fresh_registry():
    """Setting up twice into the same registry must raise — the policy
    is no-silent-overwrite. Caller (composition root) calls once."""
    from jarvis.tools import setup_local_tools
    reg = ToolRegistry(ToolsConfig())
    setup_local_tools(reg)
    with pytest.raises(ToolNameCollisionError):
        setup_local_tools(reg)


def test_all_local_tools_default_requires_confirmation_false():
    """Phase 4 contract: requires_confirmation UX is deferred. No
    shipped tool may set the flag True."""
    from jarvis.tools import setup_local_tools
    reg = ToolRegistry(ToolsConfig())
    setup_local_tools(reg)
    for tool in reg.list_enabled():
        assert tool.requires_confirmation is False, (
            f"{tool.name} ships with requires_confirmation=True but the "
            "UX is not yet wired (see registry.py header)"
        )
