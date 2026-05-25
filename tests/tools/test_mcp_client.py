"""Tests for jarvis.tools.mcp_client.

The mcp SDK is mocked at the import boundary inside MCPServerConnection.
connect() — we patch `mcp.client.streamable_http.streamablehttp_client`
and `mcp.ClientSession` via sys.modules fakes so these tests never need a
running MCP server or the real SDK transport.
"""

from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from jarvis.core.config import MCPServerConfig, ToolsConfig
from jarvis.tools.mcp_client import (
    MCPManager,
    MCPServerConnection,
    MCPTool,
    _model_from_input_schema,
    _override_url_port,
    _read_trayce_credentials,
    _sanitize_tool_name,
    resolve_trayce_endpoint,
)
from jarvis.tools.registry import ToolRegistry, ToolResult


# --- credential reading ------------------------------------------------


def test_read_credentials_missing_localappdata(monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert _read_trayce_credentials() == (None, None)


def test_read_credentials_both_files_present(monkeypatch, tmp_path):
    trayce = tmp_path / "Trayce"
    trayce.mkdir()
    (trayce / "port.txt").write_text("52001\n")
    (trayce / "http-auth-token.txt").write_text("  secret-token  \n")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    port, token = _read_trayce_credentials()
    assert port == 52001
    assert token == "secret-token"


def test_read_credentials_missing_files_returns_none(monkeypatch, tmp_path):
    (tmp_path / "Trayce").mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert _read_trayce_credentials() == (None, None)


def test_read_credentials_garbage_port_is_none(monkeypatch, tmp_path):
    trayce = tmp_path / "Trayce"
    trayce.mkdir()
    (trayce / "port.txt").write_text("not-a-number")
    (trayce / "http-auth-token.txt").write_text("tok")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    port, token = _read_trayce_credentials()
    assert port is None
    assert token == "tok"


# --- url port override -------------------------------------------------


def test_override_url_port_replaces_existing_port():
    assert _override_url_port("http://127.0.0.1:52945/mcp", 9999) == (
        "http://127.0.0.1:9999/mcp"
    )


def test_override_url_port_adds_missing_port():
    assert _override_url_port("http://localhost/mcp", 8080) == (
        "http://localhost:8080/mcp"
    )


def test_override_url_port_no_path():
    assert _override_url_port("http://127.0.0.1:52945", 1) == "http://127.0.0.1:1"


# --- endpoint resolution -----------------------------------------------


def test_resolve_endpoint_token_from_file(monkeypatch, tmp_path):
    trayce = tmp_path / "Trayce"
    trayce.mkdir()
    (trayce / "port.txt").write_text("60000")
    (trayce / "http-auth-token.txt").write_text("file-token")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    cfg = MCPServerConfig(name="trayce", url="http://127.0.0.1:52945/mcp")
    url, token = resolve_trayce_endpoint(cfg)
    assert url == "http://127.0.0.1:60000/mcp"  # port overridden
    assert token == "file-token"


def test_resolve_endpoint_literal_token_when_not_from_file():
    cfg = MCPServerConfig(
        name="custom",
        url="http://example.local/mcp",
        auth_token_from_file=False,
        auth_token="literal",
    )
    url, token = resolve_trayce_endpoint(cfg)
    assert url == "http://example.local/mcp"
    assert token == "literal"


def test_resolve_endpoint_missing_files_keeps_config_url(monkeypatch, tmp_path):
    (tmp_path / "Trayce").mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    cfg = MCPServerConfig(name="trayce", url="http://127.0.0.1:52945/mcp")
    url, token = resolve_trayce_endpoint(cfg)
    assert url == "http://127.0.0.1:52945/mcp"
    assert token is None


# --- tool name sanitisation --------------------------------------------


def test_sanitize_tool_name_prefixes_and_cleans():
    assert _sanitize_tool_name("trayce", "search_context") == "trayce_search_context"


def test_sanitize_tool_name_replaces_illegal_chars():
    assert _sanitize_tool_name("trayce", "fs.read file") == "trayce_fs_read_file"


def test_sanitize_tool_name_raises_on_empty():
    with pytest.raises(ValueError):
        _sanitize_tool_name("", "!!!")


# --- dynamic args schema -----------------------------------------------


def test_model_from_input_schema_echoes_server_schema():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    model = _model_from_input_schema("Foo_Args", schema)
    assert model.model_json_schema() == schema


def test_model_from_input_schema_accepts_arbitrary_args():
    model = _model_from_input_schema("Foo_Args", {"type": "object"})
    instance = model.model_validate({"query": "hello", "limit": 5})
    dumped = instance.model_dump(exclude_none=True)
    assert dumped == {"query": "hello", "limit": 5}


def test_model_from_input_schema_handles_none():
    model = _model_from_input_schema("Foo_Args", None)
    assert model.model_json_schema() == {"type": "object", "properties": {}}


# --- mcp SDK fakes -----------------------------------------------------


class _FakeTool:
    def __init__(self, name, description="", input_schema=None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object"}


class _FakeListToolsResult:
    def __init__(self, tools):
        self.tools = tools


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeCallResult:
    def __init__(self, texts, is_error=False):
        self.content = [_FakeTextBlock(t) for t in texts]
        self.isError = is_error


class _FakeSession:
    def __init__(self, tools, call_result=None, call_raises=None):
        self._tools = tools
        self._call_result = call_result
        self._call_raises = call_raises
        self.initialized = False
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return _FakeListToolsResult(self._tools)

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        if self._call_raises is not None:
            raise self._call_raises
        return self._call_result


def _install_fake_mcp(monkeypatch, *, session: _FakeSession):
    """Install fake `mcp` and `mcp.client.streamable_http` modules so the
    lazy imports inside MCPServerConnection.connect() resolve to fakes."""

    @asynccontextmanager
    async def _fake_streamable(url, headers=None):
        # yields (read, write, get_session_id)
        yield (MagicMock(name="read"), MagicMock(name="write"), lambda: "sid")

    def _fake_client_session(read, write):
        return session

    mcp_mod = types.ModuleType("mcp")
    mcp_mod.ClientSession = _fake_client_session  # type: ignore[attr-defined]
    client_mod = types.ModuleType("mcp.client")
    streamable_mod = types.ModuleType("mcp.client.streamable_http")
    streamable_mod.streamablehttp_client = _fake_streamable  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", streamable_mod)
    return session


# --- connection flow ---------------------------------------------------


@pytest.mark.asyncio
async def test_connect_returns_tool_specs(monkeypatch):
    session = _FakeSession(tools=[
        _FakeTool("search_context", "Search.", {"type": "object"}),
        _FakeTool("add_fact", "Add a fact."),
    ])
    _install_fake_mcp(monkeypatch, session=session)
    conn = MCPServerConnection("trayce", "http://127.0.0.1:52945/mcp", "tok")
    specs = await conn.connect()
    assert session.initialized
    assert [s.name for s in specs] == ["search_context", "add_fact"]
    assert conn.connected is True


@pytest.mark.asyncio
async def test_connect_unavailable_when_sdk_missing(monkeypatch):
    # Ensure the import fails: remove mcp from sys.modules and block import.
    monkeypatch.setitem(sys.modules, "mcp", None)
    conn = MCPServerConnection("trayce", "http://x/mcp", None)
    from jarvis.tools.mcp_client import MCPUnavailable
    with pytest.raises(MCPUnavailable):
        await conn.connect()


@pytest.mark.asyncio
async def test_call_tool_translates_text_result(monkeypatch):
    session = _FakeSession(
        tools=[_FakeTool("search_context")],
        call_result=_FakeCallResult(["line one", "line two"]),
    )
    _install_fake_mcp(monkeypatch, session=session)
    conn = MCPServerConnection("trayce", "http://x/mcp", None)
    await conn.connect()
    result = await conn.call_tool("search_context", {"query": "x"})
    assert result.success is True
    assert result.output == "line one\nline two"
    assert session.calls == [("search_context", {"query": "x"})]


@pytest.mark.asyncio
async def test_call_tool_error_result(monkeypatch):
    session = _FakeSession(
        tools=[_FakeTool("t")],
        call_result=_FakeCallResult(["boom"], is_error=True),
    )
    _install_fake_mcp(monkeypatch, session=session)
    conn = MCPServerConnection("trayce", "http://x/mcp", None)
    await conn.connect()
    result = await conn.call_tool("t", {})
    assert result.success is False
    assert result.error == "boom"


@pytest.mark.asyncio
async def test_call_tool_disconnect_on_exception(monkeypatch):
    session = _FakeSession(
        tools=[_FakeTool("t")],
        call_raises=ConnectionError("dropped"),
    )
    _install_fake_mcp(monkeypatch, session=session)
    conn = MCPServerConnection("trayce", "http://x/mcp", None)
    await conn.connect()
    result = await conn.call_tool("t", {})
    assert result.success is False
    assert "disconnected" in (result.error or "")
    assert conn.connected is False


@pytest.mark.asyncio
async def test_call_tool_when_not_connected():
    conn = MCPServerConnection("trayce", "http://x/mcp", None)
    result = await conn.call_tool("t", {})
    assert result.success is False
    assert "disconnected" in (result.error or "")


# --- manager: registration ---------------------------------------------


def _registry() -> ToolRegistry:
    return ToolRegistry(ToolsConfig())


@pytest.mark.asyncio
async def test_add_server_registers_prefixed_tools(monkeypatch):
    session = _FakeSession(tools=[
        _FakeTool("search_context", "Search."),
        _FakeTool("add_fact", "Add."),
        _FakeTool("get_summary", "Summary."),
    ])
    _install_fake_mcp(monkeypatch, session=session)
    reg = _registry()
    mgr = MCPManager(reg)
    await mgr.add_server(MCPServerConfig(name="trayce", url="http://x/mcp"))

    assert reg.get("trayce_search_context") is not None
    assert reg.get("trayce_add_fact") is not None
    assert reg.get("trayce_get_summary") is not None
    assert set(mgr.registered_tools("trayce")) == {
        "trayce_search_context", "trayce_add_fact", "trayce_get_summary",
    }


@pytest.mark.asyncio
async def test_add_server_skips_when_disabled(monkeypatch):
    session = _FakeSession(tools=[_FakeTool("t")])
    _install_fake_mcp(monkeypatch, session=session)
    reg = _registry()
    mgr = MCPManager(reg)
    await mgr.add_server(MCPServerConfig(name="trayce", url="http://x/mcp", enabled=False))
    assert mgr.server_names == []


@pytest.mark.asyncio
async def test_add_server_handles_connect_failure(monkeypatch):
    # streamable client raises on enter -> add_server logs + skips.
    @asynccontextmanager
    async def _boom(url, headers=None):
        raise ConnectionError("trayce down")
        yield  # pragma: no cover

    mcp_mod = types.ModuleType("mcp")
    mcp_mod.ClientSession = lambda r, w: None  # type: ignore[attr-defined]
    streamable_mod = types.ModuleType("mcp.client.streamable_http")
    streamable_mod.streamablehttp_client = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", streamable_mod)

    reg = _registry()
    mgr = MCPManager(reg)
    await mgr.add_server(MCPServerConfig(name="trayce", url="http://x/mcp"))
    assert mgr.server_names == []  # nothing registered, no crash


@pytest.mark.asyncio
async def test_mcp_tool_execute_calls_connection(monkeypatch):
    session = _FakeSession(
        tools=[_FakeTool("search_context", "Search.", {"type": "object"})],
        call_result=_FakeCallResult(["found it"]),
    )
    _install_fake_mcp(monkeypatch, session=session)
    reg = _registry()
    mgr = MCPManager(reg)
    await mgr.add_server(MCPServerConfig(name="trayce", url="http://x/mcp"))

    tool = reg.get("trayce_search_context")
    assert isinstance(tool, MCPTool)
    args = tool.args_schema.model_validate({"query": "yesterday"})
    result = await tool.execute(args)
    assert result.success is True
    assert result.output == "found it"
    # The remote (unprefixed) name is what's sent to the server.
    assert session.calls == [("search_context", {"query": "yesterday"})]


@pytest.mark.asyncio
async def test_add_server_drops_colliding_tool(monkeypatch):
    # Pre-register a local tool that will collide with the prefixed name.
    reg = _registry()

    class _Local:
        name = "trayce_search_context"
        description = "local"
        from jarvis.tools.registry import EmptyArgs as _E
        args_schema = _E
        requires_confirmation = False

        async def execute(self, args):  # pragma: no cover
            return ToolResult(success=True)

    reg.register(_Local())

    session = _FakeSession(tools=[
        _FakeTool("search_context"),
        _FakeTool("add_fact"),
    ])
    _install_fake_mcp(monkeypatch, session=session)
    mgr = MCPManager(reg)
    await mgr.add_server(MCPServerConfig(name="trayce", url="http://x/mcp"))

    # Colliding tool dropped; the other still registers.
    assert mgr.registered_tools("trayce") == ["trayce_add_fact"]
    # The local tool is untouched.
    assert type(reg.get("trayce_search_context")).__name__ == "_Local"


# --- manager: removal + reload -----------------------------------------


@pytest.mark.asyncio
async def test_remove_server_unregisters_tools(monkeypatch):
    session = _FakeSession(tools=[_FakeTool("a"), _FakeTool("b")])
    _install_fake_mcp(monkeypatch, session=session)
    reg = _registry()
    mgr = MCPManager(reg)
    await mgr.add_server(MCPServerConfig(name="trayce", url="http://x/mcp"))
    assert reg.get("trayce_a") is not None

    await mgr.remove_server("trayce")
    assert reg.get("trayce_a") is None
    assert reg.get("trayce_b") is None
    assert mgr.server_names == []


@pytest.mark.asyncio
async def test_reload_adds_and_removes(monkeypatch):
    session = _FakeSession(tools=[_FakeTool("a")])
    _install_fake_mcp(monkeypatch, session=session)
    reg = _registry()
    mgr = MCPManager(reg)

    # Start with trayce enabled.
    await mgr.reload_from_config([
        MCPServerConfig(name="trayce", url="http://x/mcp", enabled=True),
    ])
    assert "trayce" in mgr.server_names
    assert reg.get("trayce_a") is not None

    # Reload with trayce now disabled -> removed.
    await mgr.reload_from_config([
        MCPServerConfig(name="trayce", url="http://x/mcp", enabled=False),
    ])
    assert mgr.server_names == []
    assert reg.get("trayce_a") is None


@pytest.mark.asyncio
async def test_reload_reconnects_on_url_change(monkeypatch):
    session = _FakeSession(tools=[_FakeTool("a")])
    _install_fake_mcp(monkeypatch, session=session)
    reg = _registry()
    mgr = MCPManager(reg)
    await mgr.reload_from_config([
        MCPServerConfig(name="srv", url="http://a/mcp", enabled=True,
                        auth_token_from_file=False),
    ])
    conn1 = mgr._connections["srv"]
    await mgr.reload_from_config([
        MCPServerConfig(name="srv", url="http://b/mcp", enabled=True,
                        auth_token_from_file=False),
    ])
    conn2 = mgr._connections["srv"]
    assert conn1 is not conn2  # reconnected
    assert conn2.url == "http://b/mcp"


@pytest.mark.asyncio
async def test_shutdown_disconnects_all(monkeypatch):
    session = _FakeSession(tools=[_FakeTool("a")])
    _install_fake_mcp(monkeypatch, session=session)
    reg = _registry()
    mgr = MCPManager(reg)
    await mgr.add_server(MCPServerConfig(name="trayce", url="http://x/mcp"))
    await mgr.shutdown()
    assert mgr.server_names == []
    assert reg.get("trayce_a") is None
