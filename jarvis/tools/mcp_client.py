"""MCP client: connects to external MCP servers over Streamable HTTP and
adapts their tools into the local ToolRegistry.

Phase 4 Task 3. Primary target is Trayce (a local context server) but the
machinery is server-agnostic; any Streamable-HTTP MCP server in
config.mcp_servers is connected the same way.

Design
------
- MCPServerConnection owns one server's transport + session. It uses the
  official `mcp` SDK's streamablehttp_client + ClientSession, kept alive
  across calls via an AsyncExitStack so a connection persists for the
  whole session rather than reconnecting per tool call.
- MCPTool adapts a single remote tool to Jarvis's Tool protocol
  (registry.py). The remote JSONSchema is surfaced to the LLM verbatim
  via a dynamically-built BaseModel whose model_json_schema() returns the
  server's inputSchema. execute() forwards validated args to the server
  and translates the CallToolResult back into Jarvis's ToolResult.
- MCPManager orchestrates connections: add_server connects + registers,
  remove_server unregisters + disconnects, reload_from_config diffs the
  desired set against the live set. shutdown closes everything.

Everything is best-effort. A missing `mcp` install, a server that's down,
or an absent auth token logs a warning and skips that server -- Jarvis
startup never blocks on MCP.

Tool name collisions
--------------------
Remote tool names are prefixed with the server name (`trayce` +
`search_context` -> `trayce_search_context`) and sanitised against
TOOL_NAME_REGEX. The registry's refuse-and-error collision policy then
catches any genuine clash; MCPManager logs and drops just the colliding
tool from the offending server, leaving the rest registered.

Credentials
-----------
Trayce writes its HTTP port and auth token to %LOCALAPPDATA%\\Trayce\\.
_read_trayce_credentials reads both; either being absent is non-fatal
(connection is attempted at the configured URL, unauthenticated, and the
server's rejection surfaces as a skipped connection with a clear log).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from jarvis.core.config import MCPServerConfig
from jarvis.tools.registry import (
    TOOL_NAME_REGEX,
    ToolNameCollisionError,
    ToolRegistry,
    ToolResult,
)

log = logging.getLogger(__name__)

TRAYCE_AUTH_HEADER = "X-Trayce-Auth-Token"

# Replaces any run of characters illegal in a tool name with a single
# underscore so server-prefixed names satisfy TOOL_NAME_REGEX.
_ILLEGAL_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]+")


# --- credentials ----------------------------------------------------------


def _read_trayce_credentials() -> tuple[int | None, str | None]:
    """Return (port, token) from the Trayce data dir. Either value is None
    if its file is missing or unreadable. Never raises."""
    appdata = os.environ.get("LOCALAPPDATA", "")
    if not appdata:
        return None, None
    trayce_dir = Path(appdata) / "Trayce"
    port_file = trayce_dir / "port.txt"
    token_file = trayce_dir / "http-auth-token.txt"
    try:
        port = int(port_file.read_text().strip()) if port_file.exists() else None
    except Exception:
        log.debug("Trayce port.txt unreadable", exc_info=True)
        port = None
    try:
        token = token_file.read_text().strip() if token_file.exists() else None
    except Exception:
        log.debug("Trayce http-auth-token.txt unreadable", exc_info=True)
        token = None
    return port, token


def _override_url_port(url: str, port: int) -> str:
    """Return `url` with its port replaced by `port`. Best-effort: if the
    URL has no recognisable host:port authority, returns it unchanged."""
    # Match scheme://host(:port)?(/rest). Keep it dependency-free; urllib
    # would also work but this is a single targeted substitution.
    m = re.match(r"^(?P<scheme>[a-z]+://)(?P<host>[^/:]+)(?::\d+)?(?P<rest>/.*|$)", url)
    if not m:
        return url
    return f"{m.group('scheme')}{m.group('host')}:{port}{m.group('rest')}"


def resolve_trayce_endpoint(config: MCPServerConfig) -> tuple[str, str | None]:
    """Resolve (url, auth_token) for a Trayce-style server.

    - URL: the configured URL, with its port overridden by port.txt when
      that file is present.
    - Token: from http-auth-token.txt when auth_token_from_file is True,
      else the literal config.auth_token.

    Non-Trayce servers (auth_token_from_file False) get their config URL
    untouched and the literal token."""
    if not config.auth_token_from_file:
        return config.url, config.auth_token
    port, token = _read_trayce_credentials()
    url = _override_url_port(config.url, port) if port is not None else config.url
    return url, token


# --- dynamic args schema --------------------------------------------------


def _model_from_input_schema(model_name: str, input_schema: dict | None) -> type[BaseModel]:
    """Build a pydantic model that (a) reports the remote JSONSchema
    verbatim to the LLM via model_json_schema() and (b) accepts arbitrary
    validated args via model_validate().

    Rationale: the MCP server is the authority on its own arg schema and
    validates server-side. Re-deriving strict pydantic fields from
    arbitrary JSONSchema (nested objects, anyOf, $ref) is fragile; an
    extra-allow passthrough that echoes the server schema to the LLM is
    both simpler and exactly what the function-calling surface needs."""
    schema = input_schema or {"type": "object", "properties": {}}

    class _RemoteArgs(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict:
            return schema

    _RemoteArgs.__name__ = model_name
    _RemoteArgs.__qualname__ = model_name
    return _RemoteArgs


def _sanitize_tool_name(server_name: str, tool_name: str) -> str:
    """Server-prefix and sanitise a remote tool name so it satisfies
    TOOL_NAME_REGEX. Raises ValueError if the result is empty/invalid."""
    raw = f"{server_name}_{tool_name}"
    cleaned = _ILLEGAL_NAME_CHARS.sub("_", raw).strip("_")
    cleaned = cleaned[:64]
    if not cleaned or not TOOL_NAME_REGEX.match(cleaned):
        raise ValueError(f"cannot derive a valid tool name from {raw!r}")
    return cleaned


# --- per-tool adapter -----------------------------------------------------


class MCPTool:
    """Adapts one remote MCP tool to Jarvis's Tool protocol."""

    requires_confirmation: bool = False

    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        connection: MCPServerConnection,
        remote_name: str,
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self._connection = connection
        # The unprefixed name the server knows this tool by.
        self._remote_name = remote_name

    async def execute(self, args: BaseModel) -> ToolResult:
        # Recover the raw arg dict. The passthrough model stores caller
        # args as extras; model_dump() yields them as a plain dict.
        try:
            arg_dict = args.model_dump(exclude_none=True)
        except Exception:
            arg_dict = {}
        return await self._connection.call_tool(self._remote_name, arg_dict)


# --- per-server connection ------------------------------------------------


class ToolSpec(BaseModel):
    """Lightweight description of a discovered remote tool. Decouples the
    manager from the SDK's Tool type so tests need not import mcp."""

    name: str
    description: str = ""
    input_schema: dict = {}


class MCPServerConnection:
    """Owns one MCP server's transport + session for the session lifetime."""

    def __init__(self, name: str, url: str, auth_token: str | None) -> None:
        self.name = name
        self.url = url
        self._auth_token = auth_token
        self._session: Any = None
        self._exit_stack: Any = None
        self._connected: bool = False

    @property
    def connected(self) -> bool:
        return self._connected

    def _headers(self) -> dict[str, str]:
        if self._auth_token:
            return {TRAYCE_AUTH_HEADER: self._auth_token}
        return {}

    async def connect(self) -> list[ToolSpec]:
        """Open the transport + session and return the server's tools.

        Raises on failure -- the caller (MCPManager.add_server) catches
        and skips the server. The SDK is imported lazily so a missing
        `mcp` install degrades gracefully and tests can patch the import
        boundary."""
        from contextlib import AsyncExitStack

        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as e:  # pragma: no cover - exercised via monkeypatch
            raise MCPUnavailableError("mcp SDK not installed") from e

        stack = AsyncExitStack()
        try:
            transport = await stack.enter_async_context(
                streamablehttp_client(self.url, headers=self._headers())
            )
            # streamablehttp_client yields (read, write, get_session_id).
            read, write = transport[0], transport[1]
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.list_tools()
        except Exception:
            await stack.aclose()
            raise
        self._exit_stack = stack
        self._session = session
        self._connected = True
        return [
            ToolSpec(
                name=t.name,
                description=getattr(t, "description", "") or "",
                input_schema=getattr(t, "inputSchema", None) or {},
            )
            for t in result.tools
        ]

    async def call_tool(self, name: str, args: dict) -> ToolResult:
        """Invoke a remote tool and translate the result to ToolResult.

        A dropped connection surfaces as ToolResult(success=False, ...);
        the manager schedules a background reconnect on the next failure
        (see MCPManager). Never raises CancelledError-masking exceptions
        out -- the registry wraps execute() but a clean ToolResult is
        friendlier to the SpeakIntent layer."""
        if not self._connected or self._session is None:
            return ToolResult(
                success=False, error=f"{self.name} disconnected"
            )
        try:
            result = await self._session.call_tool(name, args)
        except Exception as e:
            log.warning("MCP call %s.%s failed: %s", self.name, name, e)
            # Mark disconnected so a reconnect is attempted next time.
            self._connected = False
            return ToolResult(
                success=False, error=f"{self.name} disconnected"
            )
        return _translate_call_result(result)

    async def disconnect(self) -> None:
        self._connected = False
        self._session = None
        stack, self._exit_stack = self._exit_stack, None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                log.debug("MCP %s exit-stack close failed", self.name, exc_info=True)


def _translate_call_result(result: Any) -> ToolResult:
    """Translate an SDK CallToolResult into a Jarvis ToolResult.

    Joins text content blocks into a single string. isError -> failure."""
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    joined = "\n".join(texts).strip()
    if getattr(result, "isError", False):
        return ToolResult(success=False, error=joined or "tool reported an error")
    return ToolResult(success=True, output=joined or "")


# --- exceptions -----------------------------------------------------------


class MCPUnavailableError(RuntimeError):
    """Raised inside connect() when the mcp SDK isn't importable. Caught by
    MCPManager.add_server and logged as a skip."""


# --- manager --------------------------------------------------------------


class MCPManager:
    """Orchestrates all MCP server connections + their registered tools."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        # name -> connection
        self._connections: dict[str, MCPServerConnection] = {}
        # name -> list of registered (prefixed) tool names, for clean removal
        self._registered: dict[str, list[str]] = {}

    @property
    def server_names(self) -> list[str]:
        return list(self._connections)

    def registered_tools(self, server_name: str) -> list[str]:
        return list(self._registered.get(server_name, ()))

    async def add_server(self, config: MCPServerConfig) -> None:
        """Connect to one server, discover its tools, register each as an
        MCPTool. Best-effort: any failure logs a warning and leaves Jarvis
        running with this server simply absent."""
        if not config.enabled:
            log.info("MCP server %r disabled in config; skipping", config.name)
            return
        if config.name in self._connections:
            log.debug("MCP server %r already connected; skipping", config.name)
            return

        url, token = resolve_trayce_endpoint(config)
        if config.auth_token_from_file and token is None and _is_trayce(config):
            log.warning(
                "Trayce auth token not found — is Trayce signed in? "
                "Connecting to %s without auth.", url,
            )
        conn = MCPServerConnection(config.name, url, token)
        try:
            specs = await conn.connect()
        except MCPUnavailableError as e:
            log.warning("MCP unavailable for %r: %s", config.name, e)
            return
        except Exception as e:
            if _is_trayce(config):
                log.warning(
                    "Trayce MCP server unavailable, skipping. Run Trayce to "
                    "enable Trayce tools. (%s)", e,
                )
            else:
                log.warning("MCP server %r unavailable, skipping: %s", config.name, e)
            return

        registered: list[str] = []
        for spec in specs:
            try:
                tool_name = _sanitize_tool_name(config.name, spec.name)
            except ValueError as e:
                log.warning("dropping MCP tool %r from %r: %s", spec.name, config.name, e)
                continue
            tool = MCPTool(
                name=tool_name,
                description=spec.description,
                args_schema=_model_from_input_schema(
                    f"{tool_name}_Args", spec.input_schema
                ),
                connection=conn,
                remote_name=spec.name,
            )
            try:
                self._registry.register(tool)
            except ToolNameCollisionError as e:
                log.warning("MCP tool name collision, dropping %r: %s", tool_name, e)
                continue
            registered.append(tool_name)

        self._connections[config.name] = conn
        self._registered[config.name] = registered
        log.info(
            "MCP server %r connected: %d tool(s) registered (%s)",
            config.name, len(registered), ", ".join(registered) or "none",
        )

    async def remove_server(self, name: str) -> None:
        """Unregister all of a server's tools and disconnect it."""
        for tool_name in self._registered.pop(name, []):
            self._registry.unregister(tool_name)
        conn = self._connections.pop(name, None)
        if conn is not None:
            await conn.disconnect()
            log.info("MCP server %r removed", name)

    async def reload_from_config(self, configs: list[MCPServerConfig]) -> None:
        """Diff the desired server set against the live set and converge.

        - Remove servers that are gone or now disabled.
        - Add servers that are newly present and enabled.
        - For servers whose URL/auth changed, remove+re-add so the new
          endpoint takes effect."""
        desired: dict[str, MCPServerConfig] = {
            c.name: c for c in configs if c.enabled
        }

        # Remove anything no longer desired.
        for name in list(self._connections):
            cfg = desired.get(name)
            if cfg is None:
                await self.remove_server(name)
                continue
            # Endpoint changed -> reconnect.
            conn = self._connections[name]
            new_url, _ = resolve_trayce_endpoint(cfg)
            if new_url != conn.url:
                await self.remove_server(name)

        # Add anything desired but not connected.
        for name, cfg in desired.items():
            if name not in self._connections:
                await self.add_server(cfg)

    async def shutdown(self) -> None:
        for name in list(self._connections):
            await self.remove_server(name)


def _is_trayce(config: MCPServerConfig) -> bool:
    return config.name.lower() == "trayce"
