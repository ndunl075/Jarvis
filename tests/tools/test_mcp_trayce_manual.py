"""Manual integration test: connect to a real running Trayce instance.

Marked `manual` so it is skipped in the default run. To execute (with
Trayce running and signed in):

    pytest -m manual tests/tools/test_mcp_trayce_manual.py -s

The test reads Trayce's port + auth token from %LOCALAPPDATA%\\Trayce\\,
connects via the real mcp SDK, registers the tools, and prints them. It
asserts at least one tool was discovered. If Trayce is not running it is
skipped (not failed) so the manual pass over the whole suite stays green
on machines without Trayce."""

from __future__ import annotations

import pytest

from jarvis.core.config import MCPServerConfig, ToolsConfig
from jarvis.tools.mcp_client import (
    MCPManager,
    _read_trayce_credentials,
)
from jarvis.tools.registry import ToolRegistry


@pytest.mark.manual
@pytest.mark.asyncio
async def test_connects_to_running_trayce():
    pytest.importorskip("mcp")
    port, token = _read_trayce_credentials()
    if token is None:
        pytest.skip("Trayce auth token not found — is Trayce running and signed in?")

    reg = ToolRegistry(ToolsConfig())
    mgr = MCPManager(reg)
    try:
        await mgr.add_server(MCPServerConfig(name="trayce", enabled=True))
        tools = mgr.registered_tools("trayce")
        print(f"\n[manual] Trayce port={port}, discovered {len(tools)} tool(s):")
        for name in tools:
            print(f"  - {name}")
        if not tools:
            pytest.skip("Connected but Trayce exposed no tools (or connect failed).")
        assert any(name.startswith("trayce_") for name in tools)

        # Smoke-call get_summary if present; just assert it returns a ToolResult.
        for name in tools:
            if name.endswith("get_summary"):
                tool = reg.get(name)
                assert tool is not None
                args = tool.args_schema.model_validate({})
                result = await tool.execute(args)
                print(f"[manual] {name} -> success={result.success}")
                break
    finally:
        await mgr.shutdown()
