"""Tool protocol, registry, and tool implementations.

Public surface:
  - Tool, ToolResult, EmptyArgs:       protocol + return type
  - ToolRegistry, ToolNameCollision:   registry + collision exception
  - TOOL_NAME_REGEX:                   shared name-validity regex

Local tool implementations live under jarvis/tools/local/. MCP-adapted
tools live under jarvis/tools/mcp_client.py (Phase 4 Task 3).
"""

from jarvis.tools.registry import (
    TOOL_NAME_REGEX,
    EmptyArgs,
    Tool,
    ToolNameCollision,
    ToolRegistry,
    ToolResult,
)
from jarvis.tools.mcp_client import MCPManager, MCPServerConnection, MCPTool


def setup_local_tools(registry: ToolRegistry, config: object = None) -> None:
    """Register every built-in local tool into `registry`.

    Built-ins register before any MCP-adapted tools (see registry.py
    header on the name-collision policy): an MCP server publishing a
    tool whose name clashes with a built-in is refused, and the
    built-in stays. Add new local tools here in the same lexical order
    as their filenames so reviewers can audit the set against
    jarvis/tools/local/ at a glance.

    `config` is the live JarvisConfig instance. Tools that need config
    (e.g. WeatherTool) receive their sub-config section at construction."""
    from jarvis.tools.local.clipboard import ClipboardTool
    from jarvis.tools.local.close_app import CloseAppTool
    from jarvis.tools.local.files import ListDirectoryTool
    from jarvis.tools.local.lock_screen import LockScreenTool
    from jarvis.tools.local.launch_steam_game import LaunchSteamGameTool
    from jarvis.tools.local.launch_workspace import LaunchWorkspaceTool
    from jarvis.tools.local.open_app import OpenAppTool
    from jarvis.tools.local.open_url import OpenUrlTool
    from jarvis.tools.local.play_youtube_music import PlayYoutubeMusicTool
    from jarvis.tools.local.screenshot import ScreenshotTool
    from jarvis.tools.local.system_stats import SystemStatsTool
    from jarvis.tools.local.type_into_active_window import TypeIntoActiveWindowTool
    from jarvis.tools.local.volume import VolumeTool
    from jarvis.tools.local.weather import WeatherTool

    weather_cfg = getattr(config, "weather", None)
    if weather_cfg is None:
        from jarvis.core.config import WeatherConfig
        weather_cfg = WeatherConfig()

    weather_save_fn = None
    if config is not None:
        from jarvis.core.config import save_config as _save_cfg
        weather_save_fn = lambda: _save_cfg(config)  # type: ignore[arg-type]

    from jarvis.core.config import WorkspaceConfig

    ws_cfg = getattr(config, "workspace", None) if config is not None else None
    workspace_apps = list(
        (ws_cfg or WorkspaceConfig()).apps
    )

    for tool in (
        ClipboardTool(),
        CloseAppTool(),
        ListDirectoryTool(),
        LockScreenTool(),
        OpenAppTool(),
        OpenUrlTool(),
        LaunchSteamGameTool(),
        LaunchWorkspaceTool(workspace_apps=workspace_apps),
        PlayYoutubeMusicTool(),
        ScreenshotTool(),
        SystemStatsTool(),
        TypeIntoActiveWindowTool(),
        VolumeTool(),
        WeatherTool(weather_config=weather_cfg, save_fn=weather_save_fn),
    ):
        registry.register(tool)


__all__ = [
    "TOOL_NAME_REGEX",
    "EmptyArgs",
    "MCPManager",
    "MCPServerConnection",
    "MCPTool",
    "Tool",
    "ToolNameCollision",
    "ToolRegistry",
    "ToolResult",
    "setup_local_tools",
]
