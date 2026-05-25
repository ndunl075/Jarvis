"""Tests for jarvis.tools.local.launch_steam_game."""

from __future__ import annotations

from unittest.mock import patch

from jarvis.platform.windows_apps import SteamGame
from jarvis.tools.local.launch_steam_game import LaunchSteamGameArgs, LaunchSteamGameTool


async def test_launch_steam_game_success():
    game = SteamGame(name="Portal", app_id=400)
    tool = LaunchSteamGameTool(steam_resolver=lambda q: game)
    with patch(
        "jarvis.tools.local.launch_steam_game.winplat.launch_steam_game"
    ) as launch:
        result = await tool.execute(LaunchSteamGameArgs(game_name="portal"))
    assert result.success
    launch.assert_called_once_with(400)


async def test_launch_steam_game_no_match():
    tool = LaunchSteamGameTool(steam_resolver=lambda q: None)
    result = await tool.execute(LaunchSteamGameArgs(game_name="unknown game"))
    assert not result.success
    assert "no installed Steam game" in (result.error or "")


async def test_launch_steam_game_empty_name():
    result = await LaunchSteamGameTool().execute(LaunchSteamGameArgs(game_name="  "))
    assert not result.success
