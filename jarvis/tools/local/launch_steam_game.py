"""Launch an installed Steam library game by title."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from jarvis.platform import windows as winplat
from jarvis.tools.registry import ToolResult

if TYPE_CHECKING:
    from jarvis.platform.windows_apps import SteamGame


class LaunchSteamGameArgs(BaseModel):
    game_name: str = Field(
        description=(
            "Title of the Steam game to launch, as the user said it "
            "(e.g. 'Elden Ring', 'Portal 2')."
        ),
    )


class LaunchSteamGameTool:
    name: str = "launch_steam_game"
    description: str = (
        "Launch a video game from the user's installed Steam library. "
        "Use ONLY when the user clearly wants to play, start, run, or "
        "boot a game on Steam — e.g. 'play Elden Ring', 'launch Cyberpunk', "
        "'start Portal on Steam'. Do NOT use for music, songs, artists, "
        "YouTube, or opening the Steam client itself (use open_app for "
        "the Steam application)."
    )
    args_schema = LaunchSteamGameArgs
    requires_confirmation: bool = False

    def __init__(
        self,
        *,
        steam_resolver: Callable[[str], SteamGame | None] | None = None,
    ) -> None:
        self._steam_resolver = steam_resolver

    def _resolve(self, query: str) -> SteamGame | None:
        if self._steam_resolver is not None:
            return self._steam_resolver(query)
        from jarvis.platform.windows_apps import resolve_steam_game

        return resolve_steam_game(query)

    async def execute(self, args: LaunchSteamGameArgs) -> ToolResult:
        name = args.game_name.strip()
        if not name:
            return ToolResult(success=False, error="game name is empty")

        game = await asyncio.to_thread(self._resolve, name)
        if game is None:
            return ToolResult(
                success=False,
                error=(
                    f"no installed Steam game matched {name!r}. "
                    "Check the title or install the game in Steam."
                ),
            )
        try:
            await asyncio.to_thread(winplat.launch_steam_game, game.app_id)
        except NotImplementedError as e:
            return ToolResult(success=False, error=str(e))
        except OSError as e:
            return ToolResult(
                success=False,
                error=f"could not launch {game.name!r}: {e}",
            )
        return ToolResult(
            success=True,
            output=f"Launching {game.name} on Steam, sir.",
        )
