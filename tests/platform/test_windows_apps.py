"""Tests for jarvis.platform.windows_apps (offline fixtures)."""

from __future__ import annotations

from pathlib import Path

from jarvis.platform.app_discovery import normalize_query
from jarvis.platform.windows_apps import (
    SteamGame,
    _parse_appmanifest,
    resolve_steam_game,
    scan_steam_games,
)


def test_parse_appmanifest_extracts_name_and_id(tmp_path: Path):
    manifest = tmp_path / "appmanifest_123.acf"
    manifest.write_text(
        '"AppState"\n{\n\t"appid"\t\t"123"\n\t"name"\t\t"Elden Ring"\n}\n',
        encoding="utf-8",
    )
    game = _parse_appmanifest(manifest)
    assert game == SteamGame(name="Elden Ring", app_id=123)


def test_scan_steam_games_from_fixture_tree(tmp_path: Path):
    steamapps = tmp_path / "steamapps"
    steamapps.mkdir()
    (steamapps / "appmanifest_10.acf").write_text(
        '"appid"\t\t"10"\n"name"\t\t"Half-Life"\n',
        encoding="utf-8",
    )
    (steamapps / "appmanifest_20.acf").write_text(
        '"appid"\t\t"20"\n"name"\t\t"Portal"\n',
        encoding="utf-8",
    )
    games = scan_steam_games(tmp_path)
    assert normalize_query("Half-Life") in games
    assert games[normalize_query("Half-Life")].app_id == 10


def test_resolve_steam_game_fuzzy():
    games = {
        normalize_query("Elden Ring"): SteamGame(name="Elden Ring", app_id=1245620),
    }
    hit = resolve_steam_game("elden ring", games=games)
    assert hit is not None
    assert hit.app_id == 1245620
