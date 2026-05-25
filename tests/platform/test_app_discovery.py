"""Tests for jarvis.platform.app_discovery."""

from __future__ import annotations

from jarvis.platform.app_discovery import (
    fuzzy_resolve,
    normalize_open_query,
    normalize_query,
)


def test_normalize_open_query_strips_command_words():
    assert normalize_open_query("hey jarvis please open Google Chrome") == "Google Chrome"


def test_normalize_query_strips_fillers_and_punctuation():
    assert normalize_query("My  Photoshop!") == "photoshop"


def test_fuzzy_resolve_exact_key():
    choices = {"visual studio code": "code"}
    assert fuzzy_resolve("Visual Studio Code", choices) == (
        "visual studio code",
        "code",
    )


def test_fuzzy_resolve_close_typo():
    choices = {"adobe photoshop": "/path/photoshop.lnk"}
    hit = fuzzy_resolve("photoshop", choices, cutoff=0.5)
    assert hit is not None
    assert hit[1] == "/path/photoshop.lnk"


def test_fuzzy_resolve_miss_returns_none():
    assert fuzzy_resolve("zzzzunknown", {"foo bar": 1}) is None


def test_fuzzy_resolve_single_word_matches_multiword_via_token_overlap():
    """The common real failure: a one-word spoken name against the full
    Start-Menu display name. Token overlap handles it."""
    choices = {
        "microsoft word": "word.lnk",
        "microsoft excel": "excel.lnk",
        "visual studio code": "code.lnk",
    }
    assert fuzzy_resolve("word", choices) == ("microsoft word", "word.lnk")
    assert fuzzy_resolve("excel", choices) == ("microsoft excel", "excel.lnk")
    assert fuzzy_resolve("code", choices) == ("visual studio code", "code.lnk")


def test_fuzzy_resolve_substring_prominence():
    """A prominent substring resolves even when difflib's whole-string
    ratio is low ('teams' inside 'microsoft teams classic')."""
    choices = {"microsoft teams classic": "teams.lnk"}
    hit = fuzzy_resolve("teams", choices)
    assert hit is not None
    assert hit[1] == "teams.lnk"


def test_fuzzy_resolve_prefix():
    choices = {"spotify": "spotify.lnk"}
    hit = fuzzy_resolve("spot", choices)
    assert hit is not None
    assert hit[1] == "spotify.lnk"


def test_fuzzy_resolve_typo_via_difflib():
    choices = {"photoshop": "ps.lnk"}
    hit = fuzzy_resolve("photoshpo", choices)
    assert hit is not None
    assert hit[1] == "ps.lnk"


def test_fuzzy_resolve_picks_best_of_several():
    """When several keys partially match, the highest-scoring wins."""
    choices = {
        "discord": "discord.lnk",
        "discord ptb": "ptb.lnk",
        "discord canary": "canary.lnk",
    }
    # Exact-ish single token should prefer the bare "discord".
    assert fuzzy_resolve("discord", choices) == ("discord", "discord.lnk")


def test_fuzzy_resolve_short_query_does_not_overmatch():
    """A 2-char query must not substring-match into unrelated long keys."""
    choices = {"calculator": "calc.lnk"}
    assert fuzzy_resolve("go", choices) is None
