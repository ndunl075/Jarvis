"""Tests for jarvis.core.config: defaults, validation, round-trip, migration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.core import config as cfg_mod
from jarvis.core.config import (
    CURRENT_SCHEMA_VERSION,
    ConfigMigrationError,
    JarvisConfig,
    MCPServerConfig,
    _SYSTEM_PROMPT_V6,
    default_config_path,
    load_config,
    migrate,
    save_config,
)

# --- defaults --------------------------------------------------------------


def test_defaults_match_spec():
    c = JarvisConfig()
    assert c.schema_version == CURRENT_SCHEMA_VERSION
    assert c.general.log_level == "INFO"
    assert c.general.minimize_to_tray is True
    assert c.audio.input_device == "TONOR"
    assert c.audio.output_device == "Speakers (Creative Pebble Pro)"
    assert c.audio.prefer_respeaker is True
    assert c.wake_word.model == "hey_jarvis"
    assert c.wake_word.sensitivity == 0.5
    assert c.stt.model_size == "tiny.en"
    assert c.stt.language == "en"
    assert c.stt.compute_type == "int8"
    assert c.tts.voice == "en_GB-alan-medium"
    assert c.llm.model == "qwen2.5:7b-instruct"
    assert c.llm.keep_alive_seconds == 1800
    assert c.llm.max_turns == 10
    assert c.llm.inactivity_timeout_seconds == 300
    assert c.llm.conversation_continuity_seconds == 60
    assert "Jarvis" in c.llm.system_prompt
    assert "sir" in c.llm.system_prompt
    assert "count" in c.llm.system_prompt.lower()
    assert "Don't hallucinate" in c.llm.system_prompt
    assert c.debug.log_wake_during_speaking is False
    # Fresh installs ship no default location; the weather tool's
    # ipapi.co fallback fills it on first use.
    assert c.weather.latitude is None
    assert c.weather.longitude is None
    assert c.weather.unit == "fahrenheit"
    assert c.tools.enabled == {"type_into_active_window": False}
    # Fresh installs ship a single disabled Trayce MCP entry so the
    # Settings → Tools toggle is discoverable.
    assert len(c.mcp_servers) == 1
    trayce = c.mcp_servers[0]
    assert trayce.name == "trayce"
    assert trayce.url == "http://127.0.0.1:52945/mcp"
    assert trayce.enabled is False
    assert trayce.auth_token_from_file is True
    assert c.research.api_key is None
    assert c.ui.research_panel_width == 420


# --- validation ------------------------------------------------------------


def test_sensitivity_out_of_range_rejected():
    with pytest.raises(ValidationError):
        JarvisConfig(wake_word={"sensitivity": 1.5})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        JarvisConfig(wake_word={"sensitivity": -0.1})  # type: ignore[arg-type]


def test_log_level_literal_enforced():
    with pytest.raises(ValidationError):
        JarvisConfig(general={"log_level": "TRACE"})  # type: ignore[arg-type]


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        JarvisConfig(general={"unknown_field": True})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        JarvisConfig.model_validate(
            {"schema_version": 1, "junk": 1}
        )


def test_validate_assignment_catches_bad_writes():
    c = JarvisConfig()
    with pytest.raises(ValidationError):
        c.wake_word.sensitivity = 99.0
    # Original value still intact.
    assert c.wake_word.sensitivity == 0.5


def test_max_tokens_must_be_positive():
    with pytest.raises(ValidationError):
        JarvisConfig(llm={"max_tokens": 0})  # type: ignore[arg-type]


def test_mcp_server_defaults_to_trayce():
    """MCPServerConfig now defaults to a Trayce connection so a bare
    construction is the Trayce entry (the fresh-config list ships it
    disabled)."""
    s = MCPServerConfig()
    assert s.name == "trayce"
    assert s.url == "http://127.0.0.1:52945/mcp"
    assert s.enabled is True
    assert s.auth_token_from_file is True
    assert s.auth_token is None


# --- persistence: round-trip ----------------------------------------------


def test_round_trip_preserves_values(tmp_path: Path):
    p = tmp_path / "config.json"
    original = JarvisConfig()
    original.tts.voice = "en_US-amy-medium"
    original.llm.temperature = 0.42
    original.tools.enabled = {"screenshot": False, "open_app": True}
    original.mcp_servers = [
        MCPServerConfig(name="weather", url="http://localhost:9000")
    ]
    save_config(original, p)
    loaded = load_config(p)
    assert loaded.model_dump() == original.model_dump()


def test_load_creates_default_when_missing(tmp_path: Path):
    p = tmp_path / "nested" / "config.json"
    assert not p.exists()
    loaded = load_config(p)
    assert p.exists()
    assert loaded.model_dump() == JarvisConfig().model_dump()


def test_save_is_atomic_no_tmp_left_behind(tmp_path: Path):
    p = tmp_path / "config.json"
    save_config(JarvisConfig(), p)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_load_rejects_non_object_root(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigMigrationError):
        load_config(p)


def test_load_propagates_validation_error_for_corrupt_field(tmp_path: Path):
    p = tmp_path / "config.json"
    bad = JarvisConfig().model_dump(mode="json")
    bad["wake_word"]["sensitivity"] = 5.0
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(p)


# --- migration -------------------------------------------------------------


def test_migrate_passthrough_at_current_version():
    data = JarvisConfig().model_dump(mode="json")
    assert migrate(data) is data
    assert data["schema_version"] == CURRENT_SCHEMA_VERSION


def test_migrate_rejects_missing_schema_version():
    with pytest.raises(ConfigMigrationError):
        migrate({"general": {}})


def test_migrate_rejects_non_int_schema_version():
    with pytest.raises(ConfigMigrationError):
        migrate({"schema_version": "1"})


def test_migrate_rejects_future_version():
    with pytest.raises(ConfigMigrationError):
        migrate({"schema_version": CURRENT_SCHEMA_VERSION + 5})


def test_migrate_runs_registered_chain(monkeypatch: pytest.MonkeyPatch):
    # Pretend the current version is 3 with two migrations in the chain.
    calls: list[int] = []

    def v1_to_v2(d: dict) -> dict:
        calls.append(1)
        d["schema_version"] = 2
        d["added_in_v2"] = True
        return d

    def v2_to_v3(d: dict) -> dict:
        calls.append(2)
        d["schema_version"] = 3
        d["added_in_v3"] = True
        return d

    monkeypatch.setattr(cfg_mod, "CURRENT_SCHEMA_VERSION", 3)
    monkeypatch.setattr(cfg_mod, "MIGRATIONS", {1: v1_to_v2, 2: v2_to_v3})

    out = cfg_mod.migrate({"schema_version": 1})
    assert calls == [1, 2]
    assert out["schema_version"] == 3
    assert out["added_in_v2"] is True
    assert out["added_in_v3"] is True


def test_migrate_raises_when_link_in_chain_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cfg_mod, "CURRENT_SCHEMA_VERSION", 3)
    monkeypatch.setattr(cfg_mod, "MIGRATIONS", {})  # no migrations registered
    with pytest.raises(ConfigMigrationError, match="no migration registered"):
        cfg_mod.migrate({"schema_version": 1})


def test_migrate_raises_when_migration_fails_to_bump_version(
    monkeypatch: pytest.MonkeyPatch,
):
    def buggy(d: dict) -> dict:
        return d  # forgot to bump schema_version

    monkeypatch.setattr(cfg_mod, "CURRENT_SCHEMA_VERSION", 2)
    monkeypatch.setattr(cfg_mod, "MIGRATIONS", {1: buggy})
    with pytest.raises(ConfigMigrationError, match="produced schema_version"):
        cfg_mod.migrate({"schema_version": 1})


# --- default path ---------------------------------------------------------


def test_default_config_path_uses_appdata_on_windows(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("APPDATA", r"C:\Users\test\AppData\Roaming")
    p = default_config_path()
    assert p.name == "config.json"
    assert "Jarvis" in p.parts


def test_default_config_path_falls_back_off_windows(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.platform", "linux")
    p = default_config_path()
    assert p.name == "config.json"
    assert ".jarvis" in p.parts


# --- v1 → v2 device default migration ------------------------------------


def _v1_data(**audio_overrides: object) -> dict:
    """Build a minimal v1 config dict with optional audio field overrides."""
    base = JarvisConfig().model_dump(mode="json")
    base["schema_version"] = 1
    base["audio"].update(audio_overrides)
    return base


def test_migrate_v1_to_v2_fills_none_output_device():
    """v1→v2 sets 'Pebble Pro'; the subsequent v10→v11 migration upgrades
    it to the full 'Speakers (Creative Pebble Pro)' string."""
    data = _v1_data(output_device=None)
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert out["audio"]["output_device"] == "Speakers (Creative Pebble Pro)"


def test_migrate_v1_to_v2_fills_none_input_device():
    data = _v1_data(input_device=None)
    out = migrate(data)
    assert out["audio"]["input_device"] == "TONOR"


def test_migrate_v1_to_v2_does_not_overwrite_existing_output_device():
    data = _v1_data(output_device="Realtek Speakers")
    out = migrate(data)
    assert out["audio"]["output_device"] == "Realtek Speakers"


def test_migrate_v1_to_v2_does_not_overwrite_existing_input_device():
    data = _v1_data(input_device="Blue Yeti")
    out = migrate(data)
    assert out["audio"]["input_device"] == "Blue Yeti"


def test_migrate_v1_to_v2_both_none_fills_both():
    data = _v1_data(input_device=None, output_device=None)
    out = migrate(data)
    assert out["audio"]["input_device"] == "TONOR"
    # v1→v2 sets "Pebble Pro"; v10→v11 upgrades to the full name
    assert out["audio"]["output_device"] == "Speakers (Creative Pebble Pro)"


def test_migrate_v1_to_v2_both_set_leaves_both():
    data = _v1_data(input_device="Blue Yeti", output_device="Realtek Speakers")
    out = migrate(data)
    assert out["audio"]["input_device"] == "Blue Yeti"
    assert out["audio"]["output_device"] == "Realtek Speakers"


def test_load_config_migrates_v1_file_on_disk(tmp_path: Path):
    """End-to-end: a v1 JSON file on disk gets fully migrated to current version."""
    p = tmp_path / "config.json"
    v1 = _v1_data(input_device=None, output_device=None)
    p.write_text(json.dumps(v1), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.schema_version == CURRENT_SCHEMA_VERSION
    assert cfg.audio.input_device == "TONOR"
    # v1→v2 sets "Pebble Pro"; v10→v11 upgrades to the full name
    assert cfg.audio.output_device == "Speakers (Creative Pebble Pro)"


# --- v2 → v3 STT model default migration ---------------------------------


def _v2_data(**stt_overrides: object) -> dict:
    """Build a minimal v2 config dict with optional stt field overrides."""
    base = JarvisConfig().model_dump(mode="json")
    base["schema_version"] = 2
    base["stt"].update(stt_overrides)
    return base


def test_migrate_v2_to_v3_preserves_explicit_model_size():
    """Existing configs with any model_size must NOT be overwritten — it was
    an explicit user choice (or the previous default 'base')."""
    for existing in ("base", "base.en", "small", "tiny"):
        data = _v2_data(model_size=existing)
        out = migrate(data)
        assert out["schema_version"] == CURRENT_SCHEMA_VERSION
        assert out["stt"]["model_size"] == existing, f"overwritten for {existing!r}"


def test_migrate_v2_to_v3_bumps_version_only():
    """v2→v3 migration is a no-op except for the version bump."""
    data = _v2_data()
    stt_before = dict(data["stt"])
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert out["stt"] == stt_before


def test_fresh_install_gets_tiny_en_default():
    """New installs (no config on disk) get 'tiny.en' as the STT model."""
    cfg = JarvisConfig()
    assert cfg.stt.model_size == "tiny.en"


def test_load_config_migrates_v2_file_on_disk(tmp_path: Path):
    """End-to-end: a v2 config with model_size='base' survives migration intact."""
    p = tmp_path / "config.json"
    v2 = _v2_data(model_size="base")
    p.write_text(json.dumps(v2), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.schema_version == CURRENT_SCHEMA_VERSION
    assert cfg.stt.model_size == "base"


# --- v4 → v5 sleep_confirmation field ------------------------------------


def _v4_data(**general_overrides: object) -> dict:
    """Build a minimal v4 config dict (no sleep_confirmation field)."""
    base = JarvisConfig().model_dump(mode="json")
    base["schema_version"] = 4
    base["general"].pop("sleep_confirmation", None)
    base["general"].update(general_overrides)
    return base


def test_migrate_v4_to_v5_adds_sleep_confirmation_default():
    data = _v4_data()
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert out["general"]["sleep_confirmation"] is True


def test_migrate_v4_to_v5_preserves_explicit_sleep_confirmation():
    """If a v4 dict already carries the field (e.g. test fixture), keep it."""
    data = _v4_data(sleep_confirmation=False)
    out = migrate(data)
    assert out["general"]["sleep_confirmation"] is False


def test_load_config_migrates_v4_file_on_disk(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_v4_data()), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.schema_version == CURRENT_SCHEMA_VERSION
    assert cfg.general.sleep_confirmation is True


def test_fresh_install_has_sleep_confirmation_true():
    cfg = JarvisConfig()
    assert cfg.general.sleep_confirmation is True


# --- v5 → v6 lifecycle section migration ----------------------------------


def _v5_data() -> dict:
    base = JarvisConfig().model_dump(mode="json")
    base["schema_version"] = 5
    base.pop("lifecycle", None)
    return base


def test_migrate_v5_to_v6_adds_lifecycle_defaults():
    data = _v5_data()
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert out["lifecycle"]["auto_sleep_enabled"] is False
    assert out["lifecycle"]["idle_timeout_minutes"] == 30
    assert out["lifecycle"]["auto_sleep_on_low_battery"] is False
    assert out["lifecycle"]["auto_sleep_on_user_idle"] is False


def test_migrate_v5_to_v6_preserves_explicit_lifecycle():
    data = _v5_data()
    data["lifecycle"] = {
        "auto_sleep_enabled": True,
        "idle_timeout_minutes": 15,
        "auto_sleep_on_low_battery": False,
        "auto_sleep_on_user_idle": False,
    }
    out = migrate(data)
    assert out["lifecycle"]["auto_sleep_enabled"] is True
    assert out["lifecycle"]["idle_timeout_minutes"] == 15


def test_load_config_migrates_v5_file_on_disk(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_v5_data()), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.schema_version == CURRENT_SCHEMA_VERSION
    assert cfg.lifecycle.auto_sleep_enabled is False
    assert cfg.lifecycle.idle_timeout_minutes == 30


def test_fresh_install_has_lifecycle_defaults():
    cfg = JarvisConfig()
    assert cfg.lifecycle.auto_sleep_enabled is False
    assert cfg.lifecycle.idle_timeout_minutes == 30


# --- v6 → v7 system prompt + continuity + debug --------------------------


def _v6_data(system_prompt: str | None = None) -> dict:
    """Build a minimal v6 config dict with optional system_prompt override."""
    base = JarvisConfig().model_dump(mode="json")
    base["schema_version"] = 6
    base.pop("debug", None)
    base["llm"].pop("conversation_continuity_seconds", None)
    if system_prompt is not None:
        base["llm"]["system_prompt"] = system_prompt
    else:
        base["llm"]["system_prompt"] = _SYSTEM_PROMPT_V6
    return base


def test_migrate_v6_to_v7_replaces_default_system_prompt():
    data = _v6_data()
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert "Don't hallucinate" in out["llm"]["system_prompt"]


def test_migrate_v6_to_v7_preserves_custom_system_prompt():
    data = _v6_data(system_prompt="My custom prompt")
    out = migrate(data)
    assert out["llm"]["system_prompt"] == "My custom prompt"


def test_migrate_v6_to_v7_adds_conversation_continuity_seconds():
    data = _v6_data()
    assert "conversation_continuity_seconds" not in data["llm"]
    out = migrate(data)
    assert out["llm"]["conversation_continuity_seconds"] == 60


def test_migrate_v6_to_v7_adds_debug_section():
    data = _v6_data()
    assert "debug" not in data
    out = migrate(data)
    # Full chain runs v6→v7→v8→v9; v9 sets log_wake_during_speaking=False.
    assert out["debug"]["log_wake_during_speaking"] is False


def test_load_config_migrates_v6_file_on_disk(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_v6_data()), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.schema_version == CURRENT_SCHEMA_VERSION
    assert "Don't hallucinate" in cfg.llm.system_prompt
    assert cfg.llm.conversation_continuity_seconds == 60
    # v8→v9 migration disables debug flag (AEC diagnosis complete).
    assert cfg.debug.log_wake_during_speaking is False


def test_fresh_install_has_continuity_and_debug_defaults():
    cfg = JarvisConfig()
    assert cfg.llm.conversation_continuity_seconds == 60
    assert cfg.debug.log_wake_during_speaking is False
    assert cfg.weather.latitude is None
    assert cfg.weather.longitude is None
    assert cfg.weather.unit == "fahrenheit"


# --- v7 → v8 wake debug + weather section --------------------------------


def _v7_data() -> dict:
    """Build a minimal v7 config dict (no weather section, debug=False)."""
    base = JarvisConfig().model_dump(mode="json")
    base["schema_version"] = 7
    base.pop("weather", None)
    base["debug"]["log_wake_during_speaking"] = False
    return base


def test_migrate_v7_to_v8_enables_debug_flag():
    data = _v7_data()
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    # Full chain v7→v8→v9; v9 sets flag back to False.
    assert out["debug"]["log_wake_during_speaking"] is False


def test_migrate_v7_to_v8_adds_weather_section():
    data = _v7_data()
    assert "weather" not in data
    out = migrate(data)
    # v8 adds section with None; v9 fills Reston; v10 updates to Oakwood;
    # v18 resets those stamped defaults back to None for privacy.
    assert out["weather"]["latitude"] is None
    assert out["weather"]["longitude"] is None
    assert out["weather"]["unit"] == "fahrenheit"


def test_migrate_v7_to_v8_preserves_existing_weather():
    data = _v7_data()
    data["weather"] = {"latitude": 40.7, "longitude": -74.0, "unit": "celsius"}
    out = migrate(data)
    # Explicit coords must not be overwritten by v9.
    assert out["weather"]["latitude"] == 40.7
    assert out["weather"]["unit"] == "celsius"


def test_load_config_migrates_v7_file_on_disk(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_v7_data()), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.schema_version == CURRENT_SCHEMA_VERSION
    assert cfg.debug.log_wake_during_speaking is False
    # v18 reset wipes the stamped Oakwood default.
    assert cfg.weather.latitude is None
    assert cfg.weather.unit == "fahrenheit"


# --- v8 → v9 disable debug flag + fill weather coords -------------------


def _v8_data() -> dict:
    """Build a minimal v8 config dict (debug flag True, weather lat/lon None)."""
    base = JarvisConfig().model_dump(mode="json")
    base["schema_version"] = 8
    base["debug"]["log_wake_during_speaking"] = True
    base["weather"]["latitude"] = None
    base["weather"]["longitude"] = None
    return base


def test_migrate_v8_to_v9_disables_debug_flag():
    data = _v8_data()
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert out["debug"]["log_wake_during_speaking"] is False


def test_migrate_v8_to_v9_fills_none_weather_coords():
    data = _v8_data()
    out = migrate(data)
    # v9 fills Reston; v10 updates to Oakwood; v18 wipes the stamped
    # default so the weather tool's ipapi.co fallback runs.
    assert out["weather"]["latitude"] is None
    assert out["weather"]["longitude"] is None


def test_migrate_v8_to_v9_preserves_explicit_weather_coords():
    data = _v8_data()
    data["weather"]["latitude"] = 51.5074
    data["weather"]["longitude"] = -0.1278
    out = migrate(data)
    assert out["weather"]["latitude"] == 51.5074
    assert out["weather"]["longitude"] == -0.1278


def test_load_config_migrates_v8_file_on_disk(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_v8_data()), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.schema_version == CURRENT_SCHEMA_VERSION
    assert cfg.debug.log_wake_during_speaking is False
    # v18 reset clears the migration-stamped default coords.
    assert cfg.weather.latitude is None
    assert cfg.weather.longitude is None


# --- v9 → v10 change default location from Reston to Oakwood ------------


def _v9_data(lat: float = 38.9586, lon: float = -77.3570) -> dict:
    """Build a minimal v9 config dict with Reston defaults (or overrides)."""
    base = JarvisConfig().model_dump(mode="json")
    base["schema_version"] = 9
    base["weather"]["latitude"] = lat
    base["weather"]["longitude"] = lon
    return base


def test_migrate_v9_to_v10_replaces_reston_with_oakwood():
    data = _v9_data()  # Reston defaults
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    # v10 sets Oakwood, then v18 clears it for privacy.
    assert out["weather"]["latitude"] is None
    assert out["weather"]["longitude"] is None


def test_migrate_v9_to_v10_preserves_custom_coords():
    """Non-stamped coordinates (user-configured or IP-detected) must not change."""
    data = _v9_data(lat=51.5074, lon=-0.1278)  # London
    out = migrate(data)
    assert out["weather"]["latitude"] == 51.5074
    assert out["weather"]["longitude"] == -0.1278


def test_load_config_migrates_v9_file_on_disk(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_v9_data()), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.schema_version == CURRENT_SCHEMA_VERSION
    assert cfg.weather.latitude is None
    assert cfg.weather.longitude is None


# ---------------------------------------------------------------------------
# v10 → v11: tighten "Pebble Pro" output-device default
# ---------------------------------------------------------------------------


def _v10_data(output_device: str | None = "Pebble Pro") -> dict:
    """Return a minimal v10 config dict — built manually to avoid running
    the v10→v11 migration before the test is ready to call it."""
    from jarvis.core.config import _migrate_v9_to_v10
    data = _v9_data()
    data = _migrate_v9_to_v10(data)  # v9 → v10 only
    data.setdefault("audio", {})["output_device"] = output_device
    return data


def test_migrate_v10_to_v11_upgrades_pebble_pro_default():
    """Exact 'Pebble Pro' string (set by the v1→v2 migration) is replaced
    with the more-specific 'Speakers (Creative Pebble Pro)'."""
    data = _v10_data(output_device="Pebble Pro")
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert out["audio"]["output_device"] == "Speakers (Creative Pebble Pro)"


def test_migrate_v10_to_v11_preserves_custom_output_device():
    """User-customised output_device strings are never overwritten."""
    data = _v10_data(output_device="HDMI Audio Out")
    out = migrate(data)
    assert out["audio"]["output_device"] == "HDMI Audio Out"


def test_migrate_v10_to_v11_preserves_already_correct_default():
    """If the field already has the target value, it is left as-is."""
    data = _v10_data(output_device="Speakers (Creative Pebble Pro)")
    out = migrate(data)
    assert out["audio"]["output_device"] == "Speakers (Creative Pebble Pro)"


# ---------------------------------------------------------------------------
# v13 → v14: research.api_key
# ---------------------------------------------------------------------------


def test_migrate_v13_to_v14_adds_research_section():
    data = {"schema_version": 13, "ui": {"research_panel_width": 500}}
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert out["research"]["api_key"] is None
    assert out["ui"]["research_panel_width"] == 500


# ---------------------------------------------------------------------------
# v14 → v15: workspace.apps
# ---------------------------------------------------------------------------


def test_migrate_v14_to_v15_adds_workspace_section():
    data = {"schema_version": 14, "research": {"api_key": None}}
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    apps = out["workspace"]["apps"]
    assert len(apps) >= 1
    assert apps[0]["label"]
    assert apps[0]["kind"] in ("installed_app", "executable", "shell")
    assert apps[0]["target"]


def test_migrate_v14_to_v15_preserves_existing_workspace():
    custom = {
        "apps": [
            {"label": "VS Code", "kind": "installed_app", "target": "code"},
        ],
    }
    data = {"schema_version": 14, "workspace": custom}
    out = migrate(data)
    assert out["workspace"] == custom


# ---------------------------------------------------------------------------
# v15 → v16: deep research planner/worker fields
# ---------------------------------------------------------------------------


def test_migrate_v15_to_v16_adds_deep_research_defaults():
    data = {"schema_version": 15, "research": {"api_key": None}}
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    r = out["research"]
    assert r["planner_model"] == ""
    assert r["worker_model"] == ""
    assert r["depth"] == 5
    assert r["breadth"] == 5
    assert r["fetch_pages"] is True
    assert r["max_page_chars"] == 6000
    assert r["enable_gap_fill"] is True


def test_migrate_v15_to_v16_preserves_existing_research_values():
    data = {
        "schema_version": 15,
        "research": {
            "api_key": None,
            "planner_model": "llama3.1:8b",
            "worker_model": "qwen2.5:3b-instruct",
            "depth": 7,
            "breadth": 4,
            "fetch_pages": False,
            "max_page_chars": 12000,
            "enable_gap_fill": False,
        },
    }
    out = migrate(data)
    r = out["research"]
    assert r["planner_model"] == "llama3.1:8b"
    assert r["worker_model"] == "qwen2.5:3b-instruct"
    assert r["depth"] == 7
    assert r["breadth"] == 4
    assert r["fetch_pages"] is False
    assert r["max_page_chars"] == 12000
    assert r["enable_gap_fill"] is False


# ---------------------------------------------------------------------------
# v16 → v17: deep research Ultra
# ---------------------------------------------------------------------------


def test_migrate_v16_to_v17_adds_ultra_defaults():
    data = {
        "schema_version": 16,
        "research": {
            "api_key": None,
            "planner_model": "",
            "worker_model": "",
            "depth": 5,
            "breadth": 5,
            "fetch_pages": True,
            "max_page_chars": 6000,
            "enable_gap_fill": True,
        },
    }
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    r = out["research"]
    assert r["ultra_enabled"] is False
    assert r["brave_api_key"] == ""
    assert r["groq_api_key"] == ""
    assert r["ultra_planner_model"] == "groq/llama-3.3-70b-versatile"
    assert r["ultra_gap_fill_iterations"] == 3


def test_defaults_include_ultra_fields():
    c = JarvisConfig()
    assert c.research.ultra_enabled is False
    assert c.research.ultra_gap_fill_iterations == 3


# ---------------------------------------------------------------------------
# v17 → v18: privacy reset of migration-stamped weather coords
# ---------------------------------------------------------------------------


def _v17_data(lat: float | None, lon: float | None) -> dict:
    """Minimal v17 config dict with explicit weather coords."""
    base = JarvisConfig().model_dump(mode="json")
    base["schema_version"] = 17
    base["weather"]["latitude"] = lat
    base["weather"]["longitude"] = lon
    return base


def test_migrate_v17_to_v18_clears_stamped_oakwood():
    """The maintainer's location stamped by older migrations is wiped out."""
    data = _v17_data(lat=39.7187, lon=-84.1736)
    out = migrate(data)
    assert out["schema_version"] == CURRENT_SCHEMA_VERSION
    assert out["weather"]["latitude"] is None
    assert out["weather"]["longitude"] is None


def test_migrate_v17_to_v18_clears_stamped_reston():
    """The earlier Reston stamp (v8→v9 default fill) is also wiped."""
    data = _v17_data(lat=38.9586, lon=-77.3570)
    out = migrate(data)
    assert out["weather"]["latitude"] is None
    assert out["weather"]["longitude"] is None


def test_migrate_v17_to_v18_preserves_user_coords():
    """Coordinates that do not match a stamped pair are kept verbatim."""
    data = _v17_data(lat=51.5074, lon=-0.1278)  # London
    out = migrate(data)
    assert out["weather"]["latitude"] == 51.5074
    assert out["weather"]["longitude"] == -0.1278


def test_migrate_v17_to_v18_preserves_none():
    """Already-cleared coords stay None and don't crash the reset."""
    data = _v17_data(lat=None, lon=None)
    out = migrate(data)
    assert out["weather"]["latitude"] is None
    assert out["weather"]["longitude"] is None


def test_migrate_v17_to_v18_preserves_partial_match():
    """Only the exact stamped pair triggers reset; partial match is safe."""
    data = _v17_data(lat=39.7187, lon=-100.0)  # lat matches Oakwood, lon doesn't
    out = migrate(data)
    assert out["weather"]["latitude"] == 39.7187
    assert out["weather"]["longitude"] == -100.0


# --- v18 -> v19: command_palette hotkey + first_run_completed -------------


def test_migrate_v18_to_v19_adds_command_palette_default():
    data = {"schema_version": 18}
    out = migrate(data)
    assert out["hotkeys"]["command_palette"] == "ctrl+shift+p"


def test_migrate_v18_to_v19_preserves_existing_command_palette():
    data = {
        "schema_version": 18,
        "hotkeys": {"command_palette": "ctrl+alt+space"},
    }
    out = migrate(data)
    assert out["hotkeys"]["command_palette"] == "ctrl+alt+space"


def test_migrate_v18_to_v19_marks_first_run_completed_for_upgrades():
    """Existing installs upgrading from v18 should NOT see the onboarding
    pop-up — they've been using Jarvis already."""
    data = {"schema_version": 18}
    out = migrate(data)
    assert out["general"]["first_run_completed"] is True


def test_migrate_v18_to_v19_preserves_explicit_first_run_flag():
    data = {
        "schema_version": 18,
        "general": {"first_run_completed": False},
    }
    out = migrate(data)
    assert out["general"]["first_run_completed"] is False


# --- v19 -> v20: vision section for see_screen tool ----------------------


def test_migrate_v19_to_v20_adds_vision_section_with_defaults():
    data = {"schema_version": 19}
    out = migrate(data)
    assert out["vision"]["model"] == "llava:7b"
    assert out["vision"]["max_image_dim"] == 1280
    assert out["vision"]["max_tokens"] == 512
    assert out["vision"]["temperature"] == 0.2


def test_migrate_v19_to_v20_preserves_existing_vision_settings():
    data = {
        "schema_version": 19,
        "vision": {
            "model": "moondream",
            "max_image_dim": 640,
            "max_tokens": 256,
            "temperature": 0.5,
        },
    }
    out = migrate(data)
    assert out["vision"]["model"] == "moondream"
    assert out["vision"]["max_image_dim"] == 640
    assert out["vision"]["max_tokens"] == 256
    assert out["vision"]["temperature"] == 0.5


