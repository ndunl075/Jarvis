"""Smoke tests for jarvis.ui.settings.

One test per tab: construct with a default config, verify population,
edit one widget, verify config mutation + save_config call + on_change
fire. No Qt event loop is started — widgets are constructed but not
shown. The offscreen qapp fixture (tests/ui/conftest.py) is what makes
QWidget construction work in headless CI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.core.config import JarvisConfig, MCPServerConfig
from jarvis.ui.settings import SettingsWindow
from jarvis.ui.settings.tabs import (
    AboutTab,
    GeneralTab,
    HotkeysTab,
    ModelsTab,
    ToolsTab,
    VoiceTab,
)


def _cfg() -> JarvisConfig:
    return JarvisConfig()


# --- main window -------------------------------------------------------


def test_close_hides_window_not_destroys(qapp):
    with patch("jarvis.ui.settings.tabs.about.AboutTab._refresh_status"):
        win = SettingsWindow(config=_cfg(), on_change=lambda: None)
    win.show()
    assert win.isVisible()
    win.close()
    assert not win.isVisible()
    # Object is still alive — attributes remain accessible
    assert win.windowTitle() == "Jarvis Settings"


def test_show_after_close_raises_window(qapp):
    with patch("jarvis.ui.settings.tabs.about.AboutTab._refresh_status"):
        win = SettingsWindow(config=_cfg(), on_change=lambda: None)
    win.show()
    win.close()
    assert not win.isVisible()
    win.show()
    assert win.isVisible()
    win.hide()


def test_repeated_close_show_cycles(qapp):
    with patch("jarvis.ui.settings.tabs.about.AboutTab._refresh_status"):
        win = SettingsWindow(config=_cfg(), on_change=lambda: None)
    for _ in range(3):
        win.show()
        assert win.isVisible()
        win.close()
        assert not win.isVisible()


def test_main_window_has_six_tabs_in_order(qapp):
    cfg = _cfg()
    on_change = MagicMock()
    with patch("jarvis.ui.settings.tabs.about.AboutTab._refresh_status"):
        win = SettingsWindow(config=cfg, on_change=on_change)
    try:
        labels = [win._tabs.tabText(i) for i in range(win._tabs.count())]
        assert labels == [
            "General", "Voice", "Models", "Tools", "Hotkeys", "Help", "About",
        ]
    finally:
        win.close()


def test_save_indicator_appears_on_change(qapp):
    """After any tab triggers on_change, the status bar shows '✓ Saved'."""
    cfg = _cfg()
    outer_called = MagicMock()
    with patch("jarvis.ui.settings.tabs.about.AboutTab._refresh_status"):
        win = SettingsWindow(config=cfg, on_change=outer_called)
    try:
        # Trigger the wrapped on_change by calling it directly through the
        # GeneralTab's internal callback (same wrapper all tabs share).
        win.general._on_change()
        assert win._status_bar.currentMessage() == "✓ Saved"
        outer_called.assert_called()
    finally:
        win.close()


def test_main_window_title_and_default_size(qapp):
    cfg = _cfg()
    with patch("jarvis.ui.settings.tabs.about.AboutTab._refresh_status"):
        win = SettingsWindow(config=cfg, on_change=lambda: None)
    try:
        assert win.windowTitle() == "Jarvis Settings"
        # resize() is a hint; the actual size may differ on platform but
        # the request value should be retained as a default sizeHint
        # baseline.
        size = win.size()
        assert size.width() == 780
        assert size.height() == 560
    finally:
        win.close()


# --- GeneralTab --------------------------------------------------------


def test_general_tab_populates_from_config(qapp):
    cfg = _cfg()
    cfg.general.start_with_windows = True
    cfg.general.minimize_to_tray = False
    cfg.general.show_overlay = False
    cfg.general.log_level = "DEBUG"
    cfg.weather.latitude = 51.5074
    cfg.weather.longitude = -0.1278
    tab = GeneralTab(config=cfg, on_change=lambda: None)
    assert tab.start_with_windows.isChecked() is True
    assert tab.minimize_to_tray.isChecked() is False
    assert tab.show_overlay.isChecked() is False
    assert tab.log_level.currentText() == "DEBUG"
    assert abs(tab.weather_lat.value() - 51.5074) < 0.0001
    assert abs(tab.weather_lon.value() - (-0.1278)) < 0.0001
    assert tab.workspace_list.count() == len(cfg.workspace.apps)


def test_general_tab_toggle_writes_back_and_persists(qapp):
    cfg = _cfg()
    on_change = MagicMock()
    with patch("jarvis.ui.settings.tabs.general.save_config") as save:
        tab = GeneralTab(config=cfg, on_change=on_change)
        tab.show_overlay.setChecked(not cfg.general.show_overlay)
    assert cfg.general.show_overlay is tab.show_overlay.isChecked()
    save.assert_called()
    on_change.assert_called()


# --- VoiceTab ----------------------------------------------------------


def test_voice_tab_populates_sliders_from_config(qapp):
    cfg = _cfg()
    cfg.tts.speed = 1.25
    cfg.tts.volume = 0.5
    cfg.wake_word.sensitivity = 0.7
    with patch(
        "jarvis.ui.settings.tabs.voice._enumerate_devices",
        return_value=[("Mic A", "A"), ("Mic B", "B")],
    ), patch(
        "jarvis.ui.settings.tabs.voice._enumerate_voices",
        return_value=["en_GB-alan-medium", "en_US-amy-low"],
    ):
        tab = VoiceTab(config=cfg, on_change=lambda: None)
    assert tab.speed.value() == 125
    assert tab.volume.value() == 50
    assert tab.sensitivity.value() == 70


def test_voice_tab_speed_slider_writes_back_and_labels(qapp):
    cfg = _cfg()
    on_change = MagicMock()
    with patch(
        "jarvis.ui.settings.tabs.voice._enumerate_devices",
        return_value=[],
    ), patch(
        "jarvis.ui.settings.tabs.voice._enumerate_voices",
        return_value=["en_GB-alan-medium"],
    ), patch(
        "jarvis.ui.settings.tabs.voice.save_config"
    ) as save:
        tab = VoiceTab(config=cfg, on_change=on_change)
        tab.speed.setValue(180)  # 1.80×
    assert cfg.tts.speed == 1.8
    assert "1.80" in tab.speed_label.text()
    save.assert_called()
    on_change.assert_called()


def test_voice_tab_test_voice_button_invokes_callback(qapp):
    spoken: list[str] = []
    with patch(
        "jarvis.ui.settings.tabs.voice._enumerate_devices", return_value=[]
    ), patch(
        "jarvis.ui.settings.tabs.voice._enumerate_voices",
        return_value=["en_GB-alan-medium"],
    ):
        tab = VoiceTab(
            config=_cfg(),
            on_change=lambda: None,
            on_test_voice=spoken.append,
        )
        tab.test_voice_button.click()
    assert spoken == ["hello, this is jarvis"]


def test_voice_tab_device_substring_match_selects_correctly(qapp):
    """Configured name "TONOR" should match "Microphone (TONOR TM20 Audio Device)"
    via substring — identical to the runtime audio layer — so the combo shows
    the device as connected, not as "(not connected)"."""
    cfg = _cfg()
    cfg.audio.input_device = "TONOR"
    cfg.audio.output_device = None  # keep output as system default

    with patch(
        "jarvis.ui.settings.tabs.voice._enumerate_devices",
        side_effect=lambda kind: (
            [("Microphone (TONOR TM20 Audio Device) [WASAPI]",
              "Microphone (TONOR TM20 Audio Device)")]
            if kind == "input" else []
        ),
    ), patch("jarvis.ui.settings.tabs.voice._enumerate_voices", return_value=[]):
        tab = VoiceTab(config=cfg, on_change=lambda: None)

    # Combo must not contain a "(not connected)" entry for TONOR
    input_texts = [
        tab.input_device.itemText(i) for i in range(tab.input_device.count())
    ]
    assert not any("not connected" in t for t in input_texts), (
        f"device listed as not-connected; combo items: {input_texts}"
    )
    # And the TONOR device is actually selected
    selected = tab.input_device.currentText()
    assert "TONOR" in selected


def test_voice_tab_unplugged_device_shows_not_connected(qapp):
    """When the configured device genuinely isn't in the enumeration list
    the placeholder is still shown — the fix must not suppress real failures."""
    cfg = _cfg()
    cfg.audio.input_device = "GHOST_MIC"

    with patch(
        "jarvis.ui.settings.tabs.voice._enumerate_devices",
        return_value=[("Some Other Device [WASAPI]", "Some Other Device")],
    ), patch("jarvis.ui.settings.tabs.voice._enumerate_voices", return_value=[]):
        tab = VoiceTab(config=cfg, on_change=lambda: None)

    input_texts = [
        tab.input_device.itemText(i) for i in range(tab.input_device.count())
    ]
    assert any("not connected" in t for t in input_texts)


# --- ModelsTab ---------------------------------------------------------


def test_models_tab_populates_from_config(qapp):
    cfg = _cfg()
    cfg.stt.model_size = "small"
    cfg.llm.temperature = 0.42
    cfg.llm.max_tokens = 2048
    with patch(
        "jarvis.ui.settings.tabs.models._ollama_list_models",
        return_value=["qwen2.5:7b-instruct", "llama3"],
    ):
        tab = ModelsTab(config=cfg, on_change=lambda: None)
    assert tab.stt_size.currentText() == "small"
    assert tab.llm_temperature.value() == 42
    assert tab.llm_max_tokens.value() == 2048
    assert "qwen2.5:7b-instruct" in [
        tab.llm_model.itemText(i) for i in range(tab.llm_model.count())
    ]


def test_models_tab_temperature_slider_writes_back(qapp):
    cfg = _cfg()
    on_change = MagicMock()
    with patch(
        "jarvis.ui.settings.tabs.models._ollama_list_models",
        return_value=[],
    ), patch(
        "jarvis.ui.settings.tabs.models.save_config"
    ) as save:
        tab = ModelsTab(config=cfg, on_change=on_change)
        tab.llm_temperature.setValue(120)  # 1.20
    assert cfg.llm.temperature == 1.2
    save.assert_called()
    on_change.assert_called()



# --- ToolsTab ----------------------------------------------------------


def test_tools_tab_populates_checkboxes_for_phase_4_set(qapp):
    cfg = _cfg()
    cfg.tools.enabled = {"screenshot": False}  # explicit disable
    tab = ToolsTab(config=cfg, on_change=lambda: None)
    # Every local tool shows up as a checkbox.
    assert set(tab._checkboxes.keys()) == {
        "clipboard",
        "close_app",
        "append_to_note",
        "clear_clipboard_history",
        "close_clipboard_history",
        "close_dashboard",
        "close_logs",
        "close_notes",
        "close_research",
        "copy_research",
        "deep_research",
        "delete_note",
        "pause_deep_research",
        "resume_deep_research",
        "close_deep_research",
        "delete_deep_research",
        "delete_all_deep_research",
        "get_weather",
        "launch_steam_game",
        "launch_workspace",
        "list_directory",
        "lock_screen",
        "open_app",
        "open_help",
        "open_notes",
        "open_url",
        "paste_clipboard_item",
        "play_youtube_music",
        "read_more",
        "read_note",
        "report_cpu_and_memory_percentages",
        "research",
        "screenshot",
        "see_screen",
        "show_clipboard_history",
        "show_dashboard",
        "show_logs",
        "take_note",
        "type_into_active_window",
        "volume",
        "enable_deep_research_ultra",
        "disable_deep_research_ultra",
    }
    # Explicit disable reflected; unconfigured default to enabled.
    assert tab._checkboxes["screenshot"].isChecked() is False
    assert tab._checkboxes["clipboard"].isChecked() is True


def test_tools_tab_toggle_writes_into_enabled_dict(qapp):
    cfg = _cfg()
    on_change = MagicMock()
    with patch("jarvis.ui.settings.tabs.tools.save_config") as save:
        tab = ToolsTab(config=cfg, on_change=on_change)
        tab._checkboxes["screenshot"].setChecked(False)
    assert cfg.tools.enabled["screenshot"] is False
    save.assert_called()
    on_change.assert_called()


def test_tools_tab_lists_existing_mcp_servers(qapp):
    cfg = _cfg()
    cfg.mcp_servers = [
        MCPServerConfig(name="fs", url="http://localhost:9100/mcp"),
        MCPServerConfig(name="mail", url="https://mail.local/mcp"),
    ]
    tab = ToolsTab(config=cfg, on_change=lambda: None)
    rows = [tab.mcp_list.item(i).text() for i in range(tab.mcp_list.count())]
    assert any("fs" in r for r in rows)
    assert any("mail" in r for r in rows)


# --- HotkeysTab --------------------------------------------------------


def test_hotkeys_tab_populates_capture_buttons_from_config(qapp):
    cfg = _cfg()
    cfg.hotkeys.mute = "ctrl+shift+m"
    cfg.hotkeys.push_to_talk = None
    cfg.hotkeys.open_settings = "ctrl+shift+,"
    tab = HotkeysTab(config=cfg, on_change=lambda: None)
    assert tab._capture_buttons["mute"].text() == "ctrl+shift+m"
    assert tab._capture_buttons["push_to_talk"].text() == "(none)"
    assert tab._capture_buttons["open_settings"].text() == "ctrl+shift+,"


def test_hotkeys_tab_clear_disabled_for_required_bindings(qapp):
    """mute and open_settings have can_be_empty=False; push_to_talk and
    command_palette have can_be_empty=True, so only their Clear buttons
    are enabled. Test via the form: walk each row, find the QPushButton
    labelled 'Clear', confirm enabled state matches."""
    from PySide6.QtWidgets import QHBoxLayout, QPushButton
    cfg = _cfg()
    tab = HotkeysTab(config=cfg, on_change=lambda: None)
    # Collect every Clear button child.
    clears = [
        b for b in tab.findChildren(QPushButton) if b.text() == "Clear"
    ]
    enabled_count = sum(1 for b in clears if b.isEnabled())
    # Only push_to_talk and command_palette can be cleared.
    assert enabled_count == 2


def test_hotkeys_tab_capture_writes_back_and_persists(qapp):
    cfg = _cfg()
    on_change = MagicMock()
    with patch("jarvis.ui.settings.tabs.hotkeys.save_config") as save:
        tab = HotkeysTab(config=cfg, on_change=on_change)
        # Bypass the actual key capture; invoke the captured-handler
        # directly with a synthesized combo (the capture mechanism
        # itself is the Qt slot the unit test scope can't drive cleanly).
        tab._on_captured("mute", "ctrl+alt+m")
    assert cfg.hotkeys.mute == "ctrl+alt+m"
    save.assert_called()
    on_change.assert_called()


# --- AboutTab ----------------------------------------------------------


def test_about_tab_renders_connected_when_probe_succeeds(qapp):
    cfg = _cfg()
    cfg.llm.model = "qwen2.5:7b-instruct"
    probe = MagicMock(return_value=(True, ["qwen2.5:7b-instruct", "llama3"]))
    tab = AboutTab(config=cfg, on_change=lambda: None, probe=probe)
    assert tab.ollama_status.text() == "Connected"
    assert tab.model_status.text() == "Pulled"


def test_about_tab_renders_not_detected_when_probe_fails(qapp):
    cfg = _cfg()
    probe = MagicMock(return_value=(False, []))
    tab = AboutTab(config=cfg, on_change=lambda: None, probe=probe)
    assert tab.ollama_status.text() == "Not detected"
    assert "Unknown" in tab.model_status.text()


def test_about_tab_marks_unpulled_when_model_missing(qapp):
    cfg = _cfg()
    cfg.llm.model = "exotic-model"
    probe = MagicMock(return_value=(True, ["something-else"]))
    tab = AboutTab(config=cfg, on_change=lambda: None, probe=probe)
    assert tab.model_status.text() == "Not pulled"
