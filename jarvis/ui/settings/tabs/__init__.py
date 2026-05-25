"""Per-section tabs that compose into SettingsWindow's QTabWidget.

Every tab follows the same contract:
  - Constructor: (*, config: JarvisConfig, on_change: Callable[[], None],
                  ...optional tab-specific extras)
  - Reads field values from `config` at construction.
  - On any user edit: mutates the matching config field in-place,
    persists via save_config(config), then invokes on_change() so the
    composition root can republish the ConfigChanged events the audio
    loop subscribes to.

The ConfigManager wrapper noted in core/config.py header (debounced
writes, ConfigChanged event publishing) is the Phase 6 wiring target.
For Phase 5 the tabs call save_config + on_change directly; the
composition root's on_change implementation is what bridges to the
audio loop's event bus."""

from jarvis.ui.settings.tabs.about import AboutTab
from jarvis.ui.settings.tabs.general import GeneralTab
from jarvis.ui.settings.tabs.help import HelpTab
from jarvis.ui.settings.tabs.hotkeys import HotkeysTab
from jarvis.ui.settings.tabs.models import ModelsTab
from jarvis.ui.settings.tabs.tools import ToolsTab
from jarvis.ui.settings.tabs.voice import VoiceTab

__all__ = [
    "AboutTab",
    "GeneralTab",
    "HelpTab",
    "HotkeysTab",
    "ModelsTab",
    "ToolsTab",
    "VoiceTab",
]
