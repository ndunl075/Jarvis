"""Settings UI package — the QMainWindow + six-tab QTabWidget that the
tray "Settings…" action opens. Each tab in `tabs/` owns its slice of
JarvisConfig; the main window just composes them and routes the shared
`on_change` callback."""

from jarvis.ui.settings.main_window import SettingsWindow

__all__ = ["SettingsWindow"]
