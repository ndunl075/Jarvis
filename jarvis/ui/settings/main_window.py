"""SettingsWindow — the QMainWindow that hosts the six-tab QTabWidget.

Opens from the tray "Settings…" menu item (and the open_settings
hotkey). Non-modal: closing it doesn't quit Jarvis. The composition
root keeps a single instance alive across opens; show() / raise_()
brings it back into focus."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QStatusBar, QTabWidget, QWidget

from jarvis.core.config import JarvisConfig, save_config

if TYPE_CHECKING:
    from PySide6.QtGui import QCloseEvent

from jarvis.ui.settings.tabs import (
    AboutTab,
    GeneralTab,
    HelpTab,
    HotkeysTab,
    ModelsTab,
    ToolsTab,
    VoiceTab,
)
from jarvis.ui.settings.theme import apply_theme

log = logging.getLogger(__name__)

_DEFAULT_SIZE = (780, 560)


class SettingsWindow(QMainWindow):
    def __init__(
        self,
        *,
        config: JarvisConfig,
        on_change: Callable[[], None],
        voices_dir: Path | None = None,
        on_test_voice: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self.setWindowTitle("Jarvis Settings")
        self.resize(*_DEFAULT_SIZE)

        # Non-modal: open from tray, leave Jarvis running on close. The
        # composition root keeps the instance alive across opens; closing
        # just hides it. (Setting WA_DeleteOnClose would defeat that
        # pattern, so we explicitly don't.)
        self.setAttribute(
            Qt.WidgetAttribute.WA_DeleteOnClose, on=False
        )

        # Apply the minimal black/white theme before building child widgets
        # so the stylesheet cascades into every QWidget created below.
        apply_theme(self)

        self._tabs = QTabWidget(self)
        self.setCentralWidget(self._tabs)

        # Status bar: shows "✓ Saved" for 1.5 s after any tab saves config.
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        def _on_change_with_indicator() -> None:
            self._status_bar.showMessage("✓ Saved", 1500)
            on_change()

        # Construct each tab. Constructor order is also display order.
        self.general = GeneralTab(config=config, on_change=_on_change_with_indicator)
        self.voice = VoiceTab(
            config=config,
            on_change=_on_change_with_indicator,
            voices_dir=voices_dir,
            on_test_voice=on_test_voice,
        )
        self.models = ModelsTab(config=config, on_change=_on_change_with_indicator)
        self.tools = ToolsTab(config=config, on_change=_on_change_with_indicator)
        self.hotkeys = HotkeysTab(config=config, on_change=_on_change_with_indicator)
        self.help = HelpTab(config=config, on_change=_on_change_with_indicator)
        self.about = AboutTab(config=config, on_change=_on_change_with_indicator)

        self._tabs.addTab(self.general, "General")
        self._tabs.addTab(self.voice, "Voice")
        self._tabs.addTab(self.models, "Models")
        self._tabs.addTab(self.tools, "Tools")
        self._tabs.addTab(self.hotkeys, "Hotkeys")
        self._tabs.addTab(self.help, "Help")
        self._tabs.addTab(self.about, "About")

    def _autosave(self) -> None:
        """Persist config immediately (tabs also save on each change)."""
        try:
            save_config(self._config)
            self._status_bar.showMessage("✓ Saved", 1500)
        except Exception:
            log.exception("autosave failed from SettingsWindow")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._autosave()
        event.ignore()
        self.hide()

    def hideEvent(self, event) -> None:  # noqa: N802
        self._autosave()
        super().hideEvent(event)
