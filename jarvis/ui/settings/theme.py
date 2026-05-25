"""Visual design tokens and Qt stylesheet for the Settings window.

Palette: near-black background, white primary text, subtle gray secondary
text. Accent is pure white. Inspired by the sparse, high-contrast minimal
aesthetic of the spiral-animation reference — letterSpacing, thin fonts,
no heavy chrome.

Usage
-----
    from jarvis.ui.settings.theme import apply_theme
    apply_theme(window)   # call once in SettingsWindow.__init__
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------

BG            = "#0a0a0a"
TEXT_PRIMARY  = "#ffffff"
TEXT_SECONDARY = "#a0a0a0"
ACCENT        = "#ffffff"
BORDER        = "#1f1f1f"
INPUT_BG      = "#141414"
HOVER_BG      = "#181818"

# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

_QSS = f"""
/* ── base ─────────────────────────────────────────────────────────────── */

QMainWindow,
QDialog,
QWidget {{
    background-color: {BG};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-weight: 300;
    font-size: 11pt;
}}

/* ── tabs ──────────────────────────────────────────────────────────────── */

QTabWidget::pane {{
    border: 1px solid {BORDER};
    background-color: {BG};
}}

QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    padding: 12px 24px;
    font-size: 10pt;
    letter-spacing: 3px;
    border: none;
    border-bottom: 1px solid transparent;
    text-transform: uppercase;
}}

QTabBar::tab:selected {{
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {TEXT_PRIMARY};
    background-color: transparent;
}}

QTabBar::tab:hover:!selected {{
    color: #d0d0d0;
    background-color: transparent;
}}

/* ── labels ────────────────────────────────────────────────────────────── */

QLabel {{
    color: {TEXT_PRIMARY};
    font-size: 11pt;
    font-weight: 300;
    background-color: transparent;
}}

/* ── text inputs ───────────────────────────────────────────────────────── */

QLineEdit,
QPlainTextEdit,
QSpinBox,
QDoubleSpinBox {{
    background-color: {INPUT_BG};
    color: {TEXT_PRIMARY};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px 4px;
    selection-background-color: {ACCENT};
    selection-color: #000000;
}}

QLineEdit:focus,
QPlainTextEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {{
    border-bottom: 1px solid {ACCENT};
    background-color: {INPUT_BG};
}}

/* hide the spin-box up/down buttons — keep it minimal */
QSpinBox::up-button,
QSpinBox::down-button,
QDoubleSpinBox::up-button,
QDoubleSpinBox::down-button {{
    width: 0;
    height: 0;
    border: none;
}}

/* ── combo boxes ───────────────────────────────────────────────────────── */

QComboBox {{
    background-color: {INPUT_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    padding: 7px 12px;
    selection-background-color: {ACCENT};
    selection-color: #000000;
}}

QComboBox:hover {{
    border-color: #404040;
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG};
    color: {TEXT_PRIMARY};
    selection-background-color: {HOVER_BG};
    selection-color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    outline: none;
}}

/* ── checkboxes ────────────────────────────────────────────────────────── */

QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 10px;
    font-weight: 300;
    background-color: transparent;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid #404040;
    background-color: transparent;
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    image: none;
}}

QCheckBox::indicator:hover {{
    border-color: #808080;
}}

/* ── sliders ───────────────────────────────────────────────────────────── */

QSlider::groove:horizontal {{
    height: 2px;
    background-color: {BORDER};
    margin: 0;
}}

QSlider::sub-page:horizontal {{
    background-color: {ACCENT};
    height: 2px;
}}

QSlider::handle:horizontal {{
    background-color: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background-color: #e0e0e0;
}}

/* ── push buttons ──────────────────────────────────────────────────────── */

QPushButton {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    border: 1px solid #404040;
    padding: 10px 24px;
    font-size: 10pt;
    letter-spacing: 3px;
    font-weight: 300;
    text-transform: uppercase;
}}

QPushButton:hover {{
    border-color: {ACCENT};
    background-color: {HOVER_BG};
}}

QPushButton:pressed {{
    background-color: {BORDER};
}}

QPushButton:disabled {{
    color: #404040;
    border-color: #282828;
}}

/* ── group boxes ───────────────────────────────────────────────────────── */

QGroupBox {{
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    border-radius: 0;
    margin-top: 20px;
    padding-top: 20px;
    font-size: 9pt;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 300;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
}}

/* ── scrollbars ─────────────────────────────────────────────────────────  */

QScrollBar:vertical {{
    background: {BG};
    width: 6px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: #333333;
    border-radius: 3px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background: #555555;
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {BG};
    height: 6px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: #333333;
    border-radius: 3px;
    min-width: 20px;
}}

QScrollBar::handle:horizontal:hover {{
    background: #555555;
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── status bar ────────────────────────────────────────────────────────── */

QStatusBar {{
    color: {TEXT_SECONDARY};
    background-color: {BG};
    border-top: 1px solid {BORDER};
    font-size: 9pt;
    font-weight: 300;
}}

QStatusBar::item {{
    border: none;
}}

/* ── message boxes ─────────────────────────────────────────────────────── */

QMessageBox {{
    background-color: {BG};
    color: {TEXT_PRIMARY};
}}

QMessageBox QPushButton {{
    min-width: 80px;
}}

/* ── tooltips ──────────────────────────────────────────────────────────── */

QToolTip {{
    background-color: {INPUT_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    padding: 4px 8px;
    font-size: 9pt;
}}
"""


def apply_theme(window: QWidget) -> None:
    """Apply the minimal black/white stylesheet to *window* and any children."""
    window.setStyleSheet(_QSS)
