"""Hotkeys tab: per-binding capture button that records the next key
combo pressed and writes it back to config in the same string format
HotkeysConfig uses ('ctrl+shift+m')."""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jarvis.core.config import JarvisConfig, save_config

log = logging.getLogger(__name__)

# The HotkeysConfig fields this tab manages, in display order.
_FIELDS: tuple[tuple[str, str, bool], ...] = (
    # (config_attr, ui_label, can_be_empty)
    ("mute", "Mute", False),
    ("push_to_talk", "Push to talk", True),
    ("open_settings", "Open settings", False),
    ("command_palette", "Command palette", True),
)


def _modifier_token(mod: Qt.KeyboardModifier) -> str | None:
    """Translate a single Qt modifier flag to its config-string form."""
    if mod == Qt.KeyboardModifier.ControlModifier:
        return "ctrl"
    if mod == Qt.KeyboardModifier.ShiftModifier:
        return "shift"
    if mod == Qt.KeyboardModifier.AltModifier:
        return "alt"
    if mod == Qt.KeyboardModifier.MetaModifier:
        return "win"
    return None


def _qt_key_to_string(key: int) -> str | None:
    """Translate a Qt.Key value to its config-string form. Returns None
    for raw modifier-only events (those don't bind to a usable hotkey)."""
    k = Qt.Key(key)
    if k in (
        Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta,
        Qt.Key.Key_AltGr,
    ):
        return None
    # Named keys we know how to round-trip through hotkeys.py.
    named = {
        Qt.Key.Key_Space: "space",
        Qt.Key.Key_Tab: "tab",
        Qt.Key.Key_Return: "enter",
        Qt.Key.Key_Enter: "enter",
        Qt.Key.Key_Escape: "esc",
        Qt.Key.Key_Backspace: "backspace",
        Qt.Key.Key_Delete: "delete",
        Qt.Key.Key_Insert: "insert",
        Qt.Key.Key_Home: "home",
        Qt.Key.Key_End: "end",
        Qt.Key.Key_PageUp: "pageup",
        Qt.Key.Key_PageDown: "pagedown",
        Qt.Key.Key_Up: "up",
        Qt.Key.Key_Down: "down",
        Qt.Key.Key_Left: "left",
        Qt.Key.Key_Right: "right",
        Qt.Key.Key_Comma: ",",
        Qt.Key.Key_Period: ".",
    }
    if k in named:
        return named[k]
    for i in range(1, 25):
        if k == getattr(Qt.Key, f"Key_F{i}"):
            return f"f{i}"
    # Alpha-numerics: Qt.Key.Key_A == 0x41 etc. Just lowercase the
    # ASCII representation.
    if Qt.Key.Key_A.value <= key <= Qt.Key.Key_Z.value:
        return chr(key).lower()
    if Qt.Key.Key_0.value <= key <= Qt.Key.Key_9.value:
        return chr(key)
    return None


def _format_combo(modifiers: Qt.KeyboardModifier, key: int) -> str | None:
    """Return the 'ctrl+shift+m' style string for a key event, or None
    if it's not bindable (modifier-only, or a key we don't translate)."""
    parts: list[str] = []
    for flag in (
        Qt.KeyboardModifier.ControlModifier,
        Qt.KeyboardModifier.ShiftModifier,
        Qt.KeyboardModifier.AltModifier,
        Qt.KeyboardModifier.MetaModifier,
    ):
        if modifiers & flag:
            token = _modifier_token(flag)
            if token:
                parts.append(token)
    key_str = _qt_key_to_string(key)
    if key_str is None:
        return None
    parts.append(key_str)
    return "+".join(parts) if parts else None


class _CaptureButton(QPushButton):
    """A QPushButton that, when clicked, listens for the next key combo
    and reports it via on_captured. Capture mode is exited as soon as a
    bindable combo is pressed, or on Escape (cancel). While capturing,
    the button text shows "Press keys…"."""

    def __init__(
        self,
        *,
        current: str,
        on_captured: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_captured = on_captured
        self._capturing = False
        self._current = current
        self._render()
        self.clicked.connect(self._start_capture)
        # Need focus to receive key events.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_current(self, value: str) -> None:
        self._current = value
        self._render()

    def _render(self) -> None:
        if self._capturing:
            self.setText("Press keys…")
        elif self._current:
            self.setText(self._current)
        else:
            self.setText("(none)")

    def _start_capture(self) -> None:
        self._capturing = True
        self._render()
        self.setFocus(Qt.FocusReason.MouseFocusReason)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if not self._capturing:
            return super().keyPressEvent(event)
        if event.key() == Qt.Key.Key_Escape:
            # Cancel without changing.
            self._capturing = False
            self._render()
            return
        combo = _format_combo(event.modifiers(), event.key())
        if combo is None:
            # Modifier-only press; keep waiting.
            return
        self._capturing = False
        self._current = combo
        self._render()
        try:
            self._on_captured(combo)
        except Exception:
            log.exception("hotkey capture callback raised")

    def event(self, e: QEvent) -> bool:
        # Cancel capture if focus is lost (user clicked elsewhere) so
        # the button doesn't sit forever waiting.
        if e.type() == QEvent.Type.FocusOut and self._capturing:
            self._capturing = False
            self._render()
        return super().event(e)


class HotkeysTab(QWidget):
    def __init__(
        self,
        *,
        config: JarvisConfig,
        on_change: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._on_change = on_change

        self._capture_buttons: dict[str, _CaptureButton] = {}
        form = QFormLayout()
        for field, label, can_be_empty in _FIELDS:
            current = getattr(config.hotkeys, field) or ""
            btn = _CaptureButton(
                current=current,
                on_captured=lambda combo, f=field: self._on_captured(f, combo),
            )
            clear = QPushButton("Clear")
            clear.setEnabled(can_be_empty)
            clear.clicked.connect(
                lambda _checked, f=field: self._on_clear(f)
            )
            row = QHBoxLayout()
            row.addWidget(btn, 1)
            row.addWidget(clear)
            form.addRow(f"{label}:", row)
            self._capture_buttons[field] = btn

        hint = QLabel(
            "Click a binding to capture a new key combo. "
            "Press Escape to cancel a capture in progress."
        )
        hint.setWordWrap(True)
        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(hint)
        root.addStretch(1)

    def _on_captured(self, field: str, combo: str) -> None:
        setattr(self._config.hotkeys, field, combo)
        self._persist()

    def _on_clear(self, field: str) -> None:
        # Only callable for fields that allow empty. push_to_talk uses
        # None (Optional[str]); command_palette is a plain str so we
        # write "" to disable.
        from jarvis.core.config import HotkeysConfig

        model_field = HotkeysConfig.model_fields.get(field)
        cleared = ""
        if model_field is not None:
            anno = model_field.annotation
            if anno is not None and "None" in str(anno):
                cleared = None  # type: ignore[assignment]
        setattr(self._config.hotkeys, field, cleared)
        btn = self._capture_buttons.get(field)
        if btn is not None:
            btn.set_current("")
        self._persist()

    def _persist(self) -> None:
        try:
            save_config(self._config)
        except Exception:
            log.exception("save_config failed from HotkeysTab")
            return
        try:
            self._on_change()
        except Exception:
            log.exception("on_change callback raised from HotkeysTab")
