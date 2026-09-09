"""General-settings tab: startup, tray, overlay, log level, weather, workspace."""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from jarvis.core.config import JarvisConfig, WorkspaceAppEntry, save_config

log = logging.getLogger(__name__)

_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")

_WORKSPACE_KINDS: tuple[tuple[str, str], ...] = (
    ("installed_app", "Installed app (Start Menu name)"),
    ("executable", "Executable (.exe path)"),
    ("shell", "Shell URI (e.g. shell:AppsFolder\\…)"),
)


class GeneralTab(QWidget):
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

        self.start_with_windows = QCheckBox("Start with Windows")
        self.start_with_windows.setChecked(config.general.start_with_windows)
        self.minimize_to_tray = QCheckBox("Minimize to tray on close")
        self.minimize_to_tray.setChecked(config.general.minimize_to_tray)
        self.show_overlay = QCheckBox("Show overlay orb")
        self.show_overlay.setChecked(config.general.show_overlay)

        self.log_level = QComboBox()
        self.log_level.addItems(_LOG_LEVELS)
        self.log_level.setCurrentText(config.general.log_level)

        self.auto_sleep = QCheckBox("Auto-sleep when idle")
        self.auto_sleep.setChecked(config.lifecycle.auto_sleep_enabled)

        self.idle_timeout = QSpinBox()
        self.idle_timeout.setRange(1, 1440)
        self.idle_timeout.setSuffix(" min")
        self.idle_timeout.setValue(config.lifecycle.idle_timeout_minutes)

        # Weather location group.
        self.weather_lat = QDoubleSpinBox()
        self.weather_lat.setRange(-90.0, 90.0)
        self.weather_lat.setDecimals(4)
        self.weather_lat.setSuffix("°")
        self.weather_lat.setValue(
            config.weather.latitude if config.weather.latitude is not None else 0.0
        )

        self.weather_lon = QDoubleSpinBox()
        self.weather_lon.setRange(-180.0, 180.0)
        self.weather_lon.setDecimals(4)
        self.weather_lon.setSuffix("°")
        self.weather_lon.setValue(
            config.weather.longitude if config.weather.longitude is not None else 0.0
        )

        self.detect_location_btn = QPushButton("Detect from IP")
        self.detect_location_btn.setToolTip(
            "Auto-detect your location via ipapi.co (requires internet)"
        )
        detect_row = QHBoxLayout()
        detect_row.addWidget(self.detect_location_btn)
        detect_row.addStretch(1)

        weather_box = QGroupBox("Weather location")
        weather_form = QFormLayout(weather_box)
        weather_form.addRow("Latitude:", self.weather_lat)
        weather_form.addRow("Longitude:", self.weather_lon)
        weather_form.addRow(detect_row)

        # Workspace apps (voice: "open my workspace").
        workspace_box = QGroupBox('Workspace ("open my workspace")')
        workspace_layout = QVBoxLayout(workspace_box)
        workspace_hint = QLabel(
            "Apps launched when you say \"open my workspace\", "
            "\"launch workspace\", or \"jarvis open my workspace\"."
        )
        workspace_hint.setWordWrap(True)
        workspace_hint.setStyleSheet("color: #808080; font-size: 9pt;")
        workspace_layout.addWidget(workspace_hint)
        self.workspace_list = QListWidget()
        self._repopulate_workspace_list()
        ws_btn_row = QHBoxLayout()
        self.workspace_add = QPushButton("Add…")
        self.workspace_edit = QPushButton("Edit…")
        self.workspace_remove = QPushButton("Remove")
        ws_btn_row.addWidget(self.workspace_add)
        ws_btn_row.addWidget(self.workspace_edit)
        ws_btn_row.addWidget(self.workspace_remove)
        ws_btn_row.addStretch(1)
        workspace_layout.addWidget(self.workspace_list)
        workspace_layout.addLayout(ws_btn_row)

        root = QVBoxLayout(self)
        general_form = QFormLayout()
        general_form.addRow(self.start_with_windows)
        general_form.addRow(self.minimize_to_tray)
        general_form.addRow(self.show_overlay)
        general_form.addRow("Log level:", self.log_level)
        general_form.addRow(self.auto_sleep)
        general_form.addRow("Idle timeout:", self.idle_timeout)
        root.addLayout(general_form)
        root.addWidget(workspace_box)
        root.addWidget(weather_box)
        root.addStretch(1)

        self.start_with_windows.toggled.connect(self._on_start_toggled)
        self.minimize_to_tray.toggled.connect(self._on_tray_toggled)
        self.show_overlay.toggled.connect(self._on_overlay_toggled)
        self.log_level.currentTextChanged.connect(self._on_log_level_changed)
        self.auto_sleep.toggled.connect(self._on_auto_sleep_toggled)
        self.idle_timeout.valueChanged.connect(self._on_idle_timeout_changed)
        self.weather_lat.valueChanged.connect(self._on_weather_lat_changed)
        self.weather_lon.valueChanged.connect(self._on_weather_lon_changed)
        self.detect_location_btn.clicked.connect(self._on_detect_location)
        self.workspace_add.clicked.connect(self._on_workspace_add)
        self.workspace_edit.clicked.connect(self._on_workspace_edit)
        self.workspace_remove.clicked.connect(self._on_workspace_remove)

    # -- workspace -------------------------------------------------------

    def _repopulate_workspace_list(self) -> None:
        self.workspace_list.clear()
        for entry in self._config.workspace.apps:
            text = f"{entry.label}  —  {entry.kind}  —  {entry.target}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, entry.label)
            self.workspace_list.addItem(item)

    def _selected_workspace_index(self) -> int | None:
        row = self.workspace_list.currentRow()
        return row if row >= 0 else None

    def _on_workspace_add(self) -> None:
        dlg = _WorkspaceAppDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        entry = dlg.result_entry()
        if entry is None:
            return
        self._config.workspace.apps = [*self._config.workspace.apps, entry]
        self._repopulate_workspace_list()
        self._persist()

    def _on_workspace_edit(self) -> None:
        idx = self._selected_workspace_index()
        if idx is None:
            return
        existing = self._config.workspace.apps[idx]
        dlg = _WorkspaceAppDialog(self, existing=existing)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        entry = dlg.result_entry()
        if entry is None:
            return
        apps = list(self._config.workspace.apps)
        apps[idx] = entry
        self._config.workspace.apps = apps
        self._repopulate_workspace_list()
        self._persist()

    def _on_workspace_remove(self) -> None:
        idx = self._selected_workspace_index()
        if idx is None:
            return
        existing = self._config.workspace.apps[idx]
        confirm = QMessageBox.question(
            self,
            "Remove workspace app",
            f"Remove {existing.label!r} from your workspace?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        apps = list(self._config.workspace.apps)
        del apps[idx]
        self._config.workspace.apps = apps
        self._repopulate_workspace_list()
        self._persist()

    # -- slots -----------------------------------------------------------

    def _on_start_toggled(self, checked: bool) -> None:
        self._config.general.start_with_windows = checked
        self._persist()

    def _on_tray_toggled(self, checked: bool) -> None:
        self._config.general.minimize_to_tray = checked
        self._persist()

    def _on_overlay_toggled(self, checked: bool) -> None:
        self._config.general.show_overlay = checked
        self._persist()

    def _on_log_level_changed(self, text: str) -> None:
        if text not in _LOG_LEVELS:
            return  # combo can briefly be empty during initial fill
        self._config.general.log_level = text  # type: ignore[assignment]
        self._persist()

    def _on_auto_sleep_toggled(self, checked: bool) -> None:
        self._config.lifecycle.auto_sleep_enabled = checked
        self._persist()

    def _on_idle_timeout_changed(self, value: int) -> None:
        self._config.lifecycle.idle_timeout_minutes = value
        self._persist()

    def _on_weather_lat_changed(self, value: float) -> None:
        self._config.weather.latitude = value
        self._persist()

    def _on_weather_lon_changed(self, value: float) -> None:
        self._config.weather.longitude = value
        self._persist()

    def _on_detect_location(self) -> None:
        try:
            import httpx
            r = httpx.get("https://ipapi.co/json/", timeout=3.0)
            r.raise_for_status()
            data = r.json()
            lat = float(data["latitude"])
            lon = float(data["longitude"])
        except Exception as exc:
            log.warning("IP location detection failed: %s", exc)
            QMessageBox.warning(
                self, "Detection failed",
                f"Could not detect location from IP: {exc}",
            )
            return
        self.weather_lat.blockSignals(True)
        self.weather_lon.blockSignals(True)
        self.weather_lat.setValue(lat)
        self.weather_lon.setValue(lon)
        self.weather_lat.blockSignals(False)
        self.weather_lon.blockSignals(False)
        self._config.weather.latitude = lat
        self._config.weather.longitude = lon
        self._persist()

    def _persist(self) -> None:
        try:
            save_config(self._config)
        except Exception:
            log.exception("save_config failed from GeneralTab")
            return
        try:
            self._on_change()
        except Exception:
            log.exception("on_change callback raised from GeneralTab")


class _WorkspaceAppDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        existing: WorkspaceAppEntry | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Workspace app" if existing is None else "Edit workspace app")
        self._result: WorkspaceAppEntry | None = None

        self.label_edit = QLineEdit(existing.label if existing else "")
        self.kind_combo = QComboBox()
        for _kind, label in _WORKSPACE_KINDS:
            self.kind_combo.addItem(label, _kind)
        if existing:
            idx = self.kind_combo.findData(existing.kind)
            if idx >= 0:
                self.kind_combo.setCurrentIndex(idx)

        self.target_edit = QLineEdit(existing.target if existing else "")
        self.target_edit.setPlaceholderText(
            "e.g. Cursor, C:\\path\\app.exe, or shell:AppsFolder\\…"
        )

        form = QFormLayout()
        form.addRow("Label:", self.label_edit)
        form.addRow("Launch as:", self.kind_combo)
        form.addRow("Target:", self.target_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        label = self.label_edit.text().strip()
        target = self.target_edit.text().strip()
        kind = self.kind_combo.currentData()
        if not label or not target:
            QMessageBox.warning(self, "Invalid", "Label and target are required.")
            return
        if kind not in ("installed_app", "executable", "shell"):
            QMessageBox.warning(self, "Invalid", "Choose a launch type.")
            return
        self._result = WorkspaceAppEntry(
            label=label,
            kind=kind,  # type: ignore[arg-type]
            target=target,
        )
        self.accept()

    def result_entry(self) -> WorkspaceAppEntry | None:
        return self._result
