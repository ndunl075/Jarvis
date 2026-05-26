"""Models tab: STT model size + compute, LLM model + sampling + keep-alive.

System prompt is intentionally hidden from the UI — it is power-user
territory and editing it can break Jarvis's persona/tool-calling
behaviour. Power users can edit cfg.llm.system_prompt directly in the
JSON config file."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from jarvis.core.config import JarvisConfig, save_config

log = logging.getLogger(__name__)

_STT_SIZES: tuple[str, ...] = ("tiny", "base", "small")
_STT_COMPUTES: tuple[str, ...] = ("int8", "float16", "float32")


def _ollama_list_models() -> list[str]:
    """Shell out to `ollama list` and parse the model column. Returns []
    on any failure (daemon down, ollama not installed, weird output)."""
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.info("ollama list unavailable: %s", e)
        return []
    if proc.returncode != 0:
        log.info("ollama list returned %d: %s", proc.returncode, proc.stderr.strip())
        return []
    names: list[str] = []
    for line in proc.stdout.splitlines()[1:]:  # skip header row
        token = line.split(None, 1)[0].strip() if line.strip() else ""
        if token:
            names.append(token)
    return names


class ModelsTab(QWidget):
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

        # STT.
        self.stt_size = QComboBox()
        self.stt_size.addItems(_STT_SIZES)
        self.stt_size.setCurrentText(config.stt.model_size)
        self.stt_language = QLineEdit(config.stt.language)
        self.stt_compute = QComboBox()
        self.stt_compute.addItems(_STT_COMPUTES)
        self.stt_compute.setCurrentText(config.stt.compute_type)

        # LLM model dropdown + refresh.
        self.llm_model = QComboBox()
        self.llm_model.setEditable(True)
        refresh_models = QPushButton("Refresh")
        refresh_models.clicked.connect(self._refresh_models)
        model_row = QHBoxLayout()
        model_row.addWidget(self.llm_model, 1)
        model_row.addWidget(refresh_models)
        self._populate_models()

        # LLM sampling.
        self.llm_temperature = QSlider(Qt.Orientation.Horizontal)
        self.llm_temperature.setRange(0, 200)  # 0.00..2.00
        self.llm_temperature.setValue(int(round(config.llm.temperature * 100)))
        self.llm_temperature_label = QLabel(f"{config.llm.temperature:.2f}")
        temp_row = QHBoxLayout()
        temp_row.addWidget(self.llm_temperature)
        temp_row.addWidget(self.llm_temperature_label)

        self.llm_max_tokens = QSpinBox()
        self.llm_max_tokens.setRange(64, 8192)
        self.llm_max_tokens.setSingleStep(64)
        self.llm_max_tokens.setValue(config.llm.max_tokens)

        self.llm_keep_alive = QSpinBox()
        self.llm_keep_alive.setRange(60, 86400)
        self.llm_keep_alive.setSingleStep(60)
        self.llm_keep_alive.setValue(config.llm.keep_alive_seconds)
        self.llm_keep_alive.setSuffix(" s")

        form = QFormLayout()
        form.addRow("STT model size:", self.stt_size)
        form.addRow("STT language:", self.stt_language)
        form.addRow("STT compute:", self.stt_compute)
        form.addRow("LLM model:", model_row)
        form.addRow("LLM temperature:", temp_row)
        form.addRow("LLM max tokens:", self.llm_max_tokens)
        form.addRow("LLM keep-alive:", self.llm_keep_alive)

        # Vision: separate group because the model is unrelated to the
        # conversational LLM above (vision-capable models are pulled
        # separately and most users won't have one by default).
        vision_box = QGroupBox("Vision (see-screen tool)")
        vision_form = QFormLayout(vision_box)
        vision_hint = QLabel(
            "Used when you say \"see my screen\" / \"what's on my screen\". "
            "Requires a multimodal Ollama model — pull one with "
            "<code>ollama pull llava:7b</code>. Leave blank to disable."
        )
        vision_hint.setWordWrap(True)
        vision_hint.setTextFormat(Qt.TextFormat.RichText)
        vision_hint.setStyleSheet("color: #808080; font-size: 9pt;")
        vision_form.addRow(vision_hint)

        self.vision_model = QLineEdit(config.vision.model)
        self.vision_model.setPlaceholderText("e.g. llava:7b, moondream, minicpm-v")
        vision_form.addRow("Vision model:", self.vision_model)

        self.vision_max_dim = QSpinBox()
        self.vision_max_dim.setRange(320, 4096)
        self.vision_max_dim.setSingleStep(64)
        self.vision_max_dim.setValue(config.vision.max_image_dim)
        self.vision_max_dim.setSuffix(" px")
        self.vision_max_dim.setToolTip(
            "Long-edge pixel cap before the screenshot is sent. "
            "Lower = faster, less detail."
        )
        vision_form.addRow("Max image dimension:", self.vision_max_dim)

        self.vision_max_tokens = QSpinBox()
        self.vision_max_tokens.setRange(64, 4096)
        self.vision_max_tokens.setSingleStep(64)
        self.vision_max_tokens.setValue(config.vision.max_tokens)
        vision_form.addRow("Vision max tokens:", self.vision_max_tokens)

        self.vision_temperature = QSlider(Qt.Orientation.Horizontal)
        self.vision_temperature.setRange(0, 200)
        self.vision_temperature.setValue(int(round(config.vision.temperature * 100)))
        self.vision_temperature_label = QLabel(f"{config.vision.temperature:.2f}")
        v_temp_row = QHBoxLayout()
        v_temp_row.addWidget(self.vision_temperature)
        v_temp_row.addWidget(self.vision_temperature_label)
        vision_form.addRow("Vision temperature:", v_temp_row)

        self.vision_model.editingFinished.connect(self._on_vision_model)
        self.vision_max_dim.valueChanged.connect(self._on_vision_max_dim)
        self.vision_max_tokens.valueChanged.connect(self._on_vision_max_tokens)
        self.vision_temperature.valueChanged.connect(self._on_vision_temperature)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(vision_box)
        root.addStretch(1)

        self.stt_size.currentTextChanged.connect(self._on_stt_size)
        self.stt_language.editingFinished.connect(self._on_stt_language)
        self._stt_language_timer = QTimer(self)
        self._stt_language_timer.setSingleShot(True)
        self._stt_language_timer.setInterval(400)
        self._stt_language_timer.timeout.connect(self._on_stt_language)
        self.stt_language.textChanged.connect(self._stt_language_timer.start)
        self.stt_compute.currentTextChanged.connect(self._on_stt_compute)
        self.llm_model.currentTextChanged.connect(self._on_llm_model)
        self.llm_temperature.valueChanged.connect(self._on_llm_temperature)
        self.llm_max_tokens.valueChanged.connect(self._on_llm_max_tokens)
        self.llm_keep_alive.valueChanged.connect(self._on_llm_keep_alive)

    # -- population ------------------------------------------------------

    def _populate_models(self) -> None:
        self.llm_model.blockSignals(True)
        self.llm_model.clear()
        names = _ollama_list_models()
        if names:
            self.llm_model.addItems(names)
        current = self._config.llm.model
        idx = self.llm_model.findText(current)
        if idx >= 0:
            self.llm_model.setCurrentIndex(idx)
        else:
            # User's configured model isn't in the local pull list — keep
            # the configured value as the editable text so it's visible
            # and editable.
            self.llm_model.setEditText(current)
        self.llm_model.blockSignals(False)

    def _refresh_models(self) -> None:
        self._populate_models()

    # -- slots -----------------------------------------------------------

    def _on_stt_size(self, text: str) -> None:
        if text not in _STT_SIZES:
            return
        self._config.stt.model_size = text  # type: ignore[assignment]
        self._persist()

    def _on_stt_language(self) -> None:
        value = self.stt_language.text().strip() or "en"
        self._config.stt.language = value
        self._persist()

    def _on_stt_compute(self, text: str) -> None:
        if text not in _STT_COMPUTES:
            return
        self._config.stt.compute_type = text  # type: ignore[assignment]
        self._persist()

    def _on_llm_model(self, text: str) -> None:
        value = text.strip()
        if not value:
            return
        self._config.llm.model = value
        self._persist()

    def _on_llm_temperature(self, value: int) -> None:
        f = value / 100.0
        self._config.llm.temperature = f
        self.llm_temperature_label.setText(f"{f:.2f}")
        self._persist()

    def _on_llm_max_tokens(self, value: int) -> None:
        self._config.llm.max_tokens = value
        self._persist()

    def _on_llm_keep_alive(self, value: int) -> None:
        self._config.llm.keep_alive_seconds = value
        self._persist()

    def _on_vision_model(self) -> None:
        self._config.vision.model = self.vision_model.text().strip()
        self._persist()

    def _on_vision_max_dim(self, value: int) -> None:
        self._config.vision.max_image_dim = value
        self._persist()

    def _on_vision_max_tokens(self, value: int) -> None:
        self._config.vision.max_tokens = value
        self._persist()

    def _on_vision_temperature(self, value: int) -> None:
        f = value / 100.0
        self._config.vision.temperature = f
        self.vision_temperature_label.setText(f"{f:.2f}")
        self._persist()

    def _persist(self) -> None:
        try:
            save_config(self._config)
        except Exception:
            log.exception("save_config failed from ModelsTab")
            return
        try:
            self._on_change()
        except Exception:
            log.exception("on_change callback raised from ModelsTab")
