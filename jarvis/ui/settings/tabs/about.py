"""About tab: version, GitHub link, Ollama status, default-model status,
'check for updates' stub. Read-only — does not edit config."""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jarvis.core.config import JarvisConfig

log = logging.getLogger(__name__)

# Hardcoded so the About tab doesn't pull in importlib.metadata at
# UI-construction time. Update on each release.
JARVIS_VERSION: str = "0.1.0-dev"
GITHUB_URL: str = "https://github.com/ndunl075/Jarvis"

_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
_OLLAMA_PROBE_TIMEOUT = 1.5


def _probe_ollama() -> tuple[bool, list[str]]:
    """Return (connected, model_names). On any failure connected=False
    and model_names=[]. Short timeout so opening the About tab can't
    block the UI when the daemon is down."""
    try:
        resp = httpx.get(
            f"{_OLLAMA_ENDPOINT}/api/tags",
            timeout=_OLLAMA_PROBE_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception as e:
        log.info("ollama probe failed: %s", e)
        return False, []
    models = body.get("models") if isinstance(body, dict) else None
    if not isinstance(models, list):
        return True, []
    names: list[str] = []
    for m in models:
        if isinstance(m, dict):
            n = m.get("name")
            if isinstance(n, str):
                names.append(n)
    return True, names


class AboutTab(QWidget):
    def __init__(
        self,
        *,
        config: JarvisConfig,
        on_change: Callable[[], None],
        probe: Callable[[], tuple[bool, list[str]]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        # AboutTab doesn't mutate config, but accepts the callback for
        # signature symmetry with the other tabs. The check-for-updates
        # stub fires it so the composition root can log/observe.
        self._on_change = on_change
        self._probe = probe or _probe_ollama

        version_label = QLabel(f"Jarvis {JARVIS_VERSION}")

        github_link = QLabel(
            f'<a href="{GITHUB_URL}">{GITHUB_URL}</a>'
        )
        github_link.setOpenExternalLinks(True)
        github_link.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.ollama_status = QLabel("Probing…")
        self.model_status = QLabel("Probing…")
        self._refresh_status()

        check_updates = QPushButton("Check for updates")
        check_updates.clicked.connect(self._on_check_updates)

        form = QFormLayout()
        form.addRow("Version:", version_label)
        form.addRow("Repository:", github_link)
        form.addRow("Ollama:", self.ollama_status)
        form.addRow(f"Model {config.llm.model!r}:", self.model_status)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(check_updates)
        root.addStretch(1)

    def _refresh_status(self) -> None:
        connected, models = self._probe()
        if connected:
            self.ollama_status.setText("Connected")
        else:
            self.ollama_status.setText("Not detected")
        target = self._config.llm.model
        # `ollama list` returns names that often include a tag suffix
        # ('qwen2.5:7b-instruct'); match exact then prefix as a fallback
        # so 'qwen2.5:7b' configured against a pulled 'qwen2.5:7b-instruct'
        # doesn't flip-flop on cosmetic suffix differences.
        if not connected:
            self.model_status.setText("Unknown (Ollama not detected)")
        elif target in models:
            self.model_status.setText("Pulled")
        elif any(m.startswith(target + ":") or target.startswith(m + ":")
                 for m in models):
            self.model_status.setText("Pulled (different tag)")
        else:
            self.model_status.setText("Not pulled")

    def _on_check_updates(self) -> None:
        # Stub for Phase 5; real update-check lives in Phase 7 polish.
        log.info("check-for-updates clicked (no real check wired yet)")
        try:
            self._on_change()
        except Exception:
            log.exception("on_change callback raised from AboutTab")
