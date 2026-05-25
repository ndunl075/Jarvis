"""Test fixtures for UI tests.

Sets the Qt platform to "offscreen" BEFORE PySide6 is imported anywhere
in the test process so QObject construction works in headless CI without
a real display. The QApplication fixture is session-scoped because Qt
forbids constructing a second QApplication in the same process."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def _no_favicon_network(monkeypatch):
    """Block real network calls from _FaviconThread in every UI test.

    _FaviconThread.start is a no-op so favicon threads are created but never
    run.  This prevents test-to-test interference from lingering QThreads and
    avoids real HTTP calls to Google's favicon service in the test suite.
    """
    monkeypatch.setattr(
        "jarvis.ui.research_panel._FaviconThread.start",
        lambda self: None,
    )
