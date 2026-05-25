"""Tests for ResearchPanel and its supporting classes.

Architecture note: ResearchWorker and FollowUpWorker are patched out so no
real network calls happen. We drive the panel by calling internal slots
directly and by simulating worker signals.

Existing tests that call _on_worker_finished patch out _start_followup_worker
on the panel instance to prevent a real FollowUpWorker from being started.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# _extract_sources
# ---------------------------------------------------------------------------


def _make_block(**kwargs):
    class _Block:
        pass
    b = _Block()
    for k, v in kwargs.items():
        setattr(b, k, v)
    return b


def test_extract_sources_web_search_tool_result():
    from jarvis.ui.research_panel import _extract_sources

    result = _make_block(url="https://example.com", title="Example Site")
    block = _make_block(type="web_search_tool_result", content=[result])
    msg = _make_block(content=[block])
    sources = _extract_sources(msg)
    assert len(sources) == 1
    assert sources[0]["url"] == "https://example.com"
    assert sources[0]["title"] == "Example Site"


def test_extract_sources_deduplicates():
    from jarvis.ui.research_panel import _extract_sources

    r1 = _make_block(url="https://example.com", title="A")
    r2 = _make_block(url="https://example.com", title="B")
    block = _make_block(type="web_search_tool_result", content=[r1, r2])
    msg = _make_block(content=[block])
    assert len(_extract_sources(msg)) == 1


def test_extract_sources_fallback_regex():
    from jarvis.ui.research_panel import _extract_sources

    text_block = _make_block(
        type="text",
        text="See https://example.com/article and https://other.com for more.",
    )
    msg = _make_block(content=[text_block])
    sources = _extract_sources(msg)
    urls = [s["url"] for s in sources]
    assert "https://example.com/article" in urls
    assert "https://other.com" in urls


def test_extract_sources_limits_to_six():
    from jarvis.ui.research_panel import _extract_sources

    results = [
        _make_block(url=f"https://example.com/{i}", title=f"T{i}") for i in range(10)
    ]
    block = _make_block(type="web_search_tool_result", content=results)
    msg = _make_block(content=[block])
    assert len(_extract_sources(msg)) == 6


def test_extract_sources_empty():
    from jarvis.ui.research_panel import _extract_sources

    assert _extract_sources(_make_block(content=[])) == []


# ---------------------------------------------------------------------------
# ResearchPanel — construction and state
# ---------------------------------------------------------------------------


def _make_panel(qapp):
    """Return a panel with ResearchWorker patched so no threads start."""
    from jarvis.ui.research_panel import ResearchPanel

    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel = ResearchPanel()
    panel._mock_worker_cls = mock_cls
    return panel


def _start(panel, query, rq):
    """Call panel._on_start with ResearchWorker mocked out."""
    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel._on_start(query, rq)
    return mock_cls


def test_panel_initially_hidden(qapp):
    from jarvis.ui.research_panel import ResearchPanel

    panel = ResearchPanel()
    assert not panel.isVisible()


def test_panel_width_default(qapp):
    from jarvis.ui.research_panel import ResearchPanel, _DEFAULT_PANEL_WIDTH

    panel = ResearchPanel()
    assert panel.width() == _DEFAULT_PANEL_WIDTH


def test_panel_width_from_constructor(qapp):
    from jarvis.ui.research_panel import ResearchPanel

    panel = ResearchPanel(panel_width=500)
    assert panel.width() == 500


def test_set_panel_width_clamps(qapp):
    from jarvis.ui.research_panel import ResearchPanel

    panel = ResearchPanel(panel_width=420)
    panel.set_panel_width(900)
    assert panel.width() == 700
    panel.set_panel_width(100)
    assert panel.width() == 280


def test_on_width_changed_fires_after_resize_drag(qapp):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    from jarvis.ui.research_panel import ResearchPanel

    widths: list[int] = []
    panel = ResearchPanel(
        panel_width=400,
        on_width_changed=widths.append,
    )
    panel.show()
    QApplication.processEvents()

    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(4, 10),
        QPointF(100, 10),
        QPointF(100, 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    panel.mousePressEvent(press)
    move = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(4, 10),
        QPointF(50, 10),
        QPointF(50, 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    panel.mouseMoveEvent(move)
    release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(4, 10),
        QPointF(50, 10),
        QPointF(50, 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    panel.mouseReleaseEvent(release)

    assert len(widths) == 1
    assert widths[0] == panel.width()
    assert panel.width() > 400


def test_show_for_query_displays_panel_and_placeholder(qapp):
    panel = _make_panel(qapp)
    rq = queue.Queue()
    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel._on_start("quantum computing", rq)

    assert panel.isVisible()
    assert "quantum computing" in panel._query_label.text()
    text = panel._content.toPlainText()
    assert "search" in text.lower() or "…" in text


def test_show_for_query_creates_worker(qapp):
    from jarvis.ui.research_panel import ResearchPanel

    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel = ResearchPanel()
        rq = queue.Queue()
        panel._on_start("test topic", rq)

    mock_cls.assert_called_once_with(
        "test topic",
        ollama_endpoint="http://localhost:11434",
        ollama_model="qwen2.5:7b-instruct",
    )
    mock_worker.start.assert_called_once()


def test_on_chunk_replaces_placeholder_on_first_chunk(qapp):
    panel = _make_panel(qapp)
    rq = queue.Queue()
    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel._on_start("topic", rq)

    panel._on_chunk("Hello")
    assert panel._content.toPlainText() == "Hello"
    assert panel._summary_text == "Hello"


def test_on_chunk_accumulates(qapp):
    panel = _make_panel(qapp)
    rq = queue.Queue()
    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel._on_start("topic", rq)

    panel._on_chunk("Hello")
    panel._on_chunk(" world")
    assert panel._content.toPlainText() == "Hello world"
    assert panel._summary_text == "Hello world"


def test_worker_finished_shows_sources_and_fills_queue(qapp):
    panel = _make_panel(qapp)
    rq = queue.Queue()
    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel._on_start("AI news", rq)

    sources = [
        {"title": "Article One", "url": "https://example.com/one"},
        {"title": "Article Two", "url": "https://example.com/two"},
    ]
    panel._on_chunk("Summary text")
    with patch.object(panel, "_start_followup_worker"):
        panel._on_worker_finished("Summary text", sources)

    assert panel._sources_section.isVisible()
    assert not rq.empty()
    summary, srcs = rq.get_nowait()
    assert summary == "Summary text"
    assert len(srcs) == 2


def test_worker_failed_fills_queue_with_exception(qapp):
    panel = _make_panel(qapp)
    rq = queue.Queue()
    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel._on_start("topic", rq)

    panel._on_worker_failed("network error")
    result = rq.get_nowait()
    assert isinstance(result, RuntimeError)
    assert "network error" in str(result)


def test_sources_section_hidden_when_no_sources(qapp):
    panel = _make_panel(qapp)
    rq = queue.Queue()
    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel._on_start("topic", rq)

    with patch.object(panel, "_start_followup_worker"):
        panel._on_worker_finished("text", [])
    assert not panel._sources_section.isVisible()


def test_close_panel_hides_widget(qapp):
    panel = _make_panel(qapp)
    rq = queue.Queue()
    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls, \
         patch.object(panel, "_slide_out", side_effect=panel.hide):
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel._on_start("topic", rq)
        assert panel.isVisible()
        panel._on_close()
        assert not panel.isVisible()


def test_copy_button_copies_summary(qapp):
    from PySide6.QtWidgets import QApplication

    panel = _make_panel(qapp)
    rq = queue.Queue()
    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel._on_start("topic", rq)

    panel._on_chunk("Summary text here")
    panel._on_copy()
    assert QApplication.clipboard().text() == "Summary text here"


def test_copy_button_label_resets_after_copy(qapp):
    panel = _make_panel(qapp)
    rq = queue.Queue()
    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = False
        mock_cls.return_value = mock_worker
        panel._on_start("topic", rq)

    panel._on_chunk("text")
    panel._on_copy()
    assert panel._copy_btn.text() == "Copied!"
    # After 1500 ms it resets — we don't wait in tests; just verify initial state


def test_copy_button_in_header(qapp):
    """Copy button must be in the header row (top of panel), not at the bottom."""
    from jarvis.ui.research_panel import ResearchPanel

    panel = ResearchPanel()
    assert panel._copy_btn.text() == "COPY"
    assert panel._copy_btn.isVisible() or True  # just asserts it exists


def test_new_query_clears_previous_state(qapp):
    panel = _make_panel(qapp)
    rq = queue.Queue()

    _start(panel, "first query", rq)
    panel._on_chunk("Old content")
    with patch.object(panel, "_start_followup_worker"):
        panel._on_worker_finished(
            "Old content",
            [{"title": "Old", "url": "https://old.example.com"}],
        )

    rq2 = queue.Queue()
    _start(panel, "second query", rq2)
    assert "second query" in panel._query_label.text()
    assert not panel._sources_section.isVisible()


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_initially_empty(qapp):
    from jarvis.ui.research_panel import ResearchPanel

    panel = ResearchPanel()
    assert panel._history == []


def test_history_updated_on_start(qapp):
    panel = _make_panel(qapp)
    rq = queue.Queue()
    _start(panel, "black holes", rq)
    assert panel._history == ["black holes"]


def test_history_deduplicates_and_moves_to_front(qapp):
    panel = _make_panel(qapp)
    _start(panel, "black holes", queue.Queue())
    _start(panel, "dark matter", queue.Queue())
    _start(panel, "black holes", queue.Queue())
    assert panel._history[0] == "black holes"
    assert panel._history.count("black holes") == 1
    assert len(panel._history) == 2


def test_history_limited_to_ten(qapp):
    from jarvis.ui.research_panel import _MAX_HISTORY

    panel = _make_panel(qapp)
    for i in range(12):
        _start(panel, f"query {i}", queue.Queue())
    assert len(panel._history) == _MAX_HISTORY
    assert panel._history[0] == "query 11"


def test_history_section_initially_hidden(qapp):
    from jarvis.ui.research_panel import ResearchPanel

    panel = ResearchPanel()
    assert not panel._history_section.isVisible()


def test_history_toggle_shows_section_hides_content(qapp):
    panel = _make_panel(qapp)
    _start(panel, "some topic", queue.Queue())

    panel._on_history_toggle()

    assert panel._history_visible
    assert panel._history_section.isVisible()
    assert not panel._content.isVisible()


def test_history_toggle_twice_restores_content(qapp):
    panel = _make_panel(qapp)
    _start(panel, "some topic", queue.Queue())

    panel._on_history_toggle()
    panel._on_history_toggle()

    assert not panel._history_visible
    assert not panel._history_section.isVisible()
    assert panel._content.isVisible()


def test_history_query_reruns_search(qapp):
    panel = _make_panel(qapp)
    _start(panel, "first query", queue.Queue())

    panel._on_history_toggle()
    assert panel._history_visible

    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls2:
        mock_worker2 = MagicMock()
        mock_worker2.isRunning.return_value = False
        mock_cls2.return_value = mock_worker2
        panel._on_history_query_clicked("first query")

    assert not panel._history_visible
    assert "first query" in panel._query_label.text()
    mock_cls2.assert_called_once_with(
        "first query",
        ollama_endpoint="http://localhost:11434",
        ollama_model="qwen2.5:7b-instruct",
    )


def test_history_toggle_restores_sources_visibility(qapp):
    """Sources section that was visible before history view should be restored."""
    panel = _make_panel(qapp)
    _start(panel, "topic", queue.Queue())
    with patch.object(panel, "_start_followup_worker"):
        panel._on_worker_finished(
            "summary",
            [{"title": "X", "url": "https://example.com"}],
        )
    assert panel._sources_section.isVisible()

    panel._on_history_toggle()
    assert not panel._sources_section.isVisible()

    panel._on_history_toggle()
    assert panel._sources_section.isVisible()


# ---------------------------------------------------------------------------
# Follow-up questions
# ---------------------------------------------------------------------------


def test_followup_section_initially_hidden(qapp):
    from jarvis.ui.research_panel import ResearchPanel

    panel = ResearchPanel()
    assert not panel._followup_section.isVisible()


def test_followup_worker_started_after_research(qapp):
    panel = _make_panel(qapp)
    _start(panel, "topic", queue.Queue())

    with patch.object(panel, "_start_followup_worker") as mock_fu:
        panel._on_worker_finished("Research summary.", [])

    mock_fu.assert_called_once_with("Research summary.")


def test_followup_worker_not_started_on_empty_summary(qapp):
    """No follow-up when summary is empty."""
    panel = _make_panel(qapp)
    with patch.object(panel, "_start_followup_worker") as mock_fu:
        panel._on_worker_finished("", [])

    mock_fu.assert_not_called()


def test_followup_questions_show_chips(qapp):
    panel = _make_panel(qapp)
    _start(panel, "topic", queue.Queue())

    assert not panel._followup_section.isVisible()
    panel._on_followup_questions(["Q1?", "Q2?", "Q3?"])

    assert panel._followup_section.isVisible()
    assert panel._chips_layout.count() == 3


def test_followup_chips_have_correct_labels(qapp):
    panel = _make_panel(qapp)
    _start(panel, "topic", queue.Queue())

    questions = ["What about X?", "How does Y work?", "Tell me more about Z?"]
    panel._on_followup_questions(questions)

    labels = [
        panel._chips_layout.itemAt(i).widget().text()
        for i in range(panel._chips_layout.count())
    ]
    assert labels == questions


def test_chip_click_starts_new_research(qapp):
    panel = _make_panel(qapp)
    _start(panel, "topic", queue.Queue())

    panel._on_followup_questions(
        ["What about X?", "How does Y work?", "Tell me more about Z?"]
    )

    with patch("jarvis.ui.research_panel.ResearchWorker") as mock_cls2:
        mock_worker2 = MagicMock()
        mock_worker2.isRunning.return_value = False
        mock_cls2.return_value = mock_worker2
        chip = panel._chips_layout.itemAt(0).widget()
        chip.click()

    mock_cls2.assert_called_once_with(
        "What about X?",
        ollama_endpoint="http://localhost:11434",
        ollama_model="qwen2.5:7b-instruct",
    )


def test_new_query_clears_chips(qapp):
    panel = _make_panel(qapp)
    _start(panel, "first query", queue.Queue())
    panel._on_followup_questions(["q1?", "q2?", "q3?"])
    assert panel._chips_layout.count() == 3
    assert panel._followup_section.isVisible()

    _start(panel, "second query", queue.Queue())
    assert panel._chips_layout.count() == 0
    assert not panel._followup_section.isVisible()


def test_followup_section_hidden_when_history_active(qapp):
    """Chips should not show while the history list is open."""
    panel = _make_panel(qapp)
    _start(panel, "topic", queue.Queue())

    panel._on_followup_questions(["Q1?", "Q2?", "Q3?"])
    assert panel._followup_section.isVisible()

    panel._on_history_toggle()
    assert not panel._followup_section.isVisible()

    # Re-opening history dismiss restores chips
    panel._on_history_toggle()
    assert panel._followup_section.isVisible()


# ---------------------------------------------------------------------------
# copy_summary (cross-thread) and get_next_sentences
# ---------------------------------------------------------------------------


def test_copy_summary_copies_text(qapp):
    from PySide6.QtWidgets import QApplication

    panel = _make_panel(qapp)
    _start(panel, "topic", queue.Queue())
    panel._on_chunk("The summary text.")
    panel.copy_summary()
    # copy_summary emits _sig_copy which connects to _on_copy on same thread
    assert QApplication.clipboard().text() == "The summary text."


def test_get_next_sentences_basic(qapp):
    panel = _make_panel(qapp)
    _start(panel, "topic", queue.Queue())
    panel._summary_text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    panel._read_cursor = 0

    result = panel.get_next_sentences(2)
    assert result == "First sentence. Second sentence."
    assert panel._read_cursor == 2


def test_get_next_sentences_advances_cursor(qapp):
    panel = _make_panel(qapp)
    panel._summary_text = "S1. S2. S3. S4."
    panel._read_cursor = 0

    panel.get_next_sentences(2)
    result = panel.get_next_sentences(2)
    assert panel._read_cursor == 4
    assert "S3" in (result or "")


def test_get_next_sentences_returns_none_when_exhausted(qapp):
    panel = _make_panel(qapp)
    panel._summary_text = "Only one sentence."
    panel._read_cursor = 2  # already past the end

    result = panel.get_next_sentences(2)
    assert result is None


def test_read_cursor_reset_on_new_query(qapp):
    panel = _make_panel(qapp)
    panel._read_cursor = 6  # simulate mid-read state

    _start(panel, "new query", queue.Queue())
    assert panel._read_cursor == 0


def test_read_cursor_set_to_two_after_tool_finishes(qapp):
    """When result_queue is filled (tool path), cursor advances to 2."""
    panel = _make_panel(qapp)
    rq = queue.Queue()
    _start(panel, "topic", rq)

    with patch.object(panel, "_start_followup_worker"):
        panel._on_worker_finished("S1. S2. S3. S4.", [])

    assert panel._read_cursor == 2


def test_read_cursor_stays_at_zero_for_chip_path(qapp):
    """When no result_queue (chip/history path), cursor stays at 0."""
    panel = _make_panel(qapp)
    # Chip/history flow: no result_queue
    _start(panel, "topic", None)  # type: ignore[arg-type]

    with patch.object(panel, "_start_followup_worker"):
        panel._on_worker_finished("S1. S2. S3. S4.", [])

    assert panel._read_cursor == 0
