"""Tests for the Qt tool-confirmation prompt.

Two layers:

  - ToolConfirmationDialog on its own — approve is the only path to
    Accepted; Escape, the close button and the countdown all reject.
  - QtToolConfirmer across a real thread boundary — an asyncio loop
    running on its own thread (standing in for the audio thread) awaits
    a verdict produced by the Qt main thread, which is the actual hop
    jarvis/app.py has to survive.

The second layer is why this file exists. Everything else about the gate
is testable without Qt (tests/tools/test_tool_confirmation.py); the hop
is not, and the hop is where a deadlock would live.

The pattern below — run the loop on a background thread, drive Qt by
pumping processEvents() on the main thread — mirrors production exactly:
there, Qt's own exec() does the pumping. Every wait is bounded, so a
regression that reintroduces a deadlock fails the test in seconds
instead of hanging CI.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from jarvis.app import _AUDIO_SHUTDOWN_TIMEOUT
from jarvis.core.config import ToolsConfig
from jarvis.tools.registry import ConfirmationRequest, ToolRegistry, ToolResult
from jarvis.ui.tool_confirm import (
    _TIMEOUT_SECONDS,
    _WATCHDOG_GRACE_SECONDS,
    QtToolConfirmer,
    ToolConfirmationDialog,
    format_arguments,
)

# --- helpers ------------------------------------------------------------


def _pump(qapp, predicate, timeout: float = 5.0) -> bool:
    """Spin the Qt event loop on this thread until `predicate` holds.

    Bounded: every use of this is a claim that something resolves
    *promptly*, so the timeout failing is the test failing."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(predicate())


def _request(name: str = "type_into_active_window") -> ConfirmationRequest:
    return ConfirmationRequest(
        tool_name=name,
        summary="Type 8 character(s) into the focused window.",
        arguments={"text": "rm -rf /", "interval_seconds": 0.0},
    )


@pytest.fixture
def audio_loop():
    """An asyncio loop on its own thread — the audio thread's stand-in."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="test-audio", daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "test audio loop thread did not stop"
    loop.close()


@pytest.fixture
def confirmer(qapp):
    """A confirmer with production timeouts; individual tests override."""
    c = QtToolConfirmer()
    yield c
    c.close()
    qapp.processEvents()


# --- the timeout constants ----------------------------------------------


def test_watchdog_expires_before_the_shutdown_join():
    """The invariant that makes shutdown-during-prompt safe by arithmetic
    rather than by hope.

    _on_quit() joins the audio thread for _AUDIO_SHUTDOWN_TIMEOUT. If the
    audio-side await could outlast that, a prompt open at quit would turn
    into a 10 s stall and a "audio thread did not stop" warning. It
    cannot: the await is bounded by dialog timeout + watchdog grace, and
    that sum is strictly smaller."""
    assert _TIMEOUT_SECONDS + _WATCHDOG_GRACE_SECONDS < _AUDIO_SHUTDOWN_TIMEOUT


def test_timeout_is_long_enough_to_read_the_prompt():
    """A gate nobody can answer in time is a gate that only ever denies,
    which is indistinguishable from not shipping the feature."""
    assert 5.0 <= _TIMEOUT_SECONDS <= 10.0


# --- argument formatting -------------------------------------------------


def test_format_arguments_is_stable_and_readable():
    text = format_arguments({"b": 2, "a": "x"})
    assert text.index('"a"') < text.index('"b"')


def test_format_arguments_survives_unserialisable_values():
    text = format_arguments({"when": object()})
    assert "when" in text


def test_format_arguments_truncates_a_blob():
    text = format_arguments({"text": "x" * 9000})
    assert len(text) < 2000
    assert "truncated" in text


# --- the dialog ----------------------------------------------------------


def _dialog(qapp, **kwargs) -> ToolConfirmationDialog:
    dlg = ToolConfirmationDialog(
        tool_name="type_into_active_window",
        summary="Type 8 character(s) into the focused window.",
        arguments_text='{"text": "rm -rf /"}',
        **kwargs,
    )
    return dlg


def test_dialog_shows_the_name_summary_and_arguments(qapp):
    dlg = _dialog(qapp)
    try:
        assert "focused window" in dlg.summary_label.text()
        assert "rm -rf /" in dlg.arguments_view.toPlainText()
    finally:
        dlg.reject()


def test_dialog_approve_accepts(qapp):
    dlg = _dialog(qapp)
    dlg.approve_button.click()
    assert dlg.result() == QDialog.DialogCode.Accepted


def test_dialog_deny_rejects(qapp):
    dlg = _dialog(qapp)
    dlg.deny_button.click()
    assert dlg.result() == QDialog.DialogCode.Rejected


def test_dialog_close_rejects(qapp):
    """Dismissing the window is an answer, and the answer is no."""
    dlg = _dialog(qapp)
    dlg.show()
    qapp.processEvents()
    dlg.close()
    assert dlg.result() == QDialog.DialogCode.Rejected


def test_dialog_defaults_to_deny_so_a_stray_return_is_safe(qapp):
    dlg = _dialog(qapp)
    try:
        assert dlg.deny_button.isDefault()
        assert not dlg.approve_button.isDefault()
    finally:
        dlg.reject()


def test_dialog_is_application_modal(qapp):
    dlg = _dialog(qapp)
    try:
        assert dlg.windowModality() == Qt.WindowModality.ApplicationModal
    finally:
        dlg.reject()


def test_dialog_times_out_into_a_rejection(qapp):
    """The default-deny timeout, at the widget level."""
    dlg = _dialog(qapp, timeout_seconds=0.05)
    dlg.show()
    assert _pump(qapp, lambda: not dlg.isVisible(), timeout=3.0)
    assert dlg.result() == QDialog.DialogCode.Rejected


def test_dialog_countdown_counts_down(qapp):
    dlg = _dialog(qapp, timeout_seconds=8.0)
    try:
        first = dlg.remaining_seconds
        assert _pump(qapp, lambda: dlg.remaining_seconds < first, timeout=3.0)
        assert "Denied automatically" in dlg.countdown_label.text()
    finally:
        dlg.reject()


# --- the cross-thread hop ------------------------------------------------


def test_approval_crosses_back_to_the_audio_loop(qapp, audio_loop, confirmer):
    """The whole mechanism, end to end.

    confirm() is awaited on a loop running on another thread; the dialog
    is built and clicked on this one; the verdict arrives back on the
    loop. Nothing here blocks either side — the loop thread is free the
    entire time the dialog is up, which is the property that keeps the
    audio pipeline alive during a prompt."""
    pending = asyncio.run_coroutine_threadsafe(
        confirmer.confirm(_request()), audio_loop
    )

    assert _pump(qapp, lambda: confirmer._dialog is not None)
    assert confirmer._dialog.arguments_view.toPlainText().count("rm -rf /") == 1
    confirmer._dialog.approve_button.click()

    assert _pump(qapp, pending.done)
    assert pending.result(timeout=1.0) is True


def test_denial_crosses_back_to_the_audio_loop(qapp, audio_loop, confirmer):
    pending = asyncio.run_coroutine_threadsafe(
        confirmer.confirm(_request()), audio_loop
    )

    assert _pump(qapp, lambda: confirmer._dialog is not None)
    dialog = confirmer._dialog
    assert dialog is not None
    dialog.deny_button.click()

    assert _pump(qapp, pending.done)
    assert pending.result(timeout=1.0) is False


def test_timeout_crosses_back_as_a_denial(qapp, audio_loop):
    """Default-deny on timeout, all the way through to the awaiting
    coroutine."""
    confirmer = QtToolConfirmer(timeout_seconds=0.1, watchdog_grace_seconds=2.0)
    try:
        pending = asyncio.run_coroutine_threadsafe(
            confirmer.confirm(_request()), audio_loop
        )
        assert _pump(qapp, pending.done, timeout=5.0)
        assert pending.result(timeout=1.0) is False
    finally:
        confirmer.close()
        qapp.processEvents()


def test_watchdog_denies_when_qt_never_answers(qapp, audio_loop):
    """The Qt thread is not required to be healthy for the audio loop to
    make progress.

    Nothing pumps the Qt event loop here, so _show_prompt is never even
    delivered — the situation during teardown, or before exec() starts.
    The await still resolves, and it resolves to a denial."""
    confirmer = QtToolConfirmer(timeout_seconds=0.05, watchdog_grace_seconds=0.05)
    try:
        pending = asyncio.run_coroutine_threadsafe(
            confirmer.confirm(_request()), audio_loop
        )
        assert pending.result(timeout=5.0) is False
        # And the abandoned prompt does not sneak a dialog up afterwards.
        qapp.processEvents()
        assert confirmer._dialog is None
    finally:
        confirmer.close()
        qapp.processEvents()


def test_a_second_request_while_one_is_open_is_denied(qapp, audio_loop, confirmer):
    """Denied, not queued. A queued prompt would be answered several
    seconds after the utterance that caused it, with its own timeout
    already ticking behind the first dialog — an approval aimed at the
    wrong question."""
    first = asyncio.run_coroutine_threadsafe(
        confirmer.confirm(_request("first")), audio_loop
    )
    assert _pump(qapp, lambda: confirmer._dialog is not None)

    second = asyncio.run_coroutine_threadsafe(
        confirmer.confirm(_request("second")), audio_loop
    )
    assert second.result(timeout=5.0) is False

    # The first prompt is untouched and still answerable.
    assert confirmer._dialog is not None
    confirmer._dialog.approve_button.click()
    assert _pump(qapp, first.done)
    assert first.result(timeout=1.0) is True


def test_requests_are_serialisable_again_after_one_completes(
    qapp, audio_loop, confirmer
):
    for expected, click in (("approve", True), ("deny", False)):
        pending = asyncio.run_coroutine_threadsafe(
            confirmer.confirm(_request(expected)), audio_loop
        )
        assert _pump(qapp, lambda: confirmer._dialog is not None)
        button = (
            confirmer._dialog.approve_button
            if click
            else confirmer._dialog.deny_button
        )
        button.click()
        assert _pump(qapp, pending.done)
        assert pending.result(timeout=1.0) is click


# --- shutdown during a prompt --------------------------------------------


def test_close_during_a_prompt_resolves_immediately(qapp, audio_loop):
    """Shutdown mid-prompt cannot hang the app.

    This is the case _on_quit() has to survive: a dialog is open, nobody
    answers it, and the app is quitting. close() runs on the Qt thread
    before the audio loop is even signalled, and the awaiting coroutine
    resolves to a denial straight away — well inside the 10 s join, and
    without depending on the 8 s countdown or the watchdog."""
    confirmer = QtToolConfirmer(timeout_seconds=300.0, watchdog_grace_seconds=300.0)
    try:
        pending = asyncio.run_coroutine_threadsafe(
            confirmer.confirm(_request()), audio_loop
        )
        assert _pump(qapp, lambda: confirmer._dialog is not None)

        started = time.monotonic()
        confirmer.close()

        assert pending.result(timeout=5.0) is False
        assert time.monotonic() - started < 2.0
        qapp.processEvents()
        assert confirmer._dialog is None
    finally:
        confirmer.close()
        qapp.processEvents()


def test_close_makes_later_requests_deny_without_a_dialog(qapp, audio_loop):
    """After quit has begun, a tool call still in flight on the audio
    thread must not raise a dialog onto a UI that is being destroyed."""
    confirmer = QtToolConfirmer(timeout_seconds=300.0, watchdog_grace_seconds=300.0)
    confirmer.close()

    pending = asyncio.run_coroutine_threadsafe(
        confirmer.confirm(_request()), audio_loop
    )
    assert pending.result(timeout=5.0) is False
    qapp.processEvents()
    assert confirmer._dialog is None


def test_close_is_idempotent(qapp):
    confirmer = QtToolConfirmer()
    confirmer.close()
    confirmer.close()


def test_cancelling_the_await_dismisses_the_dialog(qapp, audio_loop, confirmer):
    """A cancelled interaction takes its prompt down with it rather than
    leaving an orphan dialog whose answer nobody is waiting for."""
    pending = asyncio.run_coroutine_threadsafe(
        confirmer.confirm(_request()), audio_loop
    )
    assert _pump(qapp, lambda: confirmer._dialog is not None)

    pending.cancel()
    assert _pump(qapp, lambda: confirmer._dialog is None, timeout=5.0)


def test_a_late_verdict_cannot_answer_a_newer_request(qapp, audio_loop):
    """Token check.

    The abandoned prompt's dialog is still alive for an instant after the
    audio side gives up on it; when it finally closes, its verdict must
    not be applied to whatever request came next."""
    confirmer = QtToolConfirmer(timeout_seconds=0.05, watchdog_grace_seconds=0.05)
    try:
        abandoned = asyncio.run_coroutine_threadsafe(
            confirmer.confirm(_request("abandoned")), audio_loop
        )
        assert abandoned.result(timeout=5.0) is False

        # A fresh request, while the first token may still be in flight.
        confirmer._timeout_seconds = 300.0
        confirmer._watchdog_seconds = 300.0
        pending = asyncio.run_coroutine_threadsafe(
            confirmer.confirm(_request("fresh")), audio_loop
        )
        assert _pump(qapp, lambda: confirmer._dialog is not None)
        # The stale verdict, replayed. It names a token nobody holds.
        confirmer._resolve(1, approved=True)
        qapp.processEvents()
        assert not pending.done()

        dialog = confirmer._dialog
        assert dialog is not None
        dialog.deny_button.click()
        assert _pump(qapp, pending.done)
        assert pending.result(timeout=1.0) is False
    finally:
        confirmer.close()
        qapp.processEvents()


# --- registry + confirmer together ---------------------------------------


def _register(reg: ToolRegistry, tool) -> None:
    """See tests/tools/test_tool_confirmation.py — pyright rejects
    duck-typed tools over `args_schema` invariance."""
    reg.register(tool)  # type: ignore[arg-type]


def _spy_tool():
    from pydantic import BaseModel

    class _Args(BaseModel):
        text: str

    class _Spy:
        name = "type_into_active_window"
        description = "Types text into the focused window."
        args_schema = _Args
        requires_confirmation = True

        def __init__(self) -> None:
            self.typed: list[str] = []

        async def execute(self, args) -> ToolResult:
            self.typed.append(args.text)
            return ToolResult(success=True, output="Typed, sir.")

    return _Spy()


def test_registry_runs_the_tool_only_after_the_dialog_is_approved(
    qapp, audio_loop, confirmer
):
    """The composed path: registry.execute() on the audio loop, dialog on
    the Qt thread, tool dispatched only once Approve is clicked."""
    # enabled={} rather than the default: ToolsConfig disables
    # type_into_active_window out of the box, and a disabled tool is
    # dropped before the gate is ever reached.
    registry = ToolRegistry(ToolsConfig(enabled={}), confirmer=confirmer)
    tool = _spy_tool()
    _register(registry, tool)

    pending = asyncio.run_coroutine_threadsafe(
        registry.execute("type_into_active_window", {"text": "hello"}),
        audio_loop,
    )
    assert _pump(qapp, lambda: confirmer._dialog is not None)
    assert tool.typed == [], "the tool ran before the user answered"

    confirmer._dialog.approve_button.click()
    assert _pump(qapp, pending.done)

    assert pending.result(timeout=1.0) == ToolResult(
        success=True, output="Typed, sir."
    )
    assert tool.typed == ["hello"]


def test_registry_refuses_the_tool_when_the_dialog_is_denied(
    qapp, audio_loop, confirmer
):
    # enabled={} rather than the default: ToolsConfig disables
    # type_into_active_window out of the box, and a disabled tool is
    # dropped before the gate is ever reached.
    registry = ToolRegistry(ToolsConfig(enabled={}), confirmer=confirmer)
    tool = _spy_tool()
    _register(registry, tool)

    pending = asyncio.run_coroutine_threadsafe(
        registry.execute("type_into_active_window", {"text": "hello"}),
        audio_loop,
    )
    assert _pump(qapp, lambda: confirmer._dialog is not None)
    confirmer._dialog.deny_button.click()
    assert _pump(qapp, pending.done)

    result = pending.result(timeout=1.0)
    assert result.success is False
    assert "not approved" in (result.error or "")
    assert tool.typed == []
