"""Modal confirmation prompt — the UX behind `Tool.requires_confirmation`.

`ToolRegistry.execute()` refuses to dispatch a tool that asks for
confirmation until something says yes. On the desktop that something is
`QtToolConfirmer`: a Qt-main-thread object that raises the dialog below,
default-DENIES on timeout, and hands the verdict back to the audio loop.

Why a dialog
------------
The original design note deferred this feature because both candidate
UXes needed the audio path — a spoken "cancel" window needs barge-in
(speaker → mic feedback keeps it disabled) and a spoken "yes" needs STT
during TTS. A dialog needs neither. It never touches audio at all.

Threading
---------
The request originates on the audio thread, inside an async round of
`IntentRouter._run_tools`. The dialog must be built and shown on the Qt
main thread. Both directions use the bridges `jarvis/app.py` documents,
and nothing here invents a third:

    audio → Qt   QMetaObject.invokeMethod(..., Qt.QueuedConnection)
    Qt → audio   loop.call_soon_threadsafe(<resolve the future>)

`confirm()` runs on the audio loop. It creates an `asyncio.Future` on
that loop, posts `_show_prompt` to Qt, and then **awaits** the future —
it never blocks the audio thread, so the loop keeps servicing the
pipeline, and it never blocks the Qt thread, because the dialog is shown
with `show()` under `ApplicationModal` rather than `exec()`. `exec()`
would spin a nested Qt event loop, and a nested loop is what turns an
innocuous "quit while a prompt is open" into a re-entrancy puzzle.

Four things had to be true, and each has a mechanism:

1. The audio loop must not be able to wait forever. The dialog owns a
   visible countdown and denies at zero; `confirm()` additionally awaits
   under `asyncio.wait_for` with a slightly longer watchdog, so even a
   Qt thread that never runs (no event loop yet, UI wedged, app tearing
   down) resolves to a denial rather than a hang.

2. Shutdown must not be able to hang. The watchdog above expires at
   `_TIMEOUT + _WATCHDOG_GRACE`, deliberately under `app.py`'s 10 s
   audio-thread join. Belt and braces: `JarvisApp._on_quit()` calls
   `close()` *before* it signals the audio loop, which resolves any
   pending verdict to False immediately, so a prompt open at quit costs
   the shutdown nothing at all.

3. A second request while one is open must behave. It is DENIED, not
   queued. Queueing means showing someone a prompt for an utterance
   several seconds stale, whose own timeout has been ticking down behind
   the first dialog — an approval given to the wrong question. The two
   concurrent-round sources (a voice command and a command-palette
   submission) make this rare, and rare-and-refused beats rare-and
   -confusing. The refusal is a normal denial, so the tool does not run.

4. Late answers must not resolve the wrong request. Every prompt carries
   a token; a verdict is only applied if the token still matches the
   pending request.

Fail-closed everywhere: the only path that returns True is a click on
Approve.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass

from PySide6.QtCore import (
    Q_ARG,
    QMetaObject,
    QObject,
    Qt,
    QTimer,
    Slot,
)
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from jarvis.tools.registry import ConfirmationRequest

log = logging.getLogger(__name__)

# How long the user has to answer.
#
# 8 s, chosen against three constraints rather than taste:
#   - long enough to read a tool name, one summary line and a couple of
#     JSON arguments and then move a mouse. Under ~5 s the honest answer
#     for anyone not already hovering the button is "denied", which makes
#     the gate a blocker rather than a question.
#   - short enough that an unanswered prompt (user walked away, spoke to
#     the room, is looking at another screen) clears itself well inside
#     the attention span of the utterance that caused it.
#   - strictly under app.py's 10 s shutdown join once the watchdog grace
#     below is added, so a prompt can never be the thing that makes quit
#     take longer than quitting is allowed to take.
_TIMEOUT_SECONDS = 8.0

# Extra slack on the audio-side await, so the dialog's own countdown is
# what normally decides and the watchdog only fires when Qt never
# answered at all. 8.0 + 1.5 = 9.5 s < the 10 s audio-thread join.
_WATCHDOG_GRACE_SECONDS = 1.5

# Countdown label refresh. Fast enough to look live, slow enough to be
# invisible in a profile.
_TICK_MS = 200

# The arguments box is evidence, not a document: a pathological blob
# gets elided rather than pushing the buttons off screen.
_MAX_ARGS_CHARS = 1500

_BG = "#0d0d0d"
_BG_CARD = "#141414"
_BORDER = "#1f1f1f"
_TEXT = "#ffffff"
_TEXT_DIM = "#808080"
_TEXT_MID = "#b0b0b0"
_CYAN = "#38f4ff"
_AMBER = "#ffb648"

_DENY_STYLE = (
    f"QPushButton {{ color:{_TEXT}; background:{_BG_CARD}; "
    f"border:1px solid {_BORDER}; border-radius:4px; padding:7px 22px; "
    "font-size:10pt; }"
    f"QPushButton:hover {{ border-color:{_TEXT_DIM}; }}"
    f"QPushButton:focus {{ border-color:{_TEXT}; }}"
)
_APPROVE_STYLE = (
    f"QPushButton {{ color:{_BG}; background:{_CYAN}; border:none; "
    "border-radius:4px; padding:7px 22px; font-size:10pt; "
    "font-weight:500; }"
    "QPushButton:hover { background:#6bf8ff; }"
)


def format_arguments(arguments: dict) -> str:
    """Pretty-print tool arguments for the evidence box.

    Sorted so the same call always renders identically, `default=str` so
    an exotic value degrades to its repr instead of blanking the box,
    and elided past _MAX_ARGS_CHARS."""
    try:
        text = json.dumps(arguments, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = repr(arguments)
    if len(text) > _MAX_ARGS_CHARS:
        return text[:_MAX_ARGS_CHARS] + "\n… (truncated)"
    return text


class ToolConfirmationDialog(QDialog):
    """The prompt itself. Approve is the only path to `Accepted`.

    Escape, the window close button, and the countdown reaching zero all
    reject, and QDialog's own defaults give the first two for free —
    which is the reason this subclasses QDialog rather than assembling a
    QWidget with hand-rolled key handling.

    Deny is the default button: a stray Return goes to the safe answer."""

    def __init__(
        self,
        *,
        tool_name: str,
        summary: str,
        arguments_text: str,
        timeout_seconds: float = _TIMEOUT_SECONDS,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._timeout_seconds = max(0.0, float(timeout_seconds))
        self._elapsed = 0.0

        self.setWindowTitle("Jarvis — confirm tool")
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumWidth(460)
        self.setStyleSheet(f"QDialog {{ background:{_BG}; }}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)

        heading = QLabel("Confirm tool")
        heading.setStyleSheet(
            f"color:{_TEXT_DIM}; font-size:8pt; letter-spacing:2px; "
            "background:transparent;"
        )
        layout.addWidget(heading)

        name = QLabel(tool_name)
        name.setStyleSheet(
            f"color:{_CYAN}; font-size:14pt; font-weight:500; "
            "background:transparent;"
        )
        layout.addWidget(name)

        self.summary_label = QLabel(summary or "(no description)")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            f"color:{_TEXT_MID}; font-size:10pt; background:transparent;"
        )
        layout.addWidget(self.summary_label)

        args_frame = QFrame()
        args_frame.setStyleSheet(
            f"QFrame {{ background:{_BG_CARD}; border:1px solid {_BORDER}; "
            "border-radius:6px; }"
        )
        args_layout = QVBoxLayout(args_frame)
        args_layout.setContentsMargins(2, 2, 2, 2)

        self.arguments_view = QPlainTextEdit(arguments_text)
        self.arguments_view.setReadOnly(True)
        self.arguments_view.setMaximumHeight(160)
        self.arguments_view.setStyleSheet(
            f"QPlainTextEdit {{ background:transparent; color:{_TEXT}; "
            "border:none; font-family:Consolas,'Courier New',monospace; "
            "font-size:9pt; }"
        )
        args_layout.addWidget(self.arguments_view)
        layout.addWidget(args_frame)

        self.countdown_label = QLabel()
        self.countdown_label.setStyleSheet(
            f"color:{_AMBER}; font-size:9pt; background:transparent;"
        )
        layout.addWidget(self.countdown_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.deny_button = QPushButton("Deny")
        self.deny_button.setStyleSheet(_DENY_STYLE)
        self.deny_button.setDefault(True)
        self.deny_button.setAutoDefault(True)
        self.deny_button.clicked.connect(self.reject)
        buttons.addWidget(self.deny_button)

        self.approve_button = QPushButton("Approve")
        self.approve_button.setStyleSheet(_APPROVE_STYLE)
        self.approve_button.setAutoDefault(False)
        self.approve_button.clicked.connect(self.accept)
        buttons.addWidget(self.approve_button)
        layout.addLayout(buttons)

        self._update_countdown()

        # Two timers, on purpose. The single-shot deadline is what
        # actually denies, so the verdict does not depend on how many
        # ticks the label got; the repeating tick only paints.
        self._deadline = QTimer(self)
        self._deadline.setSingleShot(True)
        self._deadline.timeout.connect(self._on_timeout)
        self._deadline.start(int(self._timeout_seconds * 1000))

        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._on_tick)
        self._ticker.start(_TICK_MS)

        self.finished.connect(self._stop_timers)

        # Deny holds focus so keyboard-only interaction cannot approve by
        # accident; approving is a deliberate Tab-then-Space or a click.
        self.deny_button.setFocus()

    # -- countdown ----------------------------------------------------

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self._timeout_seconds - self._elapsed)

    def _update_countdown(self) -> None:
        self.countdown_label.setText(
            f"Denied automatically in {self.remaining_seconds:.0f} s."
        )

    @Slot()
    def _on_tick(self) -> None:
        self._elapsed += _TICK_MS / 1000.0
        self._update_countdown()

    @Slot()
    def _on_timeout(self) -> None:
        self._elapsed = self._timeout_seconds
        self._update_countdown()
        log.info("tool confirmation timed out; denying")
        self.reject()

    @Slot()
    def _stop_timers(self) -> None:
        self._deadline.stop()
        self._ticker.stop()


@dataclass(frozen=True, slots=True)
class _Pending:
    """One in-flight request: which loop to answer on, and with what."""

    token: int
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


def _settle(pending: _Pending, approved: bool) -> None:
    """Deliver a verdict to the audio loop (Qt → audio).

    The done() check runs *on the audio loop*, not here, because that is
    the only thread allowed to inspect the future: by the time this
    lands, `confirm()` may already have given up on it via its watchdog."""

    def _apply() -> None:
        if not pending.future.done():
            pending.future.set_result(approved)

    try:
        pending.loop.call_soon_threadsafe(_apply)
    except RuntimeError:
        # Loop already closed — the audio thread is gone and nothing is
        # waiting for this answer any more.
        log.debug("audio loop closed before the verdict was delivered")


def _close_dialog(dialog: ToolConfirmationDialog | None) -> None:
    """Reject then schedule deletion. Qt main thread; tolerant of a
    dialog Qt has already destroyed underneath us."""
    if dialog is None:
        return
    try:
        dialog.reject()
        dialog.deleteLater()
    except RuntimeError:  # pragma: no cover - already destroyed
        log.debug("confirmation dialog already gone", exc_info=True)


class QtToolConfirmer(QObject):
    """`ToolConfirmer` implementation backed by a modal Qt dialog.

    Constructed on the Qt main thread (after the QApplication exists and
    before the audio thread starts, so there is no window in which a
    confirmable tool could run unguarded). `confirm()` is awaited on the
    audio loop; every `@Slot` below runs on the Qt main thread."""

    def __init__(
        self,
        *,
        timeout_seconds: float = _TIMEOUT_SECONDS,
        watchdog_grace_seconds: float = _WATCHDOG_GRACE_SECONDS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._timeout_seconds = float(timeout_seconds)
        self._watchdog_seconds = float(timeout_seconds) + float(
            watchdog_grace_seconds
        )
        # Guards _pending and _alive, which the audio thread reads and
        # writes in confirm() while the Qt thread does the same in the
        # slots. _dialog is Qt-thread-only and needs no lock.
        self._lock = threading.Lock()
        self._pending: _Pending | None = None
        self._alive = True
        self._next_token = 0
        self._dialog: ToolConfirmationDialog | None = None
        self._dialog_token = -1

    # -- audio-thread entry point --------------------------------------

    async def confirm(self, request: ConfirmationRequest) -> bool:
        """Awaited on the audio loop. True only for an explicit approval."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        with self._lock:
            if not self._alive:
                log.info(
                    "confirmation for %r refused: shutting down",
                    request.tool_name,
                )
                return False
            if self._pending is not None:
                log.warning(
                    "confirmation for %r refused: another prompt is open",
                    request.tool_name,
                )
                return False
            self._next_token += 1
            token = self._next_token
            self._pending = _Pending(token=token, loop=loop, future=future)

        posted = QMetaObject.invokeMethod(
            self,
            "_show_prompt",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(int, token),
            Q_ARG(str, request.tool_name),
            Q_ARG(str, request.summary),
            Q_ARG(str, format_arguments(request.arguments)),
        )
        if posted is False:
            # The slot could not even be queued; nothing will ever answer.
            log.error("could not post the confirmation prompt to the Qt thread")
            self._resolve(token, approved=False)

        try:
            return await asyncio.wait_for(future, timeout=self._watchdog_seconds)
        except TimeoutError:
            log.warning(
                "no verdict for %r within %.1f s; denying",
                request.tool_name,
                self._watchdog_seconds,
            )
            self._abandon(token)
            return False
        except asyncio.CancelledError:
            self._abandon(token)
            raise

    # -- Qt main thread -------------------------------------------------

    @Slot(int, str, str, str)
    def _show_prompt(
        self, token: int, tool_name: str, summary: str, arguments_text: str
    ) -> None:
        """Raise the dialog. Runs on the Qt main thread."""
        with self._lock:
            live = self._alive
            stale = self._pending is None or self._pending.token != token
        if stale or not live:
            # Abandoned or shut down between posting and delivery.
            return

        dialog = ToolConfirmationDialog(
            tool_name=tool_name,
            summary=summary,
            arguments_text=arguments_text,
            timeout_seconds=self._timeout_seconds,
        )
        self._dialog = dialog
        self._dialog_token = token
        dialog.finished.connect(
            lambda code, t=token: self._on_finished(t, code)
        )
        # show(), not exec(): application-modal behaviour without a
        # nested event loop. See the module docstring.
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_finished(self, token: int, code: int) -> None:
        """Dialog closed for any reason. Qt main thread."""
        if self._dialog_token == token:
            dialog = self._forget_dialog()
            if dialog is not None:
                dialog.deleteLater()
        self._resolve(token, approved=code == QDialog.DialogCode.Accepted)

    @Slot(int)
    def _dismiss(self, token: int) -> None:
        """Tear down a dialog whose requester has already given up."""
        if self._dialog is not None and self._dialog_token == token:
            _close_dialog(self._forget_dialog())

    def close(self) -> None:
        """Cancel any open prompt and refuse all future ones.

        Called by `JarvisApp._on_quit()` on the Qt thread *before* the
        audio loop is signalled, so an awaiting `confirm()` is already
        resolved to False when shutdown starts. This is what keeps a
        prompt that is open at quit from costing the 10 s join."""
        with self._lock:
            self._alive = False
            pending = self._pending
            self._pending = None
        if pending is not None:
            log.info("confirmation prompt cancelled by shutdown; denying")
            _settle(pending, approved=False)
        _close_dialog(self._forget_dialog())

    # -- shared helpers -------------------------------------------------

    def _forget_dialog(self) -> ToolConfirmationDialog | None:
        """Drop the reference to the current dialog and return it."""
        dialog = self._dialog
        self._dialog = None
        self._dialog_token = -1
        return dialog

    def _resolve(self, token: int, *, approved: bool) -> None:
        """Answer the pending request iff it is still the one `token` names."""
        with self._lock:
            pending = self._pending
            if pending is None or pending.token != token:
                return
            self._pending = None
        _settle(pending, approved)

    def _abandon(self, token: int) -> None:
        """Give up on `token` from the audio side and close its dialog."""
        with self._lock:
            if self._pending is not None and self._pending.token == token:
                self._pending = None
        QMetaObject.invokeMethod(
            self,
            "_dismiss",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(int, token),
        )
