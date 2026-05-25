"""Global hotkey registration. Phase 5, [IMPL].

Listens for system-wide key combinations and dispatches the
corresponding intent — mute toggle, push-to-talk tap, open-settings.
Loads bindings from HotkeysConfig at construction; missing/empty
bindings are skipped silently.

Library choice: pynput
----------------------
SPEC § Open Questions §1 left the pynput-vs-keyboard call for this
phase. Picked pynput because:

  - `keyboard` requires administrator privileges on Windows for
    media-key hooks (volume, play/pause). Many users will eventually
    bind media keys; an installer that has to UAC-prompt on first
    launch is a non-starter for a polished local app.
  - pynput's Windows backend uses the low-level keyboard hook
    (SetWindowsHookEx WH_KEYBOARD_LL) which runs without elevation
    for the full key range, including media keys.
  - Hook registration latency is slightly higher (one-time cost at
    startup; ~100 ms vs ~10 ms for `keyboard`). We pay it once.
  - pynput is on PyPI as a pure wheel, no native build required.

Threading bridge
----------------
pynput's GlobalHotKeys listener runs on its own thread (a Windows
low-level hook thread on Win32). Action callbacks fire there, not on
the audio asyncio loop. Same constraint as TrayIcon's bus subscribers:
state-machine operations are loop-affine and must be marshalled.

We use the same pattern as tray.py:
    asyncio.run_coroutine_threadsafe(_coro(), self._audio_loop)
to push every mode toggle onto the audio loop. on_open_settings is a
Qt-thread callback (settings window construction) — we marshal it via
the injected callable, whose body is the composition root's
responsibility (it'll typically use QTimer.singleShot(0, slot) to land
on the Qt main thread).

Push-to-talk scope (Phase 5)
----------------------------
Hold-to-talk is the better UX: hold the key while speaking, VAD honours
the duration and treats key-release as a hard endpoint. pynput supports
press+release events but the wiring (key-down arms LISTENING, key-up
short-circuits the VAD endpoint timer, debouncing modifier keys,
behaviour when the key is held into a wake-word session, …) is real
work that doesn't pay off until headset users start asking for it.

Phase 5 ships tap-to-listen: pressing the configured PTT hotkey fires
the wake-word path exactly as if "hey jarvis" had been said —
publishes WakeWordDetected (confidence=1.0) and lets the existing
pipeline handle the LISTENING transition + listening-silence timeout.
Hold-to-talk slots in by adding a key-release handler in a future
phase; the current architecture supports the upgrade without
restructuring.

Hotkey string format
--------------------
HotkeysConfig stores user-readable strings: "ctrl+shift+m", "ctrl+space",
"f9". We translate those to pynput's GlobalHotKeys format which uses
"<ctrl>+<shift>+m" notation. The translation table covers the common
modifiers; unrecognised parts are passed through verbatim. Parse
failures log a warning and skip just that hotkey — they never block
register_all() because a broken binding shouldn't disable the others.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from jarvis.core.config import HotkeysConfig
from jarvis.core.events import EventBus, WakeWordDetected
from jarvis.core.state_machine import Mode, StateMachine

log = logging.getLogger(__name__)

# Translation table from config-style modifier names to pynput's
# angle-bracket notation. Pure-letter keys pass through unchanged.
_MODIFIER_TRANSLATIONS: dict[str, str] = {
    "ctrl": "<ctrl>",
    "control": "<ctrl>",
    "shift": "<shift>",
    "alt": "<alt>",
    "meta": "<cmd>",
    "win": "<cmd>",
    "cmd": "<cmd>",
    "super": "<cmd>",
    "space": "<space>",
    "tab": "<tab>",
    "enter": "<enter>",
    "return": "<enter>",
    "esc": "<esc>",
    "escape": "<esc>",
    "backspace": "<backspace>",
    "delete": "<delete>",
    "del": "<delete>",
    "up": "<up>",
    "down": "<down>",
    "left": "<left>",
    "right": "<right>",
    "home": "<home>",
    "end": "<end>",
    "pageup": "<page_up>",
    "pagedown": "<page_down>",
    "insert": "<insert>",
}
_MODIFIER_TRANSLATIONS.update(
    {f"f{i}": f"<f{i}>" for i in range(1, 25)}
)


def _to_pynput_hotkey(raw: str) -> str:
    """Translate a config hotkey string to pynput's GlobalHotKeys format.
    Raises ValueError on empty input so register_all() can log and skip
    just that one binding."""
    if not raw or not raw.strip():
        raise ValueError("empty hotkey string")
    parts = [p.strip().lower() for p in raw.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"no key parts in {raw!r}")
    translated: list[str] = []
    for part in parts:
        translated.append(_MODIFIER_TRANSLATIONS.get(part, part))
    return "+".join(translated)


class HotkeyManager:
    """Owns the pynput global hotkey listener. Lives on whatever thread
    constructed it (typically Qt main); the listener itself runs on
    pynput's hook thread. Same _alive shutdown discipline as TrayIcon
    and OverlayOrb.

    Lazy imports pynput in register_all() rather than at module import
    so the test suite can mock it cleanly without installing pynput
    into the test environment."""

    def __init__(
        self,
        *,
        sm: StateMachine,
        bus: EventBus,
        audio_loop: asyncio.AbstractEventLoop,
        hotkeys: HotkeysConfig,
        on_open_settings: Callable[[], None],
        on_mode_request: Callable[[Mode], Any] | None = None,
        on_open_command_palette: Callable[[], None] | None = None,
    ) -> None:
        self._sm = sm
        self._bus = bus
        self._audio_loop = audio_loop
        self._hotkeys = hotkeys
        self._on_open_settings = on_open_settings
        self._on_open_command_palette = on_open_command_palette
        # See TrayIcon: optional coroutine-returning hook that routes
        # mode requests through ModeCoordinator. None preserves legacy
        # direct-sm.set_mode dispatch.
        self._on_mode_request = on_mode_request
        self._alive: bool = True
        # The pynput GlobalHotKeys listener, or None until register_all
        # succeeds. Held so close() can stop it.
        self._listener: Any | None = None

    # -- registration ----------------------------------------------------

    def register_all(self) -> None:
        """Build the action map, start the pynput listener. Safe to call
        once; subsequent calls log and no-op. Parse failures on
        individual bindings are logged and skipped — they never block
        the remaining bindings from registering."""
        if self._listener is not None:
            log.info("hotkey listener already registered; ignoring")
            return
        actions: dict[str, Callable[[], None]] = {}
        for raw, action_name, action in (
            (self._hotkeys.mute, "mute", self._on_mute_hotkey),
            (
                self._hotkeys.push_to_talk,
                "push_to_talk",
                self._on_push_to_talk_hotkey,
            ),
            (
                self._hotkeys.open_settings,
                "open_settings",
                self._on_open_settings_hotkey,
            ),
            (
                getattr(self._hotkeys, "command_palette", None),
                "command_palette",
                self._on_command_palette_hotkey,
            ),
        ):
            if not raw:
                # None or empty: binding is intentionally disabled.
                continue
            try:
                key = _to_pynput_hotkey(raw)
            except ValueError as e:
                log.warning(
                    "skipping %s hotkey %r: %s", action_name, raw, e
                )
                continue
            if key in actions:
                log.warning(
                    "duplicate hotkey binding %r; %s overrides earlier",
                    key, action_name,
                )
            actions[key] = action
        if not actions:
            log.info("no hotkeys configured; listener not started")
            return
        # Lazy import: keeps the test suite mock-friendly and lets a
        # headless CI box without pynput's display dependency import
        # this module without failing.
        try:
            from pynput.keyboard import GlobalHotKeys
        except ImportError as e:
            log.error("pynput not installed; hotkeys disabled: %s", e)
            return
        try:
            self._listener = GlobalHotKeys(actions)
            self._listener.start()
        except Exception:
            log.exception("failed to start pynput hotkey listener")
            self._listener = None
            return
        log.info(
            "registered %d hotkey(s): %s",
            len(actions), ", ".join(actions),
        )

    # -- action handlers (pynput thread) --------------------------------

    def _on_mute_hotkey(self) -> None:
        """Tap to toggle between ACTIVE and MUTED. SLEEPING stays
        SLEEPING — the user uses a different signal for wake."""
        if not self._alive:
            return
        current = self._sm.mode
        if current is Mode.ACTIVE:
            target = Mode.MUTED
        elif current is Mode.MUTED:
            target = Mode.ACTIVE
        else:
            # SLEEPING: don't intercept; mute means nothing when the
            # mic is already cold. No-op.
            log.info("mute hotkey ignored: mode is SLEEPING")
            return
        self._dispatch_set_mode(target)

    def _on_push_to_talk_hotkey(self) -> None:
        """Tap-to-listen for Phase 5 (see module docstring). Publishes
        a synthetic WakeWordDetected with confidence=1.0; the pipeline's
        existing wake handler does the LISTENING transition and the
        listening-silence timeout."""
        if not self._alive:
            return
        if self._sm.mode is not Mode.ACTIVE:
            log.info(
                "push-to-talk ignored: mode is %s", self._sm.mode.name
            )
            return
        # The bus is loop-affine (publish calls call_soon_threadsafe);
        # safe to invoke from the pynput thread.
        try:
            self._bus.publish(WakeWordDetected(confidence=1.0))
        except Exception:
            log.exception("failed to publish synthetic WakeWordDetected")

    def _on_open_settings_hotkey(self) -> None:
        """Fire the injected on_open_settings callback. The composition
        root is responsible for marshalling onto the Qt thread (typical
        body: QTimer.singleShot(0, slot))."""
        if not self._alive:
            return
        try:
            self._on_open_settings()
        except Exception:
            log.exception("on_open_settings callback raised")

    def _on_command_palette_hotkey(self) -> None:
        """Fire the injected on_open_command_palette callback. The composition
        root marshals onto the Qt thread."""
        if not self._alive:
            return
        if self._on_open_command_palette is None:
            return
        try:
            self._on_open_command_palette()
        except Exception:
            log.exception("on_open_command_palette callback raised")

    # -- audio-loop marshalling -----------------------------------------

    def _dispatch_set_mode(self, new: Mode) -> None:
        """Marshal a Mode request onto the audio loop. Uses the injected
        on_mode_request coroutine when present (production: routes through
        ModeCoordinator); otherwise falls back to direct sm.set_mode.
        Fire-and-forget so a failure on the audio side never crashes the
        pynput hook thread."""
        if self._on_mode_request is not None:
            request = self._on_mode_request
            async def _coro() -> None:
                result = request(new)
                if asyncio.iscoroutine(result):
                    await result
        else:
            async def _coro() -> None:
                self._sm.set_mode(new)
        try:
            asyncio.run_coroutine_threadsafe(_coro(), self._audio_loop)
        except Exception:
            log.exception(
                "failed to dispatch set_mode(%s) to audio loop", new
            )

    # -- shutdown -------------------------------------------------------

    def close(self) -> None:
        """Stop the listener, flip _alive so any in-flight hook call
        no-ops. Idempotent. Mirrors TrayIcon.close() / OverlayOrb.close()
        so the composition root has one shutdown pattern."""
        if not self._alive:
            return
        self._alive = False
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                log.exception("hotkey listener stop failed")
            self._listener = None
