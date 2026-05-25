"""Typed pub/sub event bus.

Design notes (kept here so future readers see the rationale):

- Event payloads are frozen dataclasses (cheap, immutable, no validation
  overhead on the hot audio/LLM path; payload sources are trusted internals).
- Dispatch is per-exact-type (no subclass fan-out). SPEC: "register a handler
  for a specific event type." Predictable beats clever.
- Handlers may be sync callables or async coroutines. If a handler returns a
  coroutine, the bus awaits it. This avoids forcing `async def` on simple
  UI/counter subscribers.
- Two publish methods:
    * publish(event)              -- fire-and-forget, safe from any thread.
    * await publish_and_wait(e)   -- async, completes after every handler.
- Loop binding: bus is bound to a specific asyncio loop. From the loop's own
  thread we use loop.create_task; from another thread we use
  asyncio.run_coroutine_threadsafe. SPEC § Threading model says publishers
  may be on any thread; subscribers always run on the loop.
- Exception isolation: each handler is wrapped in try/except; failures are
  logged via the `logging` module and never propagated. SPEC: "one handler
  raising does not prevent others from running."
- Mode and ConversationalState enums live in core.state_machine (they are
  semantically owned by the state machine); see that module's header.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.core.state_machine import ConversationalState, Mode

if TYPE_CHECKING:
    from jarvis.core.config import JarvisConfig

log = logging.getLogger(__name__)


# --- event payloads --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModeChanged:
    old: Mode
    new: Mode


@dataclass(frozen=True, slots=True)
class ConversationalStateChanged:
    old: ConversationalState
    new: ConversationalState


@dataclass(frozen=True, slots=True)
class ConfigChanged:
    old: JarvisConfig
    new: JarvisConfig
    changed_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WakeWordDetected:
    confidence: float


@dataclass(frozen=True, slots=True)
class TranscriptionReady:
    text: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class LLMResponseChunk:
    text: str


@dataclass(frozen=True, slots=True)
class LLMResponseComplete:
    full_text: str


@dataclass(frozen=True, slots=True)
class ToolInvoked:
    tool_name: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_name: str
    result: dict | str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class NonFatalError:
    """A recoverable error that degraded but did not prevent operation.

    Published by audio modules when they succeed via a fallback path so the
    composition root can surface a user-visible notice (e.g. tray balloon)
    without aborting the boot sequence.

    Attributes:
        module:   name of the publishing module (e.g. "tts", "audio_input")
        issue:    machine-readable issue key (e.g. "device_fallback")
        expected: what was configured / expected (device name, etc.)
        actual:   what was actually used (the fallback)
    """

    module: str
    issue: str
    expected: str
    actual: str


# --- bus -------------------------------------------------------------------

# A handler may be sync (returns None) or async (returns Awaitable[None]).
# We don't generic-parameterize on event type because the registry is a
# heterogeneous dict; subscribers narrow at the call site.
Handler = Callable[[Any], Awaitable[None] | None]
Unsubscribe = Callable[[], None]


class EventBus:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._handlers: dict[type, list[Handler]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = loop

    # -- loop management --

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the bus to a specific asyncio loop. Required if publish()
        will be called from threads other than the loop's own."""
        self._loop = loop

    def _resolve_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        # Fall back to the running loop, if any. This makes the bus usable
        # in tests without explicit binding.
        try:
            return asyncio.get_running_loop()
        except RuntimeError as e:
            raise RuntimeError(
                "EventBus has no loop bound and no running loop in this "
                "context; call bind_loop(loop) before publishing."
            ) from e

    # -- subscription --

    def subscribe(self, event_type: type, handler: Handler) -> Unsubscribe:
        """Register `handler` for `event_type`. Returns a callable that
        removes this exact registration when invoked. Idempotent: calling
        the returned unsubscribe twice is a no-op."""
        self._handlers[event_type].append(handler)

        removed = False

        def unsubscribe() -> None:
            nonlocal removed
            if removed:
                return
            removed = True
            handlers = self._handlers.get(event_type)
            if handlers is None:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    # -- publish --

    def publish(self, event: object) -> None:
        """Fire-and-forget. Safe from any thread. Schedules dispatch on the
        bound loop; returns immediately."""
        loop = self._resolve_loop()
        if loop.is_closed():
            log.warning("EventBus.publish called after loop close; dropping %r", event)
            return
        coro = self._dispatch(event)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, loop)

    async def publish_and_wait(self, event: object) -> None:
        """Dispatch and await every handler before returning. Use this in
        tests, or when a caller genuinely needs to know dispatch finished."""
        await self._dispatch(event)

    # -- internal dispatch --

    async def _dispatch(self, event: object) -> None:
        # Snapshot the handler list so unsubscribes during dispatch don't
        # mutate what we're iterating, and so a handler can safely subscribe
        # a new handler without affecting this delivery round.
        handlers = list(self._handlers.get(type(event), ()))
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception(
                    "EventBus handler %r raised on event %r",
                    getattr(handler, "__qualname__", handler),
                    event,
                )
