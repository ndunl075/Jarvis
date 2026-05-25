"""Tests for jarvis.core.events: subscription, ordering, isolation, threading."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from jarvis.core.config import JarvisConfig
from jarvis.core.events import (
    ConfigChanged,
    ConversationalStateChanged,
    EventBus,
    LLMResponseChunk,
    ModeChanged,
    ToolInvoked,
    ToolResult,
    TranscriptionReady,
    WakeWordDetected,
)
from jarvis.core.state_machine import ConversationalState, Mode

# --- payload smoke ---------------------------------------------------------


def test_event_dataclasses_are_frozen():
    # frozen=True dataclasses raise FrozenInstanceError (a subclass of AttributeError).
    e = WakeWordDetected(confidence=0.9)
    with pytest.raises(AttributeError):
        e.confidence = 0.5  # type: ignore[misc]


def test_required_event_types_construct():
    # Just verifying the SPEC-required event surface is present and the
    # constructors are sane.
    ModeChanged(old=Mode.ACTIVE, new=Mode.MUTED)
    ConversationalStateChanged(
        old=ConversationalState.IDLE, new=ConversationalState.LISTENING
    )
    old_cfg = JarvisConfig()
    new_cfg = JarvisConfig()
    new_cfg.llm.temperature = 0.5
    ConfigChanged(old=old_cfg, new=new_cfg, changed_fields=("llm.temperature",))
    WakeWordDetected(confidence=0.95)
    TranscriptionReady(text="hello", duration_ms=500)
    LLMResponseChunk(text="partial")
    ToolInvoked(tool_name="screenshot", args={})
    ToolResult(tool_name="screenshot", result="ok", error=None)


# --- subscribe / publish_and_wait -----------------------------------------


async def test_subscribe_delivers_to_matching_handler():
    bus = EventBus()
    received: list[WakeWordDetected] = []
    bus.subscribe(WakeWordDetected, lambda e: received.append(e))
    await bus.publish_and_wait(WakeWordDetected(confidence=0.9))
    assert len(received) == 1
    assert received[0].confidence == 0.9


async def test_unsubscribed_event_type_not_delivered():
    bus = EventBus()
    received: list = []
    bus.subscribe(WakeWordDetected, lambda e: received.append(e))
    await bus.publish_and_wait(TranscriptionReady(text="x", duration_ms=10))
    assert received == []


async def test_handlers_called_in_registration_order():
    bus = EventBus()
    order: list[int] = []
    bus.subscribe(WakeWordDetected, lambda e: order.append(1))
    bus.subscribe(WakeWordDetected, lambda e: order.append(2))
    bus.subscribe(WakeWordDetected, lambda e: order.append(3))
    await bus.publish_and_wait(WakeWordDetected(confidence=1.0))
    assert order == [1, 2, 3]


async def test_async_handler_is_awaited():
    bus = EventBus()
    seen: list[str] = []

    async def handler(e: WakeWordDetected) -> None:
        await asyncio.sleep(0)
        seen.append("async")

    bus.subscribe(WakeWordDetected, handler)
    await bus.publish_and_wait(WakeWordDetected(confidence=0.5))
    assert seen == ["async"]


async def test_sync_and_async_handlers_can_coexist():
    bus = EventBus()
    seen: list[str] = []

    async def ah(e: WakeWordDetected) -> None:
        seen.append("a")

    bus.subscribe(WakeWordDetected, lambda e: seen.append("s1"))
    bus.subscribe(WakeWordDetected, ah)
    bus.subscribe(WakeWordDetected, lambda e: seen.append("s2"))
    await bus.publish_and_wait(WakeWordDetected(confidence=0.5))
    assert seen == ["s1", "a", "s2"]


# --- exception isolation ---------------------------------------------------


async def test_handler_exception_does_not_block_subsequent_handlers(
    caplog: pytest.LogCaptureFixture,
):
    bus = EventBus()
    seen: list[int] = []

    def boom(e: WakeWordDetected) -> None:
        raise RuntimeError("intentional")

    bus.subscribe(WakeWordDetected, lambda e: seen.append(1))
    bus.subscribe(WakeWordDetected, boom)
    bus.subscribe(WakeWordDetected, lambda e: seen.append(3))

    with caplog.at_level(logging.ERROR, logger="jarvis.core.events"):
        await bus.publish_and_wait(WakeWordDetected(confidence=0.5))

    assert seen == [1, 3]
    assert any("intentional" in r.message or "raised" in r.message for r in caplog.records)


async def test_async_handler_exception_isolated(caplog: pytest.LogCaptureFixture):
    bus = EventBus()
    seen: list[int] = []

    async def boom(e: WakeWordDetected) -> None:
        raise ValueError("nope")

    bus.subscribe(WakeWordDetected, boom)
    bus.subscribe(WakeWordDetected, lambda e: seen.append(99))

    with caplog.at_level(logging.ERROR, logger="jarvis.core.events"):
        await bus.publish_and_wait(WakeWordDetected(confidence=0.5))

    assert seen == [99]


# --- unsubscribe -----------------------------------------------------------


async def test_unsubscribe_removes_handler():
    bus = EventBus()
    seen: list[int] = []
    unsub = bus.subscribe(WakeWordDetected, lambda e: seen.append(1))
    bus.subscribe(WakeWordDetected, lambda e: seen.append(2))
    unsub()
    await bus.publish_and_wait(WakeWordDetected(confidence=0.5))
    assert seen == [2]


async def test_unsubscribe_is_idempotent():
    bus = EventBus()
    seen: list[int] = []
    unsub = bus.subscribe(WakeWordDetected, lambda e: seen.append(1))
    unsub()
    unsub()  # second call must be a no-op, not an error
    await bus.publish_and_wait(WakeWordDetected(confidence=0.5))
    assert seen == []


async def test_unsubscribe_during_dispatch_does_not_corrupt_iteration():
    bus = EventBus()
    seen: list[str] = []

    def first(e: WakeWordDetected) -> None:
        seen.append("first")
        unsub_second()  # remove second mid-dispatch

    def second(e: WakeWordDetected) -> None:
        seen.append("second")

    def third(e: WakeWordDetected) -> None:
        seen.append("third")

    bus.subscribe(WakeWordDetected, first)
    unsub_second = bus.subscribe(WakeWordDetected, second)
    bus.subscribe(WakeWordDetected, third)

    # Snapshot semantics: this round still calls all three; next round skips second.
    await bus.publish_and_wait(WakeWordDetected(confidence=0.5))
    assert seen == ["first", "second", "third"]

    seen.clear()
    await bus.publish_and_wait(WakeWordDetected(confidence=0.5))
    assert seen == ["first", "third"]


# --- fire-and-forget publish ----------------------------------------------


async def test_publish_from_loop_thread_schedules_task():
    bus = EventBus()
    delivered = asyncio.Event()
    received: list = []

    async def handler(e: WakeWordDetected) -> None:
        received.append(e)
        delivered.set()

    bus.subscribe(WakeWordDetected, handler)
    bus.publish(WakeWordDetected(confidence=0.5))  # not awaited
    await asyncio.wait_for(delivered.wait(), timeout=1.0)
    assert len(received) == 1


async def test_publish_from_other_thread_crosses_to_loop():
    loop = asyncio.get_running_loop()
    bus = EventBus(loop=loop)
    delivered = asyncio.Event()
    received: list[tuple[str, int]] = []

    def handler(e: WakeWordDetected) -> None:
        # Record which thread we're on. Should be the loop thread, not the
        # publisher's thread.
        received.append((threading.current_thread().name, id(threading.current_thread())))
        loop.call_soon_threadsafe(delivered.set)

    bus.subscribe(WakeWordDetected, handler)

    publisher_thread_id: list[int] = []

    def publisher() -> None:
        publisher_thread_id.append(id(threading.current_thread()))
        bus.publish(WakeWordDetected(confidence=0.7))

    t = threading.Thread(target=publisher, name="publisher-thread")
    t.start()
    t.join()

    await asyncio.wait_for(delivered.wait(), timeout=2.0)
    assert len(received) == 1
    handler_thread_id = received[0][1]
    assert handler_thread_id != publisher_thread_id[0], (
        "handler must run on loop thread, not the publisher's thread"
    )


def test_publish_with_no_loop_bound_and_no_running_loop_raises():
    bus = EventBus()
    with pytest.raises(RuntimeError, match="loop"):
        bus.publish(WakeWordDetected(confidence=0.5))
