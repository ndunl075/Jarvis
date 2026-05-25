"""Tests for jarvis.llm.ollama_client.OllamaClient.

httpx is fully stubbed; no real Ollama daemon needed in CI. The
load-bearing test is `test_cancellation_closes_underlying_stream` --
it verifies that cancelling the consumer task actually exits the httpx
stream context manager (closing the connection), not just that the
consumer stopped iterating. This is the ResponseProducer cancellation
contract documented in audio/protocols.py.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from jarvis.core.lifecycle import Loadable
from jarvis.llm.ollama_client import (
    ChatChunk,
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
)

# --- mock helpers ------------------------------------------------------


class _MockStreamCM:
    """Mock for the async-context-manager returned by httpx.AsyncClient.stream().

    Tracks entry/exit so tests can assert the stream was closed on
    cancellation. Yields lines one at a time, optionally sleeping
    between them so a test can cancel mid-stream."""

    def __init__(
        self,
        lines: list[bytes | str],
        *,
        status_code: int = 200,
        between_sleep: float = 0.0,
        body_text: str = "",
    ) -> None:
        self._lines = lines
        self._status_code = status_code
        self._between_sleep = between_sleep
        self._body_text = body_text
        self.entered = False
        self.exited = False
        self.exit_exc_type = None

    async def __aenter__(self):
        self.entered = True
        response = MagicMock()
        response.status_code = self._status_code
        response.text = self._body_text
        response.aread = AsyncMock(return_value=self._body_text.encode())
        if self._status_code >= 400 and self._status_code != 404:
            request = MagicMock()
            response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    f"HTTP {self._status_code}", request=request, response=response,
                )
            )
        else:
            response.raise_for_status = MagicMock()

        sleep_between = self._between_sleep
        lines = self._lines

        async def aiter_lines():
            for line in lines:
                if isinstance(line, bytes):
                    yield line.decode()
                else:
                    yield line
                if sleep_between:
                    await asyncio.sleep(sleep_between)

        response.aiter_lines = aiter_lines
        return response

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        self.exit_exc_type = exc_type
        return False  # don't suppress


class _RaisingStreamCM:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *_):
        return False


def _ndjson_line(payload: dict) -> str:
    return json.dumps(payload)


@pytest.fixture
def mock_httpx():
    """Patch httpx.AsyncClient at the ollama_client module's import site.
    Returns the mock instance for per-test customization."""
    with patch("jarvis.llm.ollama_client.httpx.AsyncClient") as cls:
        instance = MagicMock()
        instance.post = AsyncMock()
        instance.aclose = AsyncMock()
        cls.return_value = instance
        yield instance


# --- protocol smoke ----------------------------------------------------


def test_implements_loadable():
    c = OllamaClient()
    assert isinstance(c, Loadable)


# --- Loadable lifecycle ----------------------------------------------


async def test_load_is_noop_sets_is_loaded(mock_httpx):
    c = OllamaClient()
    assert not c.is_loaded
    await c.load()
    assert c.is_loaded
    # No HTTP made on load.
    mock_httpx.post.assert_not_called()
    mock_httpx.stream.assert_not_called()


async def test_unload_sends_keep_alive_zero_to_generate(mock_httpx):
    c = OllamaClient(model="qwen2.5:7b-instruct")
    await c.load()
    await c.unload()

    mock_httpx.post.assert_awaited_once()
    args, kwargs = mock_httpx.post.call_args
    assert args[0] == "/api/generate"
    body = kwargs["json"]
    assert body["model"] == "qwen2.5:7b-instruct"
    assert body["keep_alive"] == 0


async def test_unload_closes_client_and_clears_is_loaded(mock_httpx):
    c = OllamaClient()
    await c.load()
    await c.unload()
    mock_httpx.aclose.assert_awaited_once()
    assert not c.is_loaded
    assert c._client is None  # type: ignore[attr-defined]


async def test_unload_swallows_eviction_failure(mock_httpx, caplog):
    """Daemon may be down at SLEEPING time. Lifecycle policy says
    unloading transitions are best-effort -- must not raise."""
    import logging
    mock_httpx.post.side_effect = httpx.ConnectError("connection refused")
    c = OllamaClient()
    await c.load()
    with caplog.at_level(logging.WARNING, logger="jarvis.llm.ollama_client"):
        await c.unload()  # must not raise
    assert any("eviction request failed" in r.message for r in caplog.records)
    assert not c.is_loaded


# --- warm() ----------------------------------------------------------


async def test_warm_posts_minimal_non_streaming_chat(mock_httpx):
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    mock_httpx.post = AsyncMock(return_value=response)
    c = OllamaClient(model="qwen2.5:7b-instruct", keep_alive_seconds=1800)
    await c.warm()

    mock_httpx.post.assert_awaited_once()
    args, kwargs = mock_httpx.post.call_args
    assert args[0] == "/api/chat"
    body = kwargs["json"]
    assert body["model"] == "qwen2.5:7b-instruct"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "ready"}]
    assert body["keep_alive"] == "1800s"
    assert body["options"]["num_predict"] == 1


async def test_warm_raises_connection_error_with_hint(mock_httpx):
    mock_httpx.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    c = OllamaClient()
    with pytest.raises(OllamaConnectionError, match="ollama serve"):
        await c.warm()


async def test_warm_raises_model_not_found_on_404(mock_httpx):
    response = MagicMock()
    response.status_code = 404
    mock_httpx.post = AsyncMock(return_value=response)
    c = OllamaClient(model="ghost:1b")
    with pytest.raises(OllamaModelNotFoundError) as exc:
        await c.warm()
    assert "ghost:1b" in str(exc.value)
    assert "ollama pull" in str(exc.value)


# --- stream_chat: chunk assembly --------------------------------------


async def test_stream_chat_yields_content_chunks_in_order(mock_httpx):
    lines = [
        _ndjson_line({"message": {"content": "Hello"}, "done": False}),
        _ndjson_line({"message": {"content": " "}, "done": False}),
        _ndjson_line({"message": {"content": "world"}, "done": False}),
        _ndjson_line({"message": {"content": ""}, "done": True}),
    ]
    cm = _MockStreamCM(lines)
    mock_httpx.stream = MagicMock(return_value=cm)

    c = OllamaClient(system_prompt="")
    chunks = []
    async for ch in c.stream_chat([{"role": "user", "content": "hi"}]):
        chunks.append(ch)

    contents = [ch.content for ch in chunks]
    assert contents == ["Hello", " ", "world", ""]
    assert chunks[-1].done is True
    assert all(ch.done is False for ch in chunks[:-1])


async def test_stream_chat_emits_tool_calls_on_final_chunk(mock_httpx):
    tool_call = {"function": {"name": "screenshot", "arguments": {}}}
    lines = [
        _ndjson_line({
            "message": {"content": "", "tool_calls": [tool_call]},
            "done": True,
        }),
    ]
    mock_httpx.stream = MagicMock(return_value=_MockStreamCM(lines))

    c = OllamaClient()
    chunks = [ch async for ch in c.stream_chat([{"role": "user", "content": "shot"}])]
    assert len(chunks) == 1
    assert chunks[-1].tool_calls == (tool_call,)
    assert chunks[-1].done is True


async def test_stream_chat_skips_empty_and_malformed_lines(mock_httpx, caplog):
    import logging
    lines = [
        "",
        "not-json",
        _ndjson_line({"message": {"content": "ok"}, "done": True}),
    ]
    mock_httpx.stream = MagicMock(return_value=_MockStreamCM(lines))

    c = OllamaClient()
    with caplog.at_level(logging.WARNING, logger="jarvis.llm.ollama_client"):
        chunks = [ch async for ch in c.stream_chat([{"role": "user", "content": "hi"}])]
    assert [ch.content for ch in chunks] == ["ok"]
    assert any("non-JSON" in r.message for r in caplog.records)


async def test_stream_chat_returns_after_done(mock_httpx):
    """Lines after the done=True chunk must NOT be yielded; the generator
    must terminate as soon as done is seen."""
    lines = [
        _ndjson_line({"message": {"content": "first"}, "done": True}),
        _ndjson_line({"message": {"content": "should-not-see"}, "done": False}),
    ]
    mock_httpx.stream = MagicMock(return_value=_MockStreamCM(lines))

    c = OllamaClient()
    chunks = [ch async for ch in c.stream_chat([{"role": "user", "content": "hi"}])]
    assert len(chunks) == 1
    assert chunks[0].content == "first"


# --- request body construction ---------------------------------------


async def test_request_body_includes_model_and_options(mock_httpx):
    mock_httpx.stream = MagicMock(return_value=_MockStreamCM([
        _ndjson_line({"message": {"content": ""}, "done": True}),
    ]))
    c = OllamaClient(
        model="my-model:7b",
        temperature=0.42,
        max_tokens=999,
        keep_alive_seconds=180,
    )
    async for _ in c.stream_chat([{"role": "user", "content": "hi"}]):
        pass
    body = mock_httpx.stream.call_args.kwargs["json"]
    assert body["model"] == "my-model:7b"
    assert body["stream"] is True
    assert body["keep_alive"] == "180s"
    assert body["options"]["temperature"] == 0.42
    assert body["options"]["num_predict"] == 999


async def test_system_prompt_prepended_when_set(mock_httpx):
    mock_httpx.stream = MagicMock(return_value=_MockStreamCM([
        _ndjson_line({"message": {"content": ""}, "done": True}),
    ]))
    c = OllamaClient(system_prompt="You are Jarvis.")
    async for _ in c.stream_chat([{"role": "user", "content": "hi"}]):
        pass
    sent = mock_httpx.stream.call_args.kwargs["json"]["messages"]
    assert sent[0] == {"role": "system", "content": "You are Jarvis."}
    assert sent[1] == {"role": "user", "content": "hi"}


async def test_system_prompt_not_duplicated_when_caller_provides_one(mock_httpx):
    mock_httpx.stream = MagicMock(return_value=_MockStreamCM([
        _ndjson_line({"message": {"content": ""}, "done": True}),
    ]))
    c = OllamaClient(system_prompt="Default prompt.")
    user_messages = [
        {"role": "system", "content": "Caller override."},
        {"role": "user", "content": "hi"},
    ]
    async for _ in c.stream_chat(user_messages):
        pass
    sent = mock_httpx.stream.call_args.kwargs["json"]["messages"]
    assert sent == user_messages


async def test_no_system_prompt_when_empty(mock_httpx):
    mock_httpx.stream = MagicMock(return_value=_MockStreamCM([
        _ndjson_line({"message": {"content": ""}, "done": True}),
    ]))
    c = OllamaClient(system_prompt="")
    user = [{"role": "user", "content": "hi"}]
    async for _ in c.stream_chat(user):
        pass
    sent = mock_httpx.stream.call_args.kwargs["json"]["messages"]
    assert sent == user


async def test_tools_included_in_body_when_provided(mock_httpx):
    mock_httpx.stream = MagicMock(return_value=_MockStreamCM([
        _ndjson_line({"message": {"content": ""}, "done": True}),
    ]))
    tools = [{"type": "function", "function": {"name": "screenshot", "parameters": {}}}]
    c = OllamaClient()
    async for _ in c.stream_chat([{"role": "user", "content": "hi"}], tools=tools):
        pass
    body = mock_httpx.stream.call_args.kwargs["json"]
    assert body["tools"] == tools


async def test_tools_omitted_when_none_or_empty(mock_httpx):
    mock_httpx.stream = MagicMock(return_value=_MockStreamCM([
        _ndjson_line({"message": {"content": ""}, "done": True}),
    ]))
    c = OllamaClient()
    async for _ in c.stream_chat([{"role": "user", "content": "hi"}], tools=None):
        pass
    body = mock_httpx.stream.call_args.kwargs["json"]
    assert "tools" not in body

    mock_httpx.stream = MagicMock(return_value=_MockStreamCM([
        _ndjson_line({"message": {"content": ""}, "done": True}),
    ]))
    async for _ in c.stream_chat([{"role": "user", "content": "hi"}], tools=[]):
        pass
    body = mock_httpx.stream.call_args.kwargs["json"]
    assert "tools" not in body


# --- error mapping ---------------------------------------------------


async def test_connection_refused_raises_clear_error(mock_httpx):
    mock_httpx.stream = MagicMock(
        return_value=_RaisingStreamCM(httpx.ConnectError("connection refused"))
    )
    c = OllamaClient()
    with pytest.raises(OllamaConnectionError, match="ollama serve"):
        async for _ in c.stream_chat([{"role": "user", "content": "hi"}]):
            pass


async def test_404_raises_model_not_found_with_pull_hint(mock_httpx):
    cm = _MockStreamCM(
        lines=[],
        status_code=404,
        body_text='{"error":"model not found"}',
    )
    mock_httpx.stream = MagicMock(return_value=cm)
    c = OllamaClient(model="ghost-model:1b")
    with pytest.raises(OllamaModelNotFoundError) as exc:
        async for _ in c.stream_chat([{"role": "user", "content": "hi"}]):
            pass
    assert "ghost-model:1b" in str(exc.value)
    assert "ollama pull" in str(exc.value)
    # Stream context still exited even on the early raise.
    assert cm.exited


async def test_other_http_error_raises_ollama_error(mock_httpx):
    cm = _MockStreamCM(lines=[], status_code=500)
    mock_httpx.stream = MagicMock(return_value=cm)
    c = OllamaClient()
    with pytest.raises(OllamaError, match="HTTP 500"):
        async for _ in c.stream_chat([{"role": "user", "content": "hi"}]):
            pass


# --- cancellation: the load-bearing test ----------------------------


async def test_cancellation_closes_underlying_stream(mock_httpx):
    """When the consumer task is cancelled mid-stream, the httpx stream
    context manager's __aexit__ MUST run -- this is what closes the
    connection and tells Ollama to stop generating. Verifying the
    consumer stopped iterating is NOT enough; we verify __aexit__
    actually fired with the cancellation exception."""
    lines = [
        _ndjson_line({"message": {"content": "Hello"}, "done": False}),
        # Test will cancel before the second line arrives.
    ]
    cm = _MockStreamCM(lines, between_sleep=10.0)
    mock_httpx.stream = MagicMock(return_value=cm)

    c = OllamaClient()
    received: list[ChatChunk] = []

    async def consume():
        async for chunk in c.stream_chat([{"role": "user", "content": "hi"}]):
            received.append(chunk)

    task = asyncio.create_task(consume())

    # Wait for at least one chunk so we know we're mid-stream.
    deadline = asyncio.get_running_loop().time() + 1.0
    while not received:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("never received first chunk")
        await asyncio.sleep(0.005)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cm.exited, (
        "stream context manager __aexit__ never ran; HTTP request was "
        "abandoned client-side instead of properly cancelled"
    )
    # The cancellation should have propagated as the exit exception type.
    assert cm.exit_exc_type is asyncio.CancelledError, (
        f"stream exited but with wrong exception type: {cm.exit_exc_type}"
    )


async def test_normal_completion_also_exits_stream_cleanly(mock_httpx):
    """Sanity: non-cancelled completion still exits the context manager
    (so we know the cancellation test isn't trivially passing)."""
    cm = _MockStreamCM([
        _ndjson_line({"message": {"content": "ok"}, "done": True}),
    ])
    mock_httpx.stream = MagicMock(return_value=cm)
    c = OllamaClient()
    async for _ in c.stream_chat([{"role": "user", "content": "hi"}]):
        pass
    assert cm.exited
    assert cm.exit_exc_type is None
