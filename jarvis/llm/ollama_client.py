"""Async streaming client for Ollama with Loadable lifecycle for VRAM eviction.

Implements core.lifecycle.Loadable per the Phase 1 design:
  - load() is a no-op. SPEC § Lifecycle Contract: "LLM loads lazily on
    first use" -- the first inference request triggers Ollama's daemon
    to pull weights into VRAM.
  - unload() POSTs to /api/generate with keep_alive=0, the documented
    Ollama eviction signal: drop the model from VRAM. This is the
    SLEEPING-mode payoff and is the entire point of treating Ollama as
    a Loadable.

Streaming
---------
stream_chat() yields ChatChunk values. Each chunk carries either
incremental text content (most cases) or, on the final chunk,
done=True plus optionally tool_calls. The Phase 4 intent router
consumes this iterator and either pipes content chunks to TTS or
extracts the tool_calls list to dispatch a ToolIntent.

Cancellation contract (the load-bearing one)
--------------------------------------------
When the consumer task is cancelled (e.g., the audio pipeline cancels
_response_task on barge-in), CancelledError propagates up through the
async generator into the httpx stream's async-context-manager scope.
__aexit__ closes the response and the underlying TCP connection -- the
HTTP request is terminated, not just abandoned client-side. Ollama
sees the disconnect and stops generating. This satisfies the
ResponseProducer cancellation contract documented in
audio/protocols.py for Phase 3.

Timeouts
--------
We use httpx.Timeout(connect=10, read=None, write=10, pool=10):
  - connect bounded (10 s) so a dead daemon fails fast.
  - read=None (unlimited) because an LLM stream can take arbitrarily
    long, especially on the first inference after model load (cold
    weights → can be 30+ seconds before the first token).
  - write/pool bounded for symmetry.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT: str = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Base for Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Could not reach the Ollama daemon."""


class OllamaModelNotFoundError(OllamaError):
    """The configured model is not pulled in Ollama."""


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """One streaming chunk from a chat completion.

    Most chunks carry incremental `content`. The final chunk has
    `done=True` and may carry `tool_calls` (Ollama emits tool calls
    as a single batch on the final message rather than streamed)."""

    content: str = ""
    tool_calls: tuple[dict, ...] = ()
    done: bool = False


def _default_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)


class OllamaClient:
    name: str = "ollama_client"

    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str = "",
        keep_alive_seconds: int = 300,
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.keep_alive_seconds = keep_alive_seconds
        self.endpoint = endpoint.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self.is_loaded: bool = False

    # -- Loadable --

    async def load(self) -> None:
        # No-op per SPEC § Lifecycle Contract: LLM loads lazily on first
        # inference. We mark is_loaded so the LifecycleManager's
        # idempotency check skips redundant calls.
        self.is_loaded = True

    async def warm(self) -> None:
        """Send a minimal non-streaming chat request so Ollama loads the
        model into VRAM at startup rather than on the user's first real
        turn. Cold-load of a 7B int4/int8 model can be 15-45 s on CPU
        or moderate GPUs; warming amortizes that during the loopback /
        app boot when latency is already expected.

        Uses /api/chat with messages=[user:"ready"], stream=False,
        num_predict=1. Honors the configured keep_alive so the model
        stays resident through the boot sequence.

        Raises the same error classes as stream_chat. Callers may wrap
        in try/except to make warmup non-fatal (loopback does this --
        a missing model is not a reason to block startup of everything
        else)."""
        client = self._get_client()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ready"}],
            "stream": False,
            "keep_alive": f"{self.keep_alive_seconds}s",
            "options": {"num_predict": 1},
        }
        try:
            response = await client.post("/api/chat", json=body)
            if response.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"model {self.model!r} not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"could not connect to Ollama at {self.endpoint}. "
                "Is the daemon running? Start it with: ollama serve"
            ) from e
        except httpx.HTTPStatusError as e:
            raise OllamaError(
                f"ollama returned HTTP {e.response.status_code}"
            ) from e

    async def unload(self) -> None:
        # Always attempt eviction even if we never streamed -- the daemon
        # may have the model loaded from a prior Jarvis run with a
        # keep_alive that hasn't expired. This is the SLEEPING-mode
        # payoff per SPEC.
        client = self._get_client()
        try:
            await client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": "",
                    "keep_alive": 0,
                },
            )
        except Exception:
            # Daemon may be down; log + continue. SLEEPING must always
            # make progress (Phase 1 lifecycle policy).
            log.warning(
                "ollama eviction request failed (daemon may be down)",
                exc_info=True,
            )
        try:
            await client.aclose()
        except Exception:
            log.exception("ollama client close failed")
        self._client = None
        self.is_loaded = False

    # -- streaming chat --

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Stream a chat completion. Yields ChatChunk values.

        Args:
          messages: chat history in OpenAI/Ollama format.
          tools: optional list of tool schemas (Phase 4 will populate
            from the tool registry's as_openai_functions() output;
            Phase 3 callers pass [] or None).
        """
        client = self._get_client()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._with_system_prompt(messages),
            "stream": True,
            "keep_alive": f"{self.keep_alive_seconds}s",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if tools:
            body["tools"] = tools

        try:
            async with client.stream("POST", "/api/chat", json=body) as response:
                if response.status_code == 404:
                    # Read the body so we can surface the Ollama error message.
                    try:
                        await response.aread()
                        detail = response.text[:300]
                    except Exception:
                        detail = ""
                    raise OllamaModelNotFoundError(
                        f"model {self.model!r} not found in Ollama. "
                        f"Run: ollama pull {self.model}"
                        + (f"  (server said: {detail})" if detail else "")
                    )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("ollama: skipping non-JSON line: %r", line[:200])
                        continue
                    chunk = self._chunk_from(data)
                    yield chunk
                    if chunk.done:
                        return
        except GeneratorExit:
            # GC-driven aclose() of this async generator (e.g. the consumer
            # iterator was discarded on Ctrl+C without an explicit aclose).
            # Re-raising plain GeneratorExit lets httpx's stream context
            # manager run its own cleanup synchronously; trying to do anything
            # async-y here triggers "asynchronous generator is already running"
            # because the loop is mid-shutdown and athrow() races with the
            # GC's concurrent aclose() of the same generator.
            raise
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"could not connect to Ollama at {self.endpoint}. "
                "Is the daemon running? Start it with: ollama serve"
            ) from e
        except httpx.HTTPStatusError as e:
            raise OllamaError(
                f"ollama returned HTTP {e.response.status_code}"
            ) from e

    # -- internal --

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                timeout=_default_timeout(),
            )
        return self._client

    def _with_system_prompt(self, messages: list[dict]) -> list[dict]:
        """Prepend the configured system prompt unless the caller already
        supplied one (we don't override caller intent)."""
        if not self.system_prompt:
            return messages
        if messages and messages[0].get("role") == "system":
            return messages
        return [{"role": "system", "content": self.system_prompt}, *messages]

    def _chunk_from(self, data: dict) -> ChatChunk:
        msg = data.get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or ()
        return ChatChunk(
            content=msg.get("content", "") or "",
            tool_calls=tuple(tool_calls),
            done=bool(data.get("done", False)),
        )
