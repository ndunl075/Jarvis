"""Report CPU and RAM utilisation as a one-line spoken summary."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from jarvis.tools.registry import EmptyArgs, ToolResult


class SystemStatsTool:
    # Long action-specific name on purpose: when this tool was called
    # `system_stats`, a small LLM treated it as a generic "tell me about
    # the system" probe and fired it for ambiguous transcriptions
    # ("jot", "in chat", "capital of france"). The verbose name signals
    # to the model that this is a narrow numeric tool, not a fallback.
    name: str = "report_cpu_and_memory_percentages"
    description: str = (
        "Reports the current CPU usage percentage and memory usage "
        "percentage. Only use this when the user explicitly asks for "
        "CPU, memory, RAM, or system performance numbers. Never use "
        "this for general questions, greetings, conversations, or "
        "factual queries."
    )
    args_schema: type[BaseModel] = EmptyArgs
    requires_confirmation: bool = False

    async def execute(self, args: EmptyArgs) -> ToolResult:
        def _sample() -> tuple[float, float]:
            import psutil
            # interval=0.2: a single instantaneous sample on Windows often
            # reads 0% because Windows aggregates per-tick. A short
            # interval gives a meaningful number without noticeable lag.
            cpu = psutil.cpu_percent(interval=0.2)
            mem = psutil.virtual_memory().percent
            return cpu, mem

        try:
            cpu, mem = await asyncio.to_thread(_sample)
        except Exception as e:
            return ToolResult(success=False, error=f"could not read stats: {e}")
        return ToolResult(
            success=True,
            output=f"CPU at {cpu:.0f} percent, memory at {mem:.0f} percent, sir.",
        )
