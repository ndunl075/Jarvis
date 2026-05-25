"""Tests for jarvis.tools.local.clipboard.ClipboardTool."""

from __future__ import annotations

from unittest.mock import patch

from jarvis.tools.local.clipboard import ClipboardTool
from jarvis.tools.registry import EmptyArgs


async def test_returns_clipboard_text_verbatim():
    with patch(
        "jarvis.tools.local.clipboard.winplat.read_clipboard_text",
        return_value="hello world",
    ):
        result = await ClipboardTool().execute(EmptyArgs())
    assert result.success
    assert result.output == "hello world"


async def test_empty_clipboard_returns_friendly_message():
    with patch(
        "jarvis.tools.local.clipboard.winplat.read_clipboard_text",
        return_value="",
    ):
        result = await ClipboardTool().execute(EmptyArgs())
    assert result.success
    assert "empty" in (result.output or "").lower()


async def test_long_text_summarised_not_dumped():
    """A multi-paragraph paste should not be read verbatim into TTS.
    Asserts the cap kicks in and the output references the length."""
    text = "x" * 5000
    with patch(
        "jarvis.tools.local.clipboard.winplat.read_clipboard_text",
        return_value=text,
    ):
        result = await ClipboardTool().execute(EmptyArgs())
    assert result.success
    out = result.output or ""
    assert "5000" in out
    assert len(out) < len(text)


async def test_os_error_surfaces_as_result_error():
    with patch(
        "jarvis.tools.local.clipboard.winplat.read_clipboard_text",
        side_effect=OSError("OpenClipboard failed"),
    ):
        result = await ClipboardTool().execute(EmptyArgs())
    assert not result.success
    assert "OpenClipboard" in (result.error or "")


def test_requires_confirmation_false():
    assert ClipboardTool().requires_confirmation is False
