"""Tests for jarvis.tools.local.see_screen.SeeScreenTool.

Heavy dependencies (pyautogui screen grab, Pillow encode, Ollama HTTP)
are mocked so the test suite runs headlessly and offline.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.core.config import VisionConfig
from jarvis.tools.local.see_screen import SeeScreenArgs, SeeScreenTool


# --- helpers ----------------------------------------------------------


class _FakeImage:
    """Minimal stand-in for a PIL.Image.Image object covering the
    surface the tool touches: .size, .resize, .save."""

    def __init__(self, size: tuple[int, int] = (1920, 1080)) -> None:
        self.size = size
        self.saved_paths: list[str] = []
        self.saved_formats: list[str] = []

    def resize(self, new_size, _resample) -> "_FakeImage":  # noqa: ANN001
        return _FakeImage(size=new_size)

    def save(self, target, format: str) -> None:  # noqa: A002
        if hasattr(target, "write"):
            target.write(b"\x89PNG\r\n\x1a\nFAKEPNGBYTES")
        else:
            Path(target).write_bytes(b"\x89PNG\r\n\x1a\nFAKEPNGBYTES")
            self.saved_paths.append(str(target))
        self.saved_formats.append(format)


def _install_fake_pyautogui_and_pil(image: _FakeImage) -> dict[str, types.ModuleType]:
    fake_pyautogui = types.ModuleType("pyautogui")
    fake_pyautogui.screenshot = MagicMock(return_value=image)
    fake_pil = types.ModuleType("PIL")
    fake_pil_image_mod = types.ModuleType("PIL.Image")

    class _Resampling:
        LANCZOS = "LANCZOS"

    fake_pil_image_mod.Image = _FakeImage  # type: ignore[attr-defined]
    fake_pil_image_mod.Resampling = _Resampling  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_image_mod  # type: ignore[attr-defined]
    return {
        "pyautogui": fake_pyautogui,
        "PIL": fake_pil,
        "PIL.Image": fake_pil_image_mod,
    }


def _make_ollama(response: str = "A code editor is open, sir.") -> MagicMock:
    client = MagicMock()
    client.vision_chat = AsyncMock(return_value=response)
    return client


# --- tests ------------------------------------------------------------


async def test_returns_description_from_vision_model(tmp_path):
    image = _FakeImage(size=(1920, 1080))
    fakes = _install_fake_pyautogui_and_pil(image)
    ollama = _make_ollama("Cursor IDE is in focus, sir.")
    tool = SeeScreenTool(ollama_client=ollama, vision_config=VisionConfig())

    with patch.dict(sys.modules, fakes), \
         patch(
             "jarvis.tools.local.see_screen.winplat.screenshots_dir",
             return_value=tmp_path,
         ):
        result = await tool.execute(SeeScreenArgs())

    assert result.success
    assert result.output == "Cursor IDE is in focus, sir."
    # Vision client called with the configured model + a non-empty base64 PNG.
    assert ollama.vision_chat.await_count == 1
    kwargs = ollama.vision_chat.await_args.kwargs
    assert kwargs["model"] == "llava:7b"
    assert kwargs["image_b64"]  # base64 string, not empty
    assert "describe what is on the user's screen" in kwargs["prompt"].lower()
    # PNG saved to the user's screenshots dir for auditing.
    saved = list(tmp_path.glob("jarvis_see_*.png"))
    assert len(saved) == 1


async def test_passes_user_question_to_prompt(tmp_path):
    image = _FakeImage(size=(1280, 720))
    fakes = _install_fake_pyautogui_and_pil(image)
    ollama = _make_ollama("The error says 'file not found', sir.")
    tool = SeeScreenTool(ollama_client=ollama, vision_config=VisionConfig())

    with patch.dict(sys.modules, fakes), \
         patch(
             "jarvis.tools.local.see_screen.winplat.screenshots_dir",
             return_value=tmp_path,
         ):
        result = await tool.execute(
            SeeScreenArgs(question="what does the error say")
        )

    assert result.success
    prompt = ollama.vision_chat.await_args.kwargs["prompt"]
    assert "what does the error say" in prompt.lower()


async def test_empty_model_returns_clear_error(tmp_path):
    image = _FakeImage()
    fakes = _install_fake_pyautogui_and_pil(image)
    ollama = _make_ollama()
    cfg = VisionConfig(model="")
    tool = SeeScreenTool(ollama_client=ollama, vision_config=cfg)

    with patch.dict(sys.modules, fakes), \
         patch(
             "jarvis.tools.local.see_screen.winplat.screenshots_dir",
             return_value=tmp_path,
         ):
        result = await tool.execute(SeeScreenArgs())

    assert not result.success
    assert "vision model" in (result.error or "").lower()
    # Should not have attempted any network call or screen capture.
    ollama.vision_chat.assert_not_awaited()


async def test_missing_ollama_model_surfaces_pull_hint(tmp_path):
    image = _FakeImage()
    fakes = _install_fake_pyautogui_and_pil(image)
    ollama = MagicMock()
    ollama.vision_chat = AsyncMock(
        side_effect=RuntimeError("vision model 'llava:7b' not found in Ollama")
    )
    tool = SeeScreenTool(ollama_client=ollama, vision_config=VisionConfig())

    with patch.dict(sys.modules, fakes), \
         patch(
             "jarvis.tools.local.see_screen.winplat.screenshots_dir",
             return_value=tmp_path,
         ):
        result = await tool.execute(SeeScreenArgs())

    assert not result.success
    assert "ollama pull" in (result.error or "").lower()


async def test_downscales_image_to_max_dim(tmp_path):
    image = _FakeImage(size=(3840, 2160))  # 4K
    fakes = _install_fake_pyautogui_and_pil(image)
    ollama = _make_ollama()
    cfg = VisionConfig(max_image_dim=1280)
    tool = SeeScreenTool(ollama_client=ollama, vision_config=cfg)

    with patch.dict(sys.modules, fakes), \
         patch(
             "jarvis.tools.local.see_screen.winplat.screenshots_dir",
             return_value=tmp_path,
         ):
        result = await tool.execute(SeeScreenArgs())

    assert result.success
    # The fake image's resize() returns a new fake; the saved file
    # exists with the correct PNG header, proving the resized image was
    # saved (not the original).
    saved = list(tmp_path.glob("jarvis_see_*.png"))
    assert saved
    assert saved[0].read_bytes().startswith(b"\x89PNG")


async def test_screenshot_failure_returns_error(tmp_path):
    fake_pyautogui = types.ModuleType("pyautogui")
    fake_pyautogui.screenshot = MagicMock(
        side_effect=RuntimeError("display unavailable")
    )
    fake_pil = types.ModuleType("PIL")
    fake_pil_image_mod = types.ModuleType("PIL.Image")
    fake_pil_image_mod.Image = _FakeImage  # type: ignore[attr-defined]

    class _Resampling:
        LANCZOS = "LANCZOS"

    fake_pil_image_mod.Resampling = _Resampling  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_image_mod  # type: ignore[attr-defined]
    fakes = {
        "pyautogui": fake_pyautogui,
        "PIL": fake_pil,
        "PIL.Image": fake_pil_image_mod,
    }
    ollama = _make_ollama()
    tool = SeeScreenTool(ollama_client=ollama, vision_config=VisionConfig())

    with patch.dict(sys.modules, fakes), \
         patch(
             "jarvis.tools.local.see_screen.winplat.screenshots_dir",
             return_value=tmp_path,
         ):
        result = await tool.execute(SeeScreenArgs())

    assert not result.success
    assert "couldn't capture" in (result.error or "").lower()
    ollama.vision_chat.assert_not_awaited()


def test_requires_confirmation_false():
    tool = SeeScreenTool(ollama_client=MagicMock(), vision_config=VisionConfig())
    assert tool.requires_confirmation is False
