"""Vision tool — capture the screen, ship it to a multimodal Ollama model,
speak the description back.

Pipeline
--------
1. Grab the primary display via pyautogui (same library as ScreenshotTool;
   late-imported because it pulls in mouse/keyboard hooks).
2. Resize to the configured long-edge cap (default 1280 px) with Pillow.
   Vision models do their own internal downscale; this cap just keeps the
   HTTP payload small and the prompt-eval time bounded.
3. Encode PNG → base64.
4. POST to Ollama's /api/chat with the vision model and the user's
   question (defaults to a generic "describe the screen" prompt).
5. Save the captured PNG to the user's screenshots folder so they have a
   reference for what Jarvis was looking at, then return the model's
   response as the spoken output.

Why a separate tool, not just adding images to the main LLM
----------------------------------------------------------
Vision-capable Ollama models (llava, moondream, minicpm-v, qwen2.5-vl)
are different from the conversational text model (`llm.model`). Letting
the main router pass images to qwen2.5:7b-instruct fails opaquely inside
Ollama. Exposing a dedicated `see_screen` tool with its own model
(VisionConfig.model) makes the swap explicit and gracefully degrades
when no vision model is pulled."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from datetime import datetime

from pydantic import BaseModel, Field

from jarvis.platform import windows as winplat
from jarvis.tools.registry import ToolResult

log = logging.getLogger(__name__)


_DEFAULT_PROMPT = (
    "You are Jarvis, an AI assistant. Describe what is on the user's "
    "screen briefly, in one or two sentences. Mention the active "
    "application, the visible content, and anything noteworthy "
    "(errors, dialogs, notifications). Address the user as 'sir'."
)


class SeeScreenArgs(BaseModel):
    question: str = Field(
        default="",
        description=(
            "Optional follow-up question about what's on screen. "
            "Examples: 'what does the error message say', 'is there a "
            "notification', 'summarize this article'. Leave blank for a "
            "generic description."
        ),
    )


class SeeScreenTool:
    name: str = "see_screen"
    description: str = (
        "Captures the user's screen and describes what's on it using a "
        "vision-capable LLM. Use this whenever the user asks you to see, "
        "look at, read, or describe their screen, or to identify what's "
        "visible (errors, dialogs, content of an article, what app is "
        "open, etc.). Optionally accepts a 'question' argument for "
        "follow-up queries about the screen contents."
    )
    args_schema: type[BaseModel] = SeeScreenArgs
    requires_confirmation: bool = False

    def __init__(
        self,
        *,
        ollama_client,
        vision_config,
    ) -> None:
        # `ollama_client` is the live OllamaClient instance. We use its
        # vision_chat() method (model name is per-call, not bound to the
        # client's text-model config).
        self._ollama = ollama_client
        self._cfg = vision_config

    async def execute(self, args: SeeScreenArgs) -> ToolResult:
        model = (getattr(self._cfg, "model", "") or "").strip()
        if not model:
            return ToolResult(
                success=False,
                error=(
                    "No vision model configured, sir. Please set one "
                    "under Settings → Models → Vision model."
                ),
            )

        try:
            image_b64, saved_path = await asyncio.to_thread(
                self._capture_and_encode,
                int(getattr(self._cfg, "max_image_dim", 1280)),
            )
        except Exception as exc:
            log.exception("see_screen: screen capture failed")
            return ToolResult(
                success=False,
                error=f"I couldn't capture the screen, sir: {exc}",
            )

        prompt = self._build_prompt(args.question)
        try:
            description = await self._ollama.vision_chat(
                model=model,
                prompt=prompt,
                image_b64=image_b64,
                max_tokens=int(getattr(self._cfg, "max_tokens", 512)),
                temperature=float(getattr(self._cfg, "temperature", 0.2)),
            )
        except Exception as exc:
            log.warning("see_screen: vision_chat failed: %s", exc)
            msg = str(exc)
            if "not found" in msg.lower():
                return ToolResult(
                    success=False,
                    error=(
                        f"The vision model {model!r} isn't pulled, sir. "
                        f"Run: ollama pull {model}"
                    ),
                )
            return ToolResult(
                success=False,
                error="I couldn't reach the vision model, sir.",
            )

        if not description:
            return ToolResult(
                success=True,
                output="I'm not sure what I'm looking at, sir.",
            )
        log.info("see_screen: snapshot saved to %s", saved_path)
        return ToolResult(success=True, output=description)

    # -- internal --

    def _build_prompt(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return _DEFAULT_PROMPT
        return (
            f"{_DEFAULT_PROMPT}\n\nUser's specific question: {q}\n"
            "Answer the question directly using only what you see in the "
            "image. If the answer isn't visible, say so."
        )

    @staticmethod
    def _capture_and_encode(max_dim: int) -> tuple[str, str]:
        """Capture the primary display, downscale long edge to `max_dim`,
        save a copy to the user's screenshots dir, and return
        (base64_png, saved_path).

        Runs on a worker thread so the audio loop never sees the
        synchronous pyautogui call.
        """
        # Late imports: pyautogui pulls in mouse/keyboard hooks at import
        # time; Pillow loads native codecs lazily. Both are cold-import
        # candidates if see_screen is never used.
        import pyautogui
        from PIL import Image

        image: Image.Image = pyautogui.screenshot()
        if image is None:
            raise RuntimeError("pyautogui.screenshot returned no image")

        # Long-edge resize. LANCZOS is the right choice for downscale —
        # NEAREST mangles UI text, BILINEAR is acceptable but softer.
        width, height = image.size
        long_edge = max(width, height)
        if long_edge > max_dim and long_edge > 0:
            scale = max_dim / float(long_edge)
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        target = winplat.screenshots_dir() / (
            datetime.now().strftime("jarvis_see_%Y%m%d_%H%M%S.png")
        )
        # Save a copy for the user (so they can audit what Jarvis saw),
        # then re-encode into memory for the API call. Saving via PNG
        # twice is the cleanest path — PIL's save() flow handles the
        # format-specific compression and metadata.
        image.save(target, format="PNG")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii"), str(target)
