"""Factory for specialized VLM OCR runners."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dictextractor.ocr.vlm.glm_ocr import GlmOcrVlm
from dictextractor.ocr.vlm.mineru import MineruVlmOcr
from dictextractor.ocr.vlm.paddle_vl import PaddleVlOcr
from dictextractor.ocr.vlm.registry import VlmModelSpec, get_vlm_spec


class VlmOcrRunner(Protocol):
    """Protocol for page-level VLM OCR backends."""

    spec: VlmModelSpec

    def load(self) -> None: ...
    def run_page(self, image_path: Path, page_dir: Path, *, stem: str) -> dict[str, str]: ...
    def unload(self) -> None: ...

    @staticmethod
    def is_complete(page_dir: Path, *, stem: str) -> bool: ...


def create_vlm_runner(
    key: str,
    *,
    glm_prompt: str | None = None,
    glm_max_new_tokens: int | None = None,
) -> VlmOcrRunner:
    """Instantiate a runner for ``key`` (not loaded yet — call ``load()``)."""
    spec = get_vlm_spec(key)
    if key == "mineru2.5-pro":
        return MineruVlmOcr(spec)
    if key == "paddleocr-vl-1.5":
        return PaddleVlOcr(spec)
    if key == "glm-ocr":
        from dictextractor.ocr.vlm.glm_ocr import DEFAULT_MAX_NEW_TOKENS, DEFAULT_PROMPT

        return GlmOcrVlm(
            spec,
            prompt=glm_prompt or DEFAULT_PROMPT,
            max_new_tokens=glm_max_new_tokens or DEFAULT_MAX_NEW_TOKENS,
        )
    raise ValueError(f"No runner for {key!r}")


def page_is_complete(runner: VlmOcrRunner, page_dir: Path, *, stem: str) -> bool:
    """Return True if this page already has VLM output on disk."""
    if runner.spec.key == "paddleocr-vl-1.5":
        return PaddleVlOcr.is_complete(page_dir, stem=stem)
    if runner.spec.key == "glm-ocr":
        return GlmOcrVlm.is_complete(page_dir)
    return MineruVlmOcr.is_complete(page_dir)
