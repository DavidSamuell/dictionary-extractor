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
    def run_page(
        self,
        image_path: Path,
        page_dir: Path,
        *,
        stem: str,
        prompt: str | None = None,
    ) -> dict[str, str]: ...
    def unload(self) -> None: ...

    @staticmethod
    def is_complete(page_dir: Path, *, stem: str) -> bool: ...


def create_vlm_runner(
    key: str,
    *,
    glm_prompt: str | None = None,
    glm_max_new_tokens: int | None = None,
    mineru_backend: str | None = None,
    mineru_batch_size: int | None = None,
    mineru_max_new_tokens: int | None = None,
    glm_backend: str | None = None,
    glm_vllm_server_url: str | None = None,
    paddle_vl_rec_backend: str | None = None,
    paddle_vl_rec_server_url: str | None = None,
) -> VlmOcrRunner:
    """Instantiate a runner for ``key`` (not loaded yet — call ``load()``)."""
    spec = get_vlm_spec(key)
    if key == "mineru2.5-pro":
        from dictextractor.ocr.vlm.mineru import DEFAULT_MINERU_MAX_NEW_TOKENS

        return MineruVlmOcr(
            spec,
            backend=mineru_backend,  # type: ignore[arg-type]
            batch_size=mineru_batch_size,
            max_new_tokens=mineru_max_new_tokens or DEFAULT_MINERU_MAX_NEW_TOKENS,
        )
    if key == "paddleocr-vl-1.5":
        return PaddleVlOcr(
            spec,
            vl_rec_backend=paddle_vl_rec_backend,  # type: ignore[arg-type]
            vl_rec_server_url=paddle_vl_rec_server_url,
        )
    if key == "glm-ocr":
        from dictextractor.ocr.vlm.glm_ocr import DEFAULT_MAX_NEW_TOKENS, DEFAULT_PROMPT

        backend = glm_backend or "transformers"
        if backend == "vllm" and not glm_vllm_server_url:
            raise ValueError(
                "GLM-OCR vLLM backend requires glm_vllm_server_url or auto-start"
            )
        return GlmOcrVlm(
            spec,
            backend=backend,  # type: ignore[arg-type]
            server_url=glm_vllm_server_url,
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
