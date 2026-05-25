"""PaddleOCR-VL-1.5 document parser backend."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

from dictextractor.ocr.vlm.completion import paddle_page_has_content
from dictextractor.ocr.vlm.registry import VlmModelSpec

logger = logging.getLogger(__name__)

PaddleVlBackend = Literal["native", "vllm-server"]


def resolve_paddle_vl_backend(explicit: str | None = None) -> PaddleVlBackend:
    """Return Paddle VL recognition backend."""
    if explicit in ("native", "vllm-server"):
        return explicit
    env = os.getenv("PADDLE_VL_REC_BACKEND", "").strip().lower()
    if env == "vllm-server":
        return "vllm-server"
    return "native"


def resolve_paddle_vl_server_url(explicit: str | None = None) -> str | None:
    """Return Paddle vLLM server URL when using ``vllm-server`` backend."""
    if explicit:
        return explicit
    return os.getenv("PADDLE_VL_REC_SERVER_URL", "").strip() or None


class PaddleVlOcr:
    """Run PaddleOCR-VL-1.5 (one pipeline load per process)."""

    def __init__(
        self,
        spec: VlmModelSpec,
        *,
        vl_rec_backend: PaddleVlBackend | None = None,
        vl_rec_server_url: str | None = None,
    ) -> None:
        self.spec = spec
        self.vl_rec_backend = resolve_paddle_vl_backend(vl_rec_backend)
        self.vl_rec_server_url = resolve_paddle_vl_server_url(vl_rec_server_url)
        self._pipeline: Any = None

    def load(self) -> None:
        from paddleocr import PaddleOCRVL

        if self.vl_rec_backend == "vllm-server" and not self.vl_rec_server_url:
            raise ValueError(
                "Paddle vllm-server backend requires --paddle-vl-rec-server-url "
                "or PADDLE_VL_REC_SERVER_URL"
            )

        started = time.perf_counter()
        if self.vl_rec_backend == "vllm-server":
            logger.info(
                "Loading %s (vl_rec_backend=vllm-server, server=%s)...",
                self.spec.product_label,
                self.vl_rec_server_url,
            )
            self._pipeline = PaddleOCRVL(
                vl_rec_backend="vllm-server",
                vl_rec_server_url=self.vl_rec_server_url,
            )
        else:
            logger.info(
                "Loading %s (native backend; first run may download weights)...",
                self.spec.product_label,
            )
            self._pipeline = PaddleOCRVL()
        logger.info(
            "Loaded %s in %.1fs",
            self.spec.product_label,
            time.perf_counter() - started,
        )

    def run_page(
        self,
        image_path: Path,
        page_dir: Path,
        *,
        stem: str,
        prompt: str | None = None,
    ) -> dict[str, str]:
        page_dir.mkdir(parents=True, exist_ok=True)
        results = list(self._pipeline.predict(str(image_path)))
        artifacts: dict[str, str] = {}
        for result in results:
            result.save_to_json(save_path=str(page_dir))
            result.save_to_markdown(save_path=str(page_dir))
            result.save_to_img(save_path=str(page_dir))
        artifacts["page_dir"] = str(page_dir)
        res_json = page_dir / f"{stem}_res.json"
        if res_json.is_file():
            artifacts["res_json"] = str(res_json)
        return artifacts

    def unload(self) -> None:
        self._pipeline = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def is_complete(page_dir: Path, *, stem: str) -> bool:
        """True only when ``*_res.json`` has at least one non-empty block."""
        return paddle_page_has_content(page_dir, stem=stem)
