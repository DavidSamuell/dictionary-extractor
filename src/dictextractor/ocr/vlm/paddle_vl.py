"""PaddleOCR-VL-1.5 document parser backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dictextractor.ocr.vlm.completion import paddle_page_has_content
from dictextractor.ocr.vlm.registry import VlmModelSpec

logger = logging.getLogger(__name__)


class PaddleVlOcr:
    """Run PaddleOCR-VL-1.5 (one pipeline load per process)."""

    def __init__(self, spec: VlmModelSpec) -> None:
        self.spec = spec
        self._pipeline: Any = None

    def load(self) -> None:
        from paddleocr import PaddleOCRVL

        logger.info("Loading %s...", self.spec.product_label)
        self._pipeline = PaddleOCRVL()

    def run_page(self, image_path: Path, page_dir: Path, *, stem: str) -> dict[str, str]:
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
