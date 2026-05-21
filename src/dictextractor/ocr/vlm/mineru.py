"""MinerU2.5-Pro VLM OCR backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dictextractor.ocr.vlm.completion import file_has_non_empty_text
from dictextractor.ocr.vlm.registry import VlmModelSpec

logger = logging.getLogger(__name__)


def _patch_qwen2vl_config_for_mineru(model: Any) -> None:
    """MinerU's transformers client reads ``config.max_position_embeddings``.

    Transformers 5.x exposes this on ``text_config`` for Qwen2VL, not the root
    config. Without the patch, MinerUClient fails at load time.
    """
    cfg = model.config
    if hasattr(cfg, "max_position_embeddings"):
        return
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "max_position_embeddings"):
        cfg.max_position_embeddings = text_cfg.max_position_embeddings
        logger.debug(
            "Patched max_position_embeddings=%s from text_config",
            cfg.max_position_embeddings,
        )
        return
    cfg.max_position_embeddings = 8192
    logger.warning(
        "Qwen2VL config missing max_position_embeddings; defaulting to 8192"
    )


class MineruVlmOcr:
    """Run MinerU2.5-Pro via transformers (one model load per process)."""

    def __init__(self, spec: VlmModelSpec) -> None:
        self.spec = spec
        self._client: Any = None
        self._json2md: Any = None

    def load(self) -> None:
        from PIL import Image  # noqa: F401 — ensure pillow available
        from mineru_vl_utils import MinerUClient
        from mineru_vl_utils.post_process import json2md
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        logger.info("Loading %s (transformers)...", self.spec.product_label)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.spec.model_id,
            dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(self.spec.model_id, use_fast=True)
        _patch_qwen2vl_config_for_mineru(model)
        self._client = MinerUClient(
            backend="transformers",
            model=model,
            processor=processor,
            image_analysis=False,
        )
        self._json2md = json2md

    def run_page(self, image_path: Path, page_dir: Path, *, stem: str) -> dict[str, str]:
        from PIL import Image

        page_dir.mkdir(parents=True, exist_ok=True)
        content_list = self._client.two_step_extract(Image.open(image_path))
        markdown = self._json2md(content_list)

        json_path = page_dir / "content.json"
        md_path = page_dir / "output.md"
        json_path.write_text(
            json.dumps(content_list, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path.write_text(markdown, encoding="utf-8")
        return {"content_json": str(json_path), "output_md": str(md_path)}

    def unload(self) -> None:
        self._client = None
        self._json2md = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def is_complete(page_dir: Path) -> bool:
        """True only when MinerU produced non-empty layout (not a failed/empty run)."""
        content_path = page_dir / "content.json"
        if content_path.is_file():
            try:
                data = json.loads(content_path.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    return True
            except json.JSONDecodeError:
                pass
        return file_has_non_empty_text(page_dir / "output.md")
