"""GLM-OCR VLM backend (Hugging Face Transformers)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dictextractor.ocr.vlm.completion import glm_page_has_content
from dictextractor.ocr.vlm.registry import VlmModelSpec

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Text Recognition:"
DEFAULT_MAX_NEW_TOKENS = 8192


class GlmOcrVlm:
    """Run GLM-OCR via ``AutoModelForImageTextToText``."""

    def __init__(
        self,
        spec: VlmModelSpec,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.spec = spec
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading %s (transformers)...", self.spec.product_label)
        self._processor = AutoProcessor.from_pretrained(self.spec.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.spec.model_id,
            torch_dtype="auto",
            device_map="auto",
        )

    def run_page(self, image_path: Path, page_dir: Path, *, stem: str) -> dict[str, str]:
        page_dir.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path.resolve())},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        inputs.pop("token_type_ids", None)

        generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_len = inputs["input_ids"].shape[1]
        output_text = self._processor.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True,
        )

        md_path = page_dir / "output.md"
        txt_path = page_dir / "output.txt"
        result_path = page_dir / "result.json"
        md_path.write_text(output_text, encoding="utf-8")
        txt_path.write_text(output_text, encoding="utf-8")
        result_path.write_text(
            json.dumps({"text": output_text, "prompt": self.prompt}, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "output_md": str(md_path),
            "output_txt": str(txt_path),
            "result_json": str(result_path),
        }

    def unload(self) -> None:
        self._processor = None
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def is_complete(page_dir: Path) -> bool:
        """True only when GLM produced non-empty transcript text."""
        return glm_page_has_content(page_dir)
