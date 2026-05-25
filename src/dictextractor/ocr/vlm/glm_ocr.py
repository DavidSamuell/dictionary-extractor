"""GLM-OCR VLM backend (Transformers or vLLM server)."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from dictextractor.ocr.vlm.completion import glm_page_has_content
from dictextractor.ocr.vlm.registry import VlmModelSpec

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Text Recognition:"
DEFAULT_MAX_NEW_TOKENS = 8192
DEFAULT_GLM_SERVED_NAME = "glm-ocr"

GlmBackend = Literal["transformers", "vllm"]


class GlmOcrVlm:
    """Run GLM-OCR via local Transformers or an OpenAI-compatible vLLM server."""

    def __init__(
        self,
        spec: VlmModelSpec,
        *,
        backend: GlmBackend = "transformers",
        server_url: str | None = None,
        served_model_name: str = DEFAULT_GLM_SERVED_NAME,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.spec = spec
        self.backend = backend
        self.server_url = server_url.rstrip("/") if server_url else None
        self.served_model_name = served_model_name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> None:
        if self.backend == "vllm":
            if not self.server_url:
                raise ValueError("GLM-OCR vLLM backend requires server_url")
            logger.info(
                "Using %s via vLLM server %s",
                self.spec.product_label,
                self.server_url,
            )
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        started = time.perf_counter()
        logger.info("Loading %s (transformers)...", self.spec.product_label)
        self._processor = AutoProcessor.from_pretrained(self.spec.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.spec.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
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
        user_prompt = prompt or self.prompt
        if self.backend == "vllm":
            output_text = self._run_page_vllm(image_path, user_prompt)
        else:
            output_text = self._run_page_transformers(image_path, user_prompt)
        return self._write_page_artifacts(page_dir, output_text, user_prompt)

    def _run_page_transformers(self, image_path: Path, user_prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path.resolve())},
                    {"type": "text", "text": user_prompt},
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
        return self._processor.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True,
        )

    def _run_page_vllm(self, image_path: Path, user_prompt: str) -> str:
        import httpx

        suffix = image_path.suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime};base64,{encoded}",
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
        }
        url = f"{self.server_url}/chat/completions"
        with httpx.Client(timeout=600.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"GLM-OCR vLLM returned no choices: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"GLM-OCR vLLM returned unexpected content: {message}")
        return content

    @staticmethod
    def _write_page_artifacts(
        page_dir: Path,
        output_text: str,
        user_prompt: str,
    ) -> dict[str, str]:
        md_path = page_dir / "output.md"
        txt_path = page_dir / "output.txt"
        result_path = page_dir / "result.json"
        md_path.write_text(output_text, encoding="utf-8")
        txt_path.write_text(output_text, encoding="utf-8")
        result_path.write_text(
            json.dumps({"text": output_text, "prompt": user_prompt}, ensure_ascii=False),
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
        if self.backend == "transformers":
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
