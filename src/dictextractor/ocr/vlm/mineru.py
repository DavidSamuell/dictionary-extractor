"""MinerU2.5-Pro VLM OCR backend."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

from dictextractor.ocr.vlm.completion import file_has_non_empty_text
from dictextractor.ocr.vlm.registry import VlmModelSpec

logger = logging.getLogger(__name__)

MineruBackend = Literal["transformers", "vllm"]

# MinerU runs layout detection + one forward pass per layout block (~50 on
# dictionary pages). batch_size=1 (library default) serializes those passes.
DEFAULT_MINERU_BATCH_SIZE = 8
# Without max_new_tokens, transformers falls back to max_length=8192 per block,
# which can run for minutes on confused crops (Georgian/Cyrillic dictionary pages).
DEFAULT_MINERU_MAX_NEW_TOKENS = 1024
DEFAULT_MINERU_LAYOUT_MAX_NEW_TOKENS = 4096


def resolve_mineru_backend(explicit: str | None = None) -> MineruBackend:
    """Return MinerU inference backend (transformers or vllm)."""
    if explicit in ("transformers", "vllm"):
        return explicit
    env = os.getenv("MINERU_VL_BACKEND", os.getenv("VLM_BACKEND", "")).strip().lower()
    if env in ("transformers", "vllm"):
        return env  # type: ignore[return-value]
    return "transformers"


def resolve_mineru_batch_size(explicit: int | None = None) -> int:
    """Return GPU batch size for MinerU block extraction."""
    if explicit is not None and explicit > 0:
        return explicit
    env = os.getenv("MINERU_VL_BATCH_SIZE", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return DEFAULT_MINERU_BATCH_SIZE


def build_mineru_sampling_params(
    max_new_tokens: int = DEFAULT_MINERU_MAX_NEW_TOKENS,
    *,
    layout_max_new_tokens: int = DEFAULT_MINERU_LAYOUT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Copy MinerU defaults but cap generation length per block."""
    from mineru_vl_utils.mineru_client import DEFAULT_SAMPLING_PARAMS, MinerUSamplingParams

    capped: dict[str, Any] = {}
    for key, sp in DEFAULT_SAMPLING_PARAMS.items():
        cap = layout_max_new_tokens if key == "[layout]" else max_new_tokens
        capped[key] = MinerUSamplingParams(
            temperature=sp.temperature,
            top_p=sp.top_p,
            top_k=sp.top_k,
            presence_penalty=sp.presence_penalty,
            frequency_penalty=sp.frequency_penalty,
            repetition_penalty=sp.repetition_penalty,
            no_repeat_ngram_size=sp.no_repeat_ngram_size,
            max_new_tokens=cap,
        )
    return capped


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
    """Run MinerU2.5-Pro via transformers or vLLM (one model load per process)."""

    def __init__(
        self,
        spec: VlmModelSpec,
        *,
        backend: MineruBackend | None = None,
        batch_size: int | None = None,
        max_new_tokens: int = DEFAULT_MINERU_MAX_NEW_TOKENS,
    ) -> None:
        self.spec = spec
        self.backend = resolve_mineru_backend(backend)
        self.batch_size = resolve_mineru_batch_size(batch_size)
        self.max_new_tokens = max_new_tokens
        self._client: Any = None
        self._json2md: Any = None

    def _build_client(self, sampling_params: dict[str, Any]) -> Any:
        from mineru_vl_utils import MinerUClient

        common = {
            "image_analysis": False,
            "batch_size": self.batch_size,
            "sampling_params": sampling_params,
        }
        if self.backend == "vllm":
            return self._build_vllm_client(MinerUClient, common)
        return self._build_transformers_client(MinerUClient, common)

    def _build_transformers_client(self, mineru_client: Any, common: dict[str, Any]) -> Any:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.spec.model_id,
            dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(self.spec.model_id, use_fast=True)
        _patch_qwen2vl_config_for_mineru(model)
        return mineru_client(
            backend="transformers",
            model=model,
            processor=processor,
            **common,
        )

    def _build_vllm_client(self, mineru_client: Any, common: dict[str, Any]) -> Any:
        from vllm import LLM

        try:
            from mineru_vl_utils import MinerULogitsProcessor

            logits_processors: list[type] | None = [MinerULogitsProcessor]
        except ImportError:
            logits_processors = None

        llm_kwargs: dict[str, Any] = {
            "model": self.spec.model_id,
            "trust_remote_code": True,
        }
        if logits_processors is not None:
            llm_kwargs["logits_processors"] = logits_processors
        llm = LLM(**llm_kwargs)
        return mineru_client(backend="vllm-engine", vllm_llm=llm, **common)

    def load(self) -> None:
        from PIL import Image  # noqa: F401 — ensure pillow available
        from mineru_vl_utils.post_process import json2md

        started = time.perf_counter()
        logger.info(
            "Loading %s (%s backend)...",
            self.spec.product_label,
            self.backend,
        )
        sampling_params = build_mineru_sampling_params(self.max_new_tokens)
        self._client = self._build_client(sampling_params)
        self._json2md = json2md
        logger.info(
            "Loaded %s in %.1fs (backend=%s, batch_size=%d, max_new_tokens=%d)",
            self.spec.product_label,
            time.perf_counter() - started,
            self.backend,
            self.batch_size,
            self.max_new_tokens,
        )

    def run_page(
        self,
        image_path: Path,
        page_dir: Path,
        *,
        stem: str,
        prompt: str | None = None,
    ) -> dict[str, str]:
        from PIL import Image

        page_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with Image.open(image_path) as image:
            content_list = self._client.two_step_extract(image)
        block_count = len(content_list) if isinstance(content_list, list) else 0
        logger.info(
            "MinerU two_step_extract %s: %d blocks in %.1fs",
            image_path.name,
            block_count,
            time.perf_counter() - started,
        )
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
