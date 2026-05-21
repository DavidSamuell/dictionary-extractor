"""MinerU2.5-Pro inference runner."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from common import InferenceResult, timed_run, write_json

logger = logging.getLogger(__name__)

MODEL_ID = "opendatalab/MinerU2.5-Pro-2604-1.2B"


def run_mineru(
    pdf_path: Path,
    output_dir: Path,
    *,
    page_images: list[Path],
    backend: str = "transformers",
    use_cli: bool = False,
) -> InferenceResult:
    """Run MinerU2.5-Pro on a PDF or rendered page images."""
    if use_cli:
        return _run_mineru_cli(pdf_path, output_dir)
    if backend == "vllm":
        return _run_mineru_vllm(page_images, output_dir)
    return _run_mineru_transformers(page_images, output_dir)


def _run_mineru_cli(pdf_path: Path, output_dir: Path) -> InferenceResult:
    def _execute() -> dict[str, str]:
        cmd = ["mineru", "-p", str(pdf_path), "-o", str(output_dir)]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return {"cli_output_dir": str(output_dir)}

    return timed_run("mineru-cli", output_dir, _execute)


def _run_mineru_transformers(page_images: list[Path], output_dir: Path) -> InferenceResult:
    def _execute() -> dict[str, str]:
        from PIL import Image
        from mineru_vl_utils import MinerUClient
        from mineru_vl_utils.post_process import json2md
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        logger.info("Loading MinerU2.5-Pro (transformers backend)...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
        cfg = model.config
        if not hasattr(cfg, "max_position_embeddings"):
            text_cfg = getattr(cfg, "text_config", None)
            cfg.max_position_embeddings = (
                text_cfg.max_position_embeddings
                if text_cfg is not None and hasattr(text_cfg, "max_position_embeddings")
                else 8192
            )
        client = MinerUClient(
            backend="transformers",
            model=model,
            processor=processor,
            image_analysis=False,
        )

        artifacts: dict[str, str] = {}
        all_content: list[dict] = []

        for image_path in page_images:
            page_index = int(image_path.stem.split("_")[-1])
            page_dir = output_dir / f"page_{page_index:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)

            content_list = client.two_step_extract(Image.open(image_path))
            markdown = json2md(content_list)

            json_path = page_dir / "content.json"
            md_path = page_dir / "output.md"
            json_path.write_text(json.dumps(content_list, indent=2, ensure_ascii=False), encoding="utf-8")
            md_path.write_text(markdown, encoding="utf-8")

            all_content.append({"page": page_index, "content": content_list})
            artifacts[f"page_{page_index:04d}_markdown"] = str(md_path)

        write_json(output_dir / "all_pages.json", {"pages": all_content})
        artifacts["summary_json"] = str(output_dir / "all_pages.json")
        return artifacts

    return timed_run("mineru2.5-pro", output_dir, _execute)


def _run_mineru_vllm(page_images: list[Path], output_dir: Path) -> InferenceResult:
    def _execute() -> dict[str, str]:
        from PIL import Image
        from mineru_vl_utils import MinerUClient
        from mineru_vl_utils.post_process import json2md
        from vllm import LLM

        try:
            from mineru_vl_utils import MinerULogitsProcessor

            logits_processors = [MinerULogitsProcessor]
        except ImportError:
            logits_processors = None

        logger.info("Loading MinerU2.5-Pro (vLLM backend)...")
        llm_kwargs: dict = {"model": MODEL_ID}
        if logits_processors is not None:
            llm_kwargs["logits_processors"] = logits_processors
        llm = LLM(**llm_kwargs)

        client = MinerUClient(backend="vllm-engine", vllm_llm=llm, image_analysis=False)

        artifacts: dict[str, str] = {}
        all_content: list[dict] = []

        for image_path in page_images:
            page_index = int(image_path.stem.split("_")[-1])
            page_dir = output_dir / f"page_{page_index:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)

            content_list = client.two_step_extract(Image.open(image_path))
            markdown = json2md(content_list)

            json_path = page_dir / "content.json"
            md_path = page_dir / "output.md"
            json_path.write_text(json.dumps(content_list, indent=2, ensure_ascii=False), encoding="utf-8")
            md_path.write_text(markdown, encoding="utf-8")

            all_content.append({"page": page_index, "content": content_list})
            artifacts[f"page_{page_index:04d}_markdown"] = str(md_path)

        write_json(output_dir / "all_pages.json", {"pages": all_content})
        artifacts["summary_json"] = str(output_dir / "all_pages.json")
        return artifacts

    return timed_run("mineru2.5-pro-vllm", output_dir, _execute)
