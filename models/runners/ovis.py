"""Ovis2.6-30B-A3B inference runner."""

from __future__ import annotations

import logging
from pathlib import Path

from common import InferenceResult, timed_run

logger = logging.getLogger(__name__)

MODEL_ID = "AIDC-AI/Ovis2.6-30B-A3B"
DEFAULT_PROMPT = (
    "Transcribe all visible text from this document page. "
    "Preserve structure where possible. "
    "End your response with 'Final answer: '."
)


def run_ovis(
    output_dir: Path,
    *,
    page_images: list[Path],
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 2048,
    thinking_budget: int = 1024,
    enable_thinking: bool = True,
) -> InferenceResult:
    """Run Ovis2.6-30B-A3B on rendered page images."""
    return timed_run(
        "ovis2.6-30b-a3b",
        output_dir,
        lambda: _execute_ovis(
            output_dir=output_dir,
            page_images=page_images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            thinking_budget=thinking_budget,
            enable_thinking=enable_thinking,
        ),
    )


def _execute_ovis(
    *,
    output_dir: Path,
    page_images: list[Path],
    prompt: str,
    max_new_tokens: int,
    thinking_budget: int,
    enable_thinking: bool,
) -> dict[str, str]:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM

    logger.info("Loading Ovis2.6-30B-A3B (this may take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    artifacts: dict[str, str] = {}
    for image_path in page_images:
        page_index = int(image_path.stem.split("_")[-1])
        page_dir = output_dir / f"page_{page_index:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }]

        input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        pixel_values = pixel_values.to(device) if pixel_values is not None else None
        grid_thws = grid_thws.to(device) if grid_thws is not None else None

        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                enable_thinking=enable_thinking,
                enable_thinking_budget=enable_thinking,
                max_new_tokens=max_new_tokens,
                thinking_budget=thinking_budget,
            )

        response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_path = page_dir / "output.txt"
        output_path.write_text(response, encoding="utf-8")
        artifacts[f"page_{page_index:04d}_text"] = str(output_path)
        logger.info("Finished Ovis inference for page %d", page_index)

    return artifacts
