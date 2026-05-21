"""GLM-OCR inference runner (Hugging Face Transformers).

See: https://huggingface.co/zai-org/GLM-OCR
"""

from __future__ import annotations

import logging
from pathlib import Path

from common import InferenceResult, timed_run, write_json

logger = logging.getLogger(__name__)

MODEL_ID = "zai-org/GLM-OCR"
DEFAULT_PROMPT = "Text Recognition:"


def run_glmocr(
    pdf_path: Path,
    output_dir: Path,
    *,
    page_images: list[Path],
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 8192,
    use_full_pdf: bool = False,
) -> InferenceResult:
    """Run GLM-OCR locally via ``transformers`` (``AutoModelForImageTextToText``)."""
    if use_full_pdf and pdf_path.suffix.lower() == ".pdf":
        logger.info(
            "GLM-OCR transformers backend processes rendered page images; "
            "ignoring native full-PDF flag."
        )

    return timed_run(
        "glm-ocr",
        output_dir,
        lambda: _execute_glmocr_transformers(
            output_dir=output_dir,
            page_images=page_images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        ),
    )


def _build_messages(image_path: Path, prompt: str) -> list[dict]:
    """Build chat messages for GLM-OCR document parsing."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _execute_glmocr_transformers(
    *,
    output_dir: Path,
    page_images: list[Path],
    prompt: str,
    max_new_tokens: int,
) -> dict[str, str]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info("Loading GLM-OCR (transformers backend)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    artifacts: dict[str, str] = {}

    for image_path in page_images:
        page_index = int(image_path.stem.split("_")[-1])
        page_dir = output_dir / f"page_{page_index:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        logger.info("GLM-OCR inference for page %d (%s)", page_index, image_path.name)
        messages = _build_messages(image_path, prompt)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True,
        )

        md_path = page_dir / "output.md"
        txt_path = page_dir / "output.txt"
        md_path.write_text(output_text, encoding="utf-8")
        txt_path.write_text(output_text, encoding="utf-8")
        write_json(page_dir / "result.json", {"text": output_text, "prompt": prompt})

        artifacts[f"page_{page_index:04d}_markdown"] = str(md_path)
        artifacts[f"page_{page_index:04d}_text"] = str(txt_path)
        logger.info("Finished GLM-OCR for page %d (%d chars)", page_index, len(output_text))

    return artifacts
