"""Registry of specialized OCR/VLM models for stage-1 extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VlmModelSpec:
    """Metadata for a specialized OCR/VLM backend."""

    key: str
    experiment_name: str
    model_id: str
    product_label: str


VLM_MODELS: dict[str, VlmModelSpec] = {
    "mineru2.5-pro": VlmModelSpec(
        key="mineru2.5-pro",
        experiment_name="MinerU2.5-Pro",
        model_id="opendatalab/MinerU2.5-Pro-2604-1.2B",
        product_label="MinerU2.5-Pro",
    ),
    "paddleocr-vl-1.5": VlmModelSpec(
        key="paddleocr-vl-1.5",
        experiment_name="PaddleOCR-VL-1.5",
        model_id="PaddleOCR-VL-1.5",
        product_label="PaddleOCR-VL-1.5",
    ),
    "glm-ocr": VlmModelSpec(
        key="glm-ocr",
        experiment_name="GLM-OCR",
        model_id="zai-org/GLM-OCR",
        product_label="GLM-OCR",
    ),
}


def list_vlm_keys() -> list[str]:
    """Return registered CLI keys for ``--vlm-model``."""
    return sorted(VLM_MODELS.keys())


def get_vlm_spec(key: str) -> VlmModelSpec:
    """Look up a model spec by CLI key."""
    try:
        return VLM_MODELS[key]
    except KeyError as exc:
        valid = ", ".join(list_vlm_keys())
        raise ValueError(f"Unknown --vlm-model {key!r}. Choose from: {valid}") from exc
