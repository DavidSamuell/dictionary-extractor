"""Write ``*_stage1_flat.txt`` from OCR page directories."""

from __future__ import annotations

import csv
import logging
from enum import Enum
from pathlib import Path

from dictextractor.evaluation.stage1.flatten import (
    _metadata_lines,
    flat_output_path_for_pred,
    flatten_stage1_body_rows,
    flatten_stage1_tsv,
    write_flat_text,
)
from dictextractor.ocr.adapters.glm import glm_transcript_from_page_dir
from dictextractor.ocr.adapters.layout_to_transcript_v1 import (
    FlatTranscriptParts,
    layout_to_transcript_v1,
)
from dictextractor.ocr.adapters.mineru import mineru_blocks_from_page_dir
from dictextractor.ocr.adapters.paddle import paddle_blocks_from_page_dir

logger = logging.getLogger(__name__)


class OcrBackend(str, Enum):
    MINERU = "mineru"
    PADDLE = "paddle"
    GLM = "glm"
    COLUMN_TSV = "column_tsv"


def detect_backend(page_dir: Path, *, stem: str) -> OcrBackend | None:
    """Detect OCR backend artifacts under *page_dir*."""
    if (page_dir / "content.json").is_file() or any(
        (c / "content.json").is_file() for c in page_dir.iterdir() if c.is_dir()
    ):
        return OcrBackend.MINERU
    if list(page_dir.glob("*_res.json")) or any(
        list(c.glob("*_res.json")) for c in page_dir.iterdir() if c.is_dir()
    ):
        return OcrBackend.PADDLE
    if (page_dir / "result.json").is_file() or (page_dir / "output.txt").is_file():
        return OcrBackend.GLM
    tsv = page_dir / f"{stem}_stage1.tsv"
    if tsv.is_file():
        return OcrBackend.COLUMN_TSV
    return None


def page_dir_to_flat_parts(page_dir: Path, *, stem: str) -> FlatTranscriptParts:
    """Build flat transcript parts from a page output directory."""
    backend = detect_backend(page_dir, stem=stem)
    if backend is None:
        raise FileNotFoundError(f"No recognized OCR artifacts in {page_dir}")

    if backend is OcrBackend.MINERU:
        blocks = mineru_blocks_from_page_dir(page_dir)
        return layout_to_transcript_v1(blocks)
    if backend is OcrBackend.PADDLE:
        blocks = paddle_blocks_from_page_dir(page_dir, stem=stem)
        return layout_to_transcript_v1(blocks)
    if backend is OcrBackend.GLM:
        return glm_transcript_from_page_dir(page_dir)
    if backend is OcrBackend.COLUMN_TSV:
        tsv_path = page_dir / f"{stem}_stage1.tsv"
        with tsv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        return FlatTranscriptParts(
            header=_metadata_lines(rows, "header"),
            body=flatten_stage1_body_rows(rows),
            footer=_metadata_lines(rows, "footer"),
        )
    raise ValueError(f"Unsupported backend {backend}")


def page_dir_to_flat_lines(page_dir: Path, *, stem: str) -> list[str]:
    """Return spec-v2 flat lines for a page directory."""
    return page_dir_to_flat_parts(page_dir, stem=stem).all_lines()


def write_stage1_flat_for_page(page_dir: Path, *, stem: str) -> Path:
    """Write ``{stem}_stage1_flat.txt`` beside OCR artifacts."""
    parts = page_dir_to_flat_parts(page_dir, stem=stem)
    out = flat_output_path_for_pred(page_dir, stem)
    write_flat_text(out, parts.all_lines())
    logger.debug("Wrote %s (%d lines)", out, len(parts.all_lines()))
    return out
