"""Read PaddleOCR-VL ``*_res.json`` into layout blocks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dictextractor.ocr.adapters.blocks import LayoutBlock

logger = logging.getLogger(__name__)

_SKIP_LABELS = frozenset({"figure", "image", "seal", "chart"})
_BODY_LABELS = frozenset({"text", "paragraph", "table", "list", "reference", "title"})


def _normalize_bbox(
    bbox: list[float], width: float, height: float
) -> tuple[float, float, float, float]:
    w = max(width, 1.0)
    h = max(height, 1.0)
    return bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h


def paddle_blocks_from_json(path: Path) -> list[LayoutBlock]:
    """Parse Paddle ``parsing_res_list`` (pixel bbox → normalized)."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    width = float(data.get("width") or 1)
    height = float(data.get("height") or 1)
    blocks: list[LayoutBlock] = []
    for item in data.get("parsing_res_list") or []:
        label = str(item.get("block_label") or "text").lower()
        if label in _SKIP_LABELS:
            continue
        bbox = item.get("block_bbox") or [0, 0, width, height]
        if len(bbox) < 4:
            continue
        text = str(item.get("block_content") or "").strip()
        if not text:
            continue
        x0, y0, x1, y1 = _normalize_bbox(
            [float(v) for v in bbox[:4]], width, height
        )
        blocks.append(
            LayoutBlock(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                text=text,
                category=label,
            )
        )
    return blocks


def _warn_if_no_body_blocks(blocks: list[LayoutBlock], *, source: Path) -> None:
    """Log when Paddle JSON has no dictionary body blocks (header/footer only)."""
    body_count = sum(1 for b in blocks if b.category.lower() in _BODY_LABELS)
    if body_count == 0:
        labels = sorted({b.category for b in blocks})
        logger.warning(
            "Paddle OCR at %s: no body blocks in parsing_res_list "
            "(%d block(s), labels=%s)",
            source,
            len(blocks),
            labels or "none",
        )


def paddle_blocks_from_page_dir(page_dir: Path, *, stem: str) -> list[LayoutBlock]:
    """Load blocks from ``{stem}_res.json`` or any ``*_res.json`` in *page_dir*."""
    candidates = [
        page_dir / f"{stem}_res.json",
        *sorted(page_dir.glob("*_res.json")),
    ]
    for path in candidates:
        if path.is_file():
            blocks = paddle_blocks_from_json(path)
            _warn_if_no_body_blocks(blocks, source=path)
            return blocks
    for child in sorted(page_dir.iterdir()):
        if child.is_dir():
            for path in sorted(child.glob("*_res.json")):
                blocks = paddle_blocks_from_json(path)
                _warn_if_no_body_blocks(blocks, source=path)
                return blocks
    raise FileNotFoundError(f"No Paddle *_res.json under {page_dir}")
