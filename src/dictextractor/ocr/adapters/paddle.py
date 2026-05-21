"""Read PaddleOCR-VL ``*_res.json`` into layout blocks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dictextractor.ocr.adapters.blocks import LayoutBlock

_SKIP_LABELS = frozenset({"figure", "image", "seal", "chart"})


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


def paddle_blocks_from_page_dir(page_dir: Path, *, stem: str) -> list[LayoutBlock]:
    """Load blocks from ``{stem}_res.json`` or any ``*_res.json`` in *page_dir*."""
    candidates = [
        page_dir / f"{stem}_res.json",
        *sorted(page_dir.glob("*_res.json")),
    ]
    for path in candidates:
        if path.is_file():
            return paddle_blocks_from_json(path)
    for child in sorted(page_dir.iterdir()):
        if child.is_dir():
            for path in sorted(child.glob("*_res.json")):
                return paddle_blocks_from_json(path)
    raise FileNotFoundError(f"No Paddle *_res.json under {page_dir}")
