"""Parse Mathpix ``lines.json`` into layout blocks for flat export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dictextractor.ocr.adapters.blocks import LayoutBlock
from dictextractor.ocr.adapters.layout_to_transcript_v1 import (
    FlatTranscriptParts,
    layout_to_transcript_v1,
)

logger = logging.getLogger(__name__)

_SKIP_LINE_TYPES = frozenset({"figure", "chart", "diagram"})


def _bbox_from_cnt(
    cnt: list[list[float]],
    *,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in cnt]
    ys = [float(p[1]) for p in cnt]
    w = max(page_width, 1.0)
    h = max(page_height, 1.0)
    return min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h


def _line_text(line: dict[str, Any]) -> str:
    for key in ("text", "text_display"):
        raw = line.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def blocks_from_lines_json(path: Path) -> list[LayoutBlock]:
    """Parse Mathpix PDF ``lines.json`` into normalized layout blocks."""
    data = json.loads(path.read_text(encoding="utf-8"))
    pages: list[dict[str, Any]] = data.get("pages") or []
    if not pages and isinstance(data.get("lines"), list):
        pages = [{"lines": data["lines"], "page_width": 1, "page_height": 1}]

    blocks: list[LayoutBlock] = []
    for page in pages:
        page_width = float(page.get("page_width") or 1)
        page_height = float(page.get("page_height") or 1)
        for line in page.get("lines") or []:
            line_type = str(line.get("type") or "text").lower()
            if line_type in _SKIP_LINE_TYPES:
                continue
            text = _line_text(line)
            if not text:
                continue
            cnt = line.get("cnt") or []
            if len(cnt) >= 2:
                x0, y0, x1, y1 = _bbox_from_cnt(
                    cnt,
                    page_width=page_width,
                    page_height=page_height,
                )
            else:
                x0, y0, x1, y1 = 0.0, 0.0, 1.0, 0.01
            blocks.append(
                LayoutBlock(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    text=text,
                    category=line_type,
                )
            )
    return blocks


def mathpix_transcript_from_lines_json(path: Path) -> FlatTranscriptParts:
    """Build flat transcript parts using geometry from Mathpix ``lines.json``."""
    blocks = blocks_from_lines_json(path)
    if not blocks:
        raise ValueError(f"No usable lines in {path}")
    return layout_to_transcript_v1(blocks)
