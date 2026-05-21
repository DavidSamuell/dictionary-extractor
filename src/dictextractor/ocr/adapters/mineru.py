"""Read MinerU ``content.json`` into layout blocks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dictextractor.ocr.adapters.blocks import LayoutBlock

_SKIP_TYPES = frozenset({"figure", "image", "equation", "phonetic"})


def mineru_blocks_from_json(path: Path) -> list[LayoutBlock]:
    """Parse MinerU content list (bbox normalized 0–1)."""
    raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    blocks: list[LayoutBlock] = []
    for item in raw:
        bbox = item.get("bbox") or [0, 0, 1, 1]
        if len(bbox) < 4:
            continue
        category = str(item.get("type") or "text").lower()
        if category in _SKIP_TYPES:
            continue
        text = str(item.get("content") or "").strip()
        if not text:
            continue
        blocks.append(
            LayoutBlock(
                x0=float(bbox[0]),
                y0=float(bbox[1]),
                x1=float(bbox[2]),
                y1=float(bbox[3]),
                text=text,
                category=category,
            )
        )
    return blocks


def mineru_blocks_from_page_dir(page_dir: Path) -> list[LayoutBlock]:
    """Load blocks from ``content.json`` under *page_dir* or a child folder."""
    path = page_dir / "content.json"
    if not path.is_file():
        for child in sorted(page_dir.iterdir()):
            if child.is_dir() and (child / "content.json").is_file():
                path = child / "content.json"
                break
    if not path.is_file():
        raise FileNotFoundError(f"No MinerU content.json under {page_dir}")
    return mineru_blocks_from_json(path)
