"""Geometry-only layout → flat transcript parts (adapter v1)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from dictextractor.evaluation.stage1.normalize_typography import normalize_line
from dictextractor.ocr.adapters.blocks import LayoutBlock

logger = logging.getLogger(__name__)

ADAPTER_VERSION = "v1"

_HEADER_CATEGORIES = frozenset(
    {"header", "header_image", "title", "page_number"}
)
_FOOTER_CATEGORIES = frozenset(
    {"footer", "footer_image", "footnote", "table_footnote", "aside_text"}
)
# page_number in footer band when y > 0.85
_BODY_CATEGORIES = frozenset(
    {"text", "paragraph", "table", "list", "reference"}
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class FlatTranscriptParts:
    """Header, body, footer line lists (spec v2)."""

    header: list[str]
    body: list[str]
    footer: list[str]

    def all_lines(self) -> list[str]:
        return list(self.header) + list(self.body) + list(self.footer)


def _block_to_lines(text: str) -> list[str]:
    """Split block text into lines; expand HTML tables to one line per row."""
    if "<table" in text.lower() or "<tr" in text.lower():
        text = text.replace("</tr>", "\n</tr>")
        text = _HTML_TAG_RE.sub("", text)
    parts = [line.strip() for line in text.splitlines()]
    return [p for p in parts if p]


def _classify_block(block: LayoutBlock) -> str:
    cat = block.category.lower()
    if cat in _HEADER_CATEGORIES:
        return "header"
    if cat in _FOOTER_CATEGORIES:
        return "footer"
    if block.y_min >= 0.85 and cat in {"page_number", "text"}:
        return "footer"
    if block.y_min <= 0.12 and cat in {"page_number", "text", "title"}:
        return "header"
    if cat in _BODY_CATEGORIES or cat == "text":
        return "body"
    if cat in {"figure", "image", "seal", "chart"}:
        return "skip"
    return "body"


def _cluster_column_indices(
    blocks: list[LayoutBlock], *, max_columns: int = 3
) -> list[int]:
    """Assign 0..N-1 column ids from x-centers (gap-based, cap at max_columns)."""
    if len(blocks) <= 1:
        return [0] * len(blocks)
    centers = sorted(b.x_center for b in blocks)
    spread = centers[-1] - centers[0]
    if spread < 0.12:
        return [0] * len(blocks)

    gaps = [
        (centers[i + 1] - centers[i], i)
        for i in range(len(centers) - 1)
    ]
    gaps.sort(reverse=True)
    split_count = min(max_columns - 1, len(gaps))
    thresholds: list[float] = []
    for gap, idx in gaps[:split_count]:
        if gap < 0.08:
            break
        thresholds.append((centers[idx] + centers[idx + 1]) / 2.0)
    thresholds.sort()

    def col_id(x: float) -> int:
        c = 0
        for t in thresholds:
            if x > t:
                c += 1
        return min(c, max_columns - 1)

    return [col_id(b.x_center) for b in blocks]


def _serialize_body_column_major(blocks: list[LayoutBlock]) -> list[str]:
    if not blocks:
        return []
    col_ids = _cluster_column_indices(blocks)
    max_col = max(col_ids) if col_ids else 0
    columns: list[list[LayoutBlock]] = [[] for _ in range(max_col + 1)]
    for block, cid in zip(blocks, col_ids):
        columns[cid].append(block)

    lines: list[str] = []
    for col_blocks in columns:
        col_blocks.sort(key=lambda b: (b.y_min, b.x0))
        for block in col_blocks:
            for raw in _block_to_lines(block.text):
                line = normalize_line(raw)
                if line:
                    lines.append(line)
    return lines


def layout_to_transcript_v1(blocks: list[LayoutBlock]) -> FlatTranscriptParts:
    """
    Convert layout blocks to flat transcript parts (frozen v1 geometry).

    Header/footer blocks sorted by y; body column-major by x-clusters.
    """
    headers: list[LayoutBlock] = []
    footers: list[LayoutBlock] = []
    body: list[LayoutBlock] = []

    for block in blocks:
        role = _classify_block(block)
        if role == "header":
            headers.append(block)
        elif role == "footer":
            footers.append(block)
        elif role == "body":
            body.append(block)

    headers.sort(key=lambda b: (b.y_min, b.x0))
    footers.sort(key=lambda b: (b.y_min, b.x0))

    header_lines: list[str] = []
    for block in headers:
        for raw in _block_to_lines(block.text):
            line = normalize_line(raw)
            if line:
                header_lines.append(line)

    footer_lines: list[str] = []
    for block in footers:
        for raw in _block_to_lines(block.text):
            line = normalize_line(raw)
            if line:
                footer_lines.append(line)

    body_lines = _serialize_body_column_major(body)
    return FlatTranscriptParts(
        header=header_lines,
        body=body_lines,
        footer=footer_lines,
    )
