"""Read GLM-OCR text/md output (no layout boxes)."""

from __future__ import annotations

import json
from pathlib import Path

from dictextractor.evaluation.stage1.normalize_typography import normalize_line
from dictextractor.ocr.adapters.layout_to_transcript_v1 import FlatTranscriptParts


def _read_text_page(page_dir: Path) -> str:
    result_path = page_dir / "result.json"
    if result_path.is_file():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(data.get("text"), str):
            return data["text"]
    for name in ("output.md", "output.txt"):
        path = page_dir / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    for child in sorted(page_dir.iterdir()):
        if child.is_dir():
            try:
                return _read_text_page(child)
            except FileNotFoundError:
                continue
    raise FileNotFoundError(f"No GLM text output under {page_dir}")


def glm_transcript_from_page_dir(page_dir: Path) -> FlatTranscriptParts:
    """
    GLM fallback: split on newlines, no header/footer detection (body only).

    Reading order is model output order (top-to-bottom as emitted).
    """
    text = _read_text_page(page_dir)
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        line = normalize_line(raw)
        if line:
            lines.append(line)
    return FlatTranscriptParts(header=[], body=lines, footer=[])
