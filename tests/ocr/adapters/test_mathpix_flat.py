"""Tests for Mathpix lines.json and flat adapters."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dictextractor.ocr.adapters.mathpix_flat import (
    mathpix_docx_path,
    mathpix_lines_path,
    mathpix_transcript_from_docx,
    mathpix_transcript_from_page_dir,
)
from dictextractor.ocr.adapters.mathpix_lines import mathpix_transcript_from_lines_json
from dictextractor.ocr.adapters.paddle import paddle_blocks_from_page_dir

_SAMPLES = Path(__file__).resolve().parents[3] / "assets/dictionaries/samples"
_CIRCASSIAN_PAGE_1 = _SAMPLES / "Circassian-English-Turkish/mathpix/page_1.docx"
_CANALA_PAGE_12 = _SAMPLES / "Canala-English/mathpix/page_12.docx"
_MALAY_PADDLE = (
    _SAMPLES
    / "Malay-English/outputs/stage-1/PaddleOCR-VL-1.5/page_315/page_315_res.json"
)
_MALAY_GOLD = (
    _SAMPLES
    / "Malay-English/outputs/stage-1-gold/page_315/page_315_stage1_GOLD_flat.txt"
)

def _paddle_res_to_lines_json(res_json: Path) -> dict:
    """Build a Mathpix-like lines.json from Paddle layout (for adapter testing)."""
    page_dir = res_json.parent
    blocks = paddle_blocks_from_page_dir(page_dir, stem=res_json.stem.replace("_res", ""))
    lines = []
    for idx, block in enumerate(blocks):
        x0, y0, x1, y1 = block.x0, block.y0, block.x1, block.y1
        lines.append(
            {
                "line": idx,
                "text": block.text,
                "type": block.category,
                "conversion_output": True,
                "cnt": [
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1],
                ],
            }
        )
    return {
        "pages": [
            {
                "page": 1,
                "page_width": 1,
                "page_height": 1,
                "lines": lines,
            }
        ]
    }


@pytest.mark.skipif(not _CIRCASSIAN_PAGE_1.is_file(), reason="sample docx missing")
def test_mathpix_table_uses_column_major_order() -> None:
    parts = mathpix_transcript_from_docx(_CIRCASSIAN_PAGE_1)
    lines = parts.all_lines()

    assert lines[0] == "English."
    circ_idx = next(i for i, line in enumerate(lines) if line.startswith("Circassian"))
    turk_idx = next(
        i for i, line in enumerate(lines) if line.startswith("Turrish")
    )
    assert circ_idx > 5
    assert turk_idx > circ_idx


@pytest.mark.skipif(not _CANALA_PAGE_12.is_file(), reason="sample docx missing")
def test_mathpix_paragraphs_preserve_reading_order() -> None:
    parts = mathpix_transcript_from_docx(_CANALA_PAGE_12)
    lines = parts.all_lines()

    assert len(lines) >= 40
    assert lines[0].startswith("ak")


def test_mathpix_lines_json_two_column_geometry(tmp_path: Path) -> None:
    payload = {
        "pages": [
            {
                "page": 1,
                "page_width": 1000,
                "page_height": 2000,
                "lines": [
                    {
                        "text": "left-top",
                        "type": "text",
                        "conversion_output": True,
                        "cnt": [[50, 300], [400, 300], [400, 320], [50, 320]],
                    },
                    {
                        "text": "right-top",
                        "type": "text",
                        "conversion_output": True,
                        "cnt": [[550, 300], [900, 300], [900, 320], [550, 320]],
                    },
                    {
                        "text": "left-bottom",
                        "type": "text",
                        "conversion_output": True,
                        "cnt": [[50, 700], [400, 700], [400, 720], [50, 720]],
                    },
                    {
                        "text": "right-bottom",
                        "type": "text",
                        "conversion_output": True,
                        "cnt": [[550, 700], [900, 700], [900, 720], [550, 720]],
                    },
                ],
            }
        ]
    }
    path = tmp_path / "page.lines.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    lines = mathpix_transcript_from_lines_json(path).all_lines()
    assert lines.index("left-top") < lines.index("left-bottom")
    assert lines.index("left-bottom") < lines.index("right-top")


@pytest.mark.skipif(not _MALAY_PADDLE.is_file(), reason="Paddle output missing")
@pytest.mark.skipif(not _MALAY_GOLD.is_file(), reason="gold missing")
def test_lines_json_from_paddle_proxy_produces_flat_transcript(tmp_path: Path) -> None:
    """Paddle-derived lines.json should parse and yield a non-trivial flat transcript."""
    payload = _paddle_res_to_lines_json(_MALAY_PADDLE)
    lines_json_path = tmp_path / "page_315.lines.json"
    lines_json_path.write_text(json.dumps(payload), encoding="utf-8")
    lines = mathpix_transcript_from_lines_json(lines_json_path).all_lines()

    gold_lines = [
        line
        for line in _MALAY_GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) >= 20
    assert 0.3 <= len(lines) / len(gold_lines) <= 3.0


@pytest.mark.skipif(not _CIRCASSIAN_PAGE_1.is_file(), reason="sample docx missing")
def test_page_dir_prefers_lines_json_when_present(tmp_path: Path) -> None:
    page_dir = tmp_path / "page_1"
    page_dir.mkdir()
    shutil.copy2(_CIRCASSIAN_PAGE_1, mathpix_docx_path(page_dir))

    docx_lines = mathpix_transcript_from_page_dir(page_dir).all_lines()

    payload = {
        "pages": [
            {
                "page": 1,
                "page_width": 1000,
                "page_height": 2000,
                "lines": [
                    {
                        "text": "geometry-only-line",
                        "type": "text",
                        "conversion_output": True,
                        "cnt": [[10, 10], [900, 10], [900, 30], [10, 30]],
                    }
                ],
            }
        ]
    }
    mathpix_lines_path(page_dir).write_text(json.dumps(payload), encoding="utf-8")

    lines = mathpix_transcript_from_page_dir(page_dir).all_lines()
    assert lines != docx_lines
    assert lines == ["geometry-only-line"]
