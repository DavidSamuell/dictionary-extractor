"""Tests for specialized OCR VLM layout adapters."""

import json
from pathlib import Path

from dictextractor.ocr.adapters.mineru import mineru_blocks_from_json
from dictextractor.ocr.adapters.paddle import paddle_blocks_from_json


def test_mineru_skips_phonetic_blocks(tmp_path: Path) -> None:
    payload = [
        {
            "type": "text",
            "bbox": [0.1, 0.2, 0.5, 0.3],
            "content": "dictionary line",
        },
        {
            "type": "phonetic",
            "bbox": [0.1, 0.4, 0.9, 0.8],
            "content": "x<i>{1}</i> x<i>{2}</i>",
        },
    ]
    path = tmp_path / "content.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    blocks = mineru_blocks_from_json(path)
    assert len(blocks) == 1
    assert blocks[0].text == "dictionary line"


def test_paddle_parses_body_blocks(tmp_path: Path) -> None:
    payload = {
        "width": 1000,
        "height": 2000,
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "entry one",
                "block_bbox": [10, 100, 500, 150],
            },
            {
                "block_label": "header",
                "block_content": "wheel",
                "block_bbox": [10, 10, 100, 40],
            },
        ],
    }
    path = tmp_path / "page_res.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    blocks = paddle_blocks_from_json(path)
    assert len(blocks) == 2
    assert any(b.text == "entry one" for b in blocks)
