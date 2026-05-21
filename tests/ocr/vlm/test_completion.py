"""Tests for VLM OCR non-empty completion checks."""

from __future__ import annotations

import json
from pathlib import Path

from dictextractor.ocr.vlm.completion import (
    glm_page_has_content,
    paddle_page_has_content,
)
from dictextractor.ocr.vlm.glm_ocr import GlmOcrVlm
from dictextractor.ocr.vlm.paddle_vl import PaddleVlOcr


def test_paddle_empty_res_json_not_complete(tmp_path: Path) -> None:
    page_dir = tmp_path / "page_1"
    page_dir.mkdir()
    (page_dir / "page_1_res.json").write_text(
        json.dumps({"parsing_res_list": []}), encoding="utf-8"
    )
    assert not paddle_page_has_content(page_dir, stem="page_1")
    assert not PaddleVlOcr.is_complete(page_dir, stem="page_1")


def test_paddle_nonempty_block_is_complete(tmp_path: Path) -> None:
    page_dir = tmp_path / "page_1"
    page_dir.mkdir()
    (page_dir / "page_1_res.json").write_text(
        json.dumps(
            {
                "parsing_res_list": [
                    {"block_content": "hello", "block_bbox": [0, 0, 1, 1]}
                ]
            }
        ),
        encoding="utf-8",
    )
    assert paddle_page_has_content(page_dir, stem="page_1")


def test_glm_empty_output_not_complete(tmp_path: Path) -> None:
    page_dir = tmp_path / "page_1"
    page_dir.mkdir()
    (page_dir / "output.txt").write_text("", encoding="utf-8")
    (page_dir / "result.json").write_text(json.dumps({"text": ""}), encoding="utf-8")
    assert not glm_page_has_content(page_dir)
    assert not GlmOcrVlm.is_complete(page_dir)


def test_glm_nonempty_output_is_complete(tmp_path: Path) -> None:
    page_dir = tmp_path / "page_1"
    page_dir.mkdir()
    (page_dir / "output.txt").write_text("line one\n", encoding="utf-8")
    assert glm_page_has_content(page_dir)
