"""Tests for stage-1 flat spec v2 flattening."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dictextractor.evaluation.stage1.flatten import (
    FLAT_SPEC_VERSION,
    flat_transcription_to_text,
    flatten_stage1_body_rows,
    flatten_stage1_rows,
    flatten_stage1_tsv,
)

_SAMPLES = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "dictionaries"
    / "samples"
)
_SHILLUK = _SAMPLES / "Shilluk-English/outputs/stage-1-gold/page_28/page_28_stage1_GOLD.tsv"
_CIRCASSIAN = (
    _SAMPLES / "Circassian-English-Turkish/outputs/stage-1-gold/page_2/page_2_stage1_GOLD.tsv"
)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@pytest.mark.skipif(not _SHILLUK.is_file(), reason="sample gold missing")
def test_flat_spec_version() -> None:
    assert FLAT_SPEC_VERSION == "v2"


@pytest.mark.skipif(not _SHILLUK.is_file(), reason="sample gold missing")
def test_shilluk_header_footer_preserved() -> None:
    rows = _load_rows(_SHILLUK)
    flat = flatten_stage1_rows(rows)
    assert flat[0].startswith("Root - shk-flex")
    assert flat[-1] == "198 of 256 20/06/2023, 11:23"
    assert sum(1 for r in rows if r["column_id"] == "header") == 2
    assert sum(1 for r in rows if r["column_id"] == "footer") == 1


@pytest.mark.skipif(not _CIRCASSIAN.is_file(), reason="sample gold missing")
def test_circassian_row_major_body() -> None:
    rows = _load_rows(_CIRCASSIAN)
    body = flatten_stage1_body_rows(rows, language="Circassian-English-Turkish")
    assert body[0] == "ii"
    assert body[1] == "<b>ENGLISH.</b>"
    assert body[2] == "<b>CIRCASSIAN.</b>"
    assert body[3] == "<b>TURKISH.</b>"
    assert body[4] == "After, <i>prep.</i>"
    assert body[5] == "یِتانِه yeytáhney"
    assert body[6] == "كوره ــ اوزره"
    assert body[67] == "Alone, <i>a.</i>"
    assert "یالكز" in body
    left_count = sum(1 for r in rows if r["column_id"] == "left")
    center_count = sum(1 for r in rows if r["column_id"] == "center")
    right_count = sum(1 for r in rows if r["column_id"] == "right")
    assert len(body) == left_count + center_count + right_count


@pytest.mark.skipif(not _CIRCASSIAN.is_file(), reason="sample gold missing")
def test_circassian_tsv_path_uses_row_major() -> None:
    rows = _load_rows(_CIRCASSIAN)
    headers = [r["text"] for r in rows if r["column_id"] == "header"]
    footers = [r["text"] for r in rows if r["column_id"] == "footer"]
    body = flatten_stage1_body_rows(rows, language="Circassian-English-Turkish")
    assert flatten_stage1_tsv(_CIRCASSIAN) == flat_transcription_to_text(
        headers, body, footers
    )
