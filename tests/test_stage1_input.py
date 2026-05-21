"""Tests for Stage-1 transcript path resolution."""

from pathlib import Path

from dictextractor.utils.stage1_input import (
    resolve_stage1_transcript_path,
    stage1_transcript_kind,
)


def test_auto_prefers_tsv(tmp_path: Path) -> None:
    page = tmp_path / "page_1"
    page.mkdir()
    tsv = page / "page_1_stage1.tsv"
    flat = page / "page_1_stage1_flat.txt"
    tsv.write_text("col\ttext\n", encoding="utf-8")
    flat.write_text("flat line\n", encoding="utf-8")
    resolved = resolve_stage1_transcript_path(page, "page_1", "auto")
    assert resolved == tsv
    assert stage1_transcript_kind(resolved) == "column"


def test_auto_falls_back_to_flat(tmp_path: Path) -> None:
    page = tmp_path / "page_1"
    page.mkdir()
    flat = page / "page_1_stage1_flat.txt"
    flat.write_text("flat line\n", encoding="utf-8")
    resolved = resolve_stage1_transcript_path(page, "page_1", "auto")
    assert resolved == flat
    assert stage1_transcript_kind(resolved) == "flat"


def test_flat_only(tmp_path: Path) -> None:
    page = tmp_path / "page_1"
    page.mkdir()
    (page / "page_1_stage1.tsv").write_text("tsv\n", encoding="utf-8")
    assert resolve_stage1_transcript_path(page, "page_1", "flat") is None
