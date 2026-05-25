"""Tests for Stage-1 transcript path resolution."""

from pathlib import Path

from dictextractor.utils.stage1_input import (
    read_stage1_transcript_text,
    resolve_stage1_gold_transcript_path,
    resolve_stage1_transcript_for_stage2,
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


def test_gold_auto_prefers_tsv(tmp_path: Path) -> None:
    page = tmp_path / "page_3"
    page.mkdir()
    tsv = page / "page_3_stage1_GOLD.tsv"
    flat = page / "page_3_stage1_GOLD_flat.txt"
    tsv.write_text("column_id\tline_number\ttext\n", encoding="utf-8")
    flat.write_text("flat line\n", encoding="utf-8")
    resolved = resolve_stage1_gold_transcript_path(page, "page_3", "auto")
    assert resolved == tsv
    assert stage1_transcript_kind(resolved) == "column"


def test_gold_flat_only(tmp_path: Path) -> None:
    page = tmp_path / "page_3"
    page.mkdir()
    flat = page / "page_3_stage1_GOLD_flat.txt"
    flat.write_text("flat line\n", encoding="utf-8")
    resolved = resolve_stage1_gold_transcript_path(page, "page_3", "flat")
    assert resolved == flat
    assert read_stage1_transcript_text(resolved) == "flat line\n"


def test_stage2_predictions_flat(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    page = output_dir / "stage-1" / "GLM-OCR-vllm_flat_alpha" / "page_14"
    page.mkdir(parents=True)
    flat = page / "page_14_stage1_flat.txt"
    flat.write_text("glm flat\n", encoding="utf-8")
    resolved = resolve_stage1_transcript_for_stage2(
        output_dir,
        "page_14",
        "flat",
        source="predictions",
        experiment_name="GLM-OCR-vllm_flat_alpha",
    )
    assert resolved == flat
