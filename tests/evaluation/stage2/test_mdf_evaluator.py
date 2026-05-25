"""Tests for Stage 2 MDF evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from dictextractor.evaluation.stage2.mdf_evaluator import MdfEvaluator, DEFAULT_LINE_THRESHOLD, DEFAULT_RECORD_THRESHOLD
from dictextractor.evaluation.stage2.mdf_marker_equiv import markers_equivalent
from dictextractor.evaluation.stage2.mdf_parser import parse_mdf

ROOT = Path(__file__).resolve().parents[3]
SAMPLES = ROOT / "assets" / "dictionaries" / "samples"


def test_default_thresholds() -> None:
    evaluator = MdfEvaluator()
    assert evaluator.record_threshold == DEFAULT_RECORD_THRESHOLD == 0.6
    assert evaluator.line_threshold == DEFAULT_LINE_THRESHOLD == 0.7


def test_marker_sub_list_groups() -> None:
    assert markers_equivalent("gn", "dn")
    assert markers_equivalent("de", "ge")
    assert not markers_equivalent("gn", "ge")
    assert not markers_equivalent("gn", "un")
    assert not markers_equivalent("dl", "un")


def test_parse_mdf_blank_line_records() -> None:
    text = "\\lx alpha\n\\gn gloss one\n\n\\lx beta\n\\gn gloss two\n"
    records = parse_mdf(text)
    assert len(records) == 2
    assert records[0].headword == "alpha"
    assert records[1].headword == "beta"


@pytest.mark.parametrize(
    ("language", "page"),
    [
        ("Chukchi-Russian", "page_3"),
        ("Evenki-Russian", "page_1"),
    ],
)
def test_gold_vs_gold_perfect(language: str, page: str) -> None:
    gold_path = SAMPLES / language / "outputs" / "stage-2-gold" / page / f"{page}.mdf.txt"
    if not gold_path.is_file():
        pytest.skip(f"missing gold: {gold_path}")
    evaluator = MdfEvaluator()
    metrics = evaluator.evaluate(gold_path, gold_path, page_id=f"{language}/{page}")
    assert metrics.record_accuracy == 1.0
    assert metrics.mdf_fields_f1 == 1.0
    assert metrics.read_order.read_order_edit == 0.0


@pytest.mark.parametrize(
    ("language", "page"),
    [
        ("Chukchi-Russian", "page_3"),
        ("Evenki-Russian", "page_1"),
    ],
)
def test_pro_high_mdf_benchmark_smoke(language: str, page: str) -> None:
    gold_path = SAMPLES / language / "outputs" / "stage-2-gold" / page / f"{page}.mdf.txt"
    pred_path = SAMPLES / language / "outputs" / "stage-2" / "gemini31pro_high_mdf_intro_notoolbox" / page / f"{page}.mdf.txt"
    if not gold_path.is_file() or not pred_path.is_file():
        pytest.skip("benchmark assets missing")
    metrics = MdfEvaluator().evaluate(pred_path, gold_path, page_id=f"{language}/{page}")
    assert metrics.record.tp > 0
    if language == "Chukchi-Russian":
        assert metrics.record_accuracy == 1.0
        assert metrics.mdf_fields_f1 >= 0.9
    if language == "Evenki-Russian":
        assert metrics.record_accuracy >= 0.9
