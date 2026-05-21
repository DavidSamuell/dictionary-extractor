"""Page-collapse alignment for eval-flat character/typography metrics."""

from dictextractor.evaluation.stage1.alignment import (
    align_page_collapsed,
    align_rows,
    collapse_rows_to_page,
)
from dictextractor.evaluation.stage1.character_quality import compute_character_quality
from dictextractor.evaluation.stage1.flat_evaluator import FlatStage1Evaluator, _lines_to_rows
from dictextractor.evaluation.stage1.read_order import compute_read_order


def _row(text: str) -> dict[str, str]:
    return {"column_id": "single", "line_number": "1", "text": text}


def test_collapse_joins_lines_with_space() -> None:
    rows = [_row("alpha"), _row("beta")]
    page = collapse_rows_to_page(rows)
    assert len(page) == 1
    assert page[0]["text"] == "alpha beta"


def test_page_collapse_ignores_line_split_for_gcer() -> None:
    gold = _lines_to_rows(["hello", "world"])
    pred = _lines_to_rows(["hello world"])
    page = align_page_collapsed(pred, gold)
    cq = compute_character_quality(page)
    assert cq.gcer == 0.0
    assert cq.matched_spans == 1


def test_line_alignment_still_differs_for_read_order() -> None:
    gold = _lines_to_rows(["hello", "world"])
    pred = _lines_to_rows(["hello world"])
    line = align_rows(pred, gold, threshold=0.5, max_span_rows=3)
    ro = compute_read_order(line)
    assert ro.read_order_edit > 0.0


def test_flat_evaluator_uses_both_alignments(tmp_path) -> None:
    gold_path = tmp_path / "page_stage1_GOLD_flat.txt"
    pred_path = tmp_path / "page_stage1_flat.txt"
    gold_path.write_text("one\ntwo\n", encoding="utf-8")
    pred_path.write_text("one two\n", encoding="utf-8")
    m = FlatStage1Evaluator().evaluate(pred_path, gold_path, page_id="test/page")
    assert m.character_quality.gcer == 0.0
    assert m.read_order.read_order_edit > 0.0
