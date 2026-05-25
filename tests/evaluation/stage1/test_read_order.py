"""Read order metric for eval-flat."""

from dictextractor.evaluation.stage1.alignment import align_lines_quick_match
from dictextractor.evaluation.stage1.flat_evaluator import FlatStage1Evaluator, _lines_to_rows
from dictextractor.evaluation.stage1.read_order import compute_read_order


def test_perfect_order_scores_zero() -> None:
    gold = _lines_to_rows(["alpha", "beta", "gamma"])
    pred = _lines_to_rows(["alpha", "beta", "gamma"])
    alignment = align_lines_quick_match(pred, gold)
    ro = compute_read_order(alignment)
    assert ro.read_order_edit == 0.0


def test_swapped_lines_penalized() -> None:
    gold = _lines_to_rows(["hello", "world"])
    pred = _lines_to_rows(["world", "hello"])
    alignment = align_lines_quick_match(pred, gold)
    ro = compute_read_order(alignment)
    assert ro.read_order_edit > 0.0


def test_missing_gold_line_penalized() -> None:
    gold = _lines_to_rows(["a", "b", "c"])
    pred = _lines_to_rows(["a", "c"])
    alignment = align_lines_quick_match(pred, gold)
    ro = compute_read_order(alignment)
    assert ro.read_order_edit > 0.0


def test_merged_pred_line_matches_multiple_gold_lines() -> None:
    """quick_match allows many gold lines → one merged pred line."""
    gold = _lines_to_rows(["hello", "world"])
    pred = _lines_to_rows(["hello world"])
    alignment = align_lines_quick_match(pred, gold)
    ro = compute_read_order(alignment)
    assert ro.read_order_edit == 0.0


def test_flat_evaluator_collapsed_penalizes_swapped_lines(tmp_path) -> None:
    gold_path = tmp_path / "page_stage1_GOLD_flat.txt"
    pred_path = tmp_path / "page_stage1_flat.txt"
    gold_path.write_text("one\ntwo\n", encoding="utf-8")
    pred_path.write_text("two\none\n", encoding="utf-8")
    m = FlatStage1Evaluator(character_alignment="collapsed").evaluate(
        pred_path, gold_path, page_id="test/page"
    )
    assert m.character_quality.gcer > 0.0
