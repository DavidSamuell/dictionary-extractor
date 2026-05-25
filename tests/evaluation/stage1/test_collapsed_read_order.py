"""Tests for collapse-aligned read order."""

from dictextractor.evaluation.stage1.read_order import compute_read_order_collapsed


def test_perfect_order_zero() -> None:
    lines = ["alpha", "beta", "gamma"]
    ro = compute_read_order_collapsed(lines, lines)
    assert ro.read_order_edit == 0.0


def test_swapped_lines_penalized() -> None:
    gold = ["one", "two", "three"]
    pred = ["three", "one", "two"]
    ro = compute_read_order_collapsed(gold, pred)
    assert ro.read_order_edit > 0.0


def test_merged_pred_lines_can_still_anchor() -> None:
    gold = ["hello", "world"]
    pred = ["hello world"]
    ro = compute_read_order_collapsed(gold, pred)
    assert ro.read_order_edit == 0.0
