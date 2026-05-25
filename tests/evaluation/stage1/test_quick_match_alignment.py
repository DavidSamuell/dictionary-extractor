"""OmniDocBench quick_match line alignment for eval-flat."""

from dictextractor.evaluation.stage1.alignment import align_lines_quick_match, align_rows
from dictextractor.evaluation.stage1.character_quality import compute_character_quality
from dictextractor.evaluation.stage1.flat_evaluator import FlatStage1Evaluator, _lines_to_rows
from dictextractor.evaluation.stage1.read_order import compute_read_order


def test_swapped_lines_penalize_read_order_not_quick_match_gcer() -> None:
    gold = _lines_to_rows(["hello", "world"])
    pred = _lines_to_rows(["world", "hello"])
    alignment = align_lines_quick_match(pred, gold)
    ro = compute_read_order(alignment)
    cq = compute_character_quality(alignment)
    assert ro.read_order_edit > 0.0
    assert cq.gcer == 0.0


def test_flat_evaluator_reports_read_order_quick_match(tmp_path) -> None:
    gold_path = tmp_path / "page_stage1_GOLD_flat.txt"
    pred_path = tmp_path / "page_stage1_flat.txt"
    gold_path.write_text("one\ntwo\n", encoding="utf-8")
    pred_path.write_text("two\none\n", encoding="utf-8")
    metrics = FlatStage1Evaluator(character_alignment="quick_match").evaluate(
        pred_path, gold_path, page_id="test/page"
    )
    assert metrics.read_order.read_order_edit > 0.0
    assert metrics.character_quality.gcer == 0.0


def test_quick_match_decouples_order_from_character_quality(tmp_path) -> None:
    gold_path = tmp_path / "page_stage1_GOLD_flat.txt"
    pred_path = tmp_path / "page_stage1_flat.txt"
    gold_path.write_text("one\ntwo\n", encoding="utf-8")
    pred_path.write_text("two\none\n", encoding="utf-8")
    quick = FlatStage1Evaluator(character_alignment="quick_match").evaluate(
        pred_path, gold_path, page_id="test/page"
    )
    collapsed = FlatStage1Evaluator(character_alignment="collapsed").evaluate(
        pred_path, gold_path, page_id="test/page"
    )
    assert quick.character_quality.gcer == 0.0
    assert collapsed.character_quality.gcer > 0.0
    assert quick.read_order.read_order_edit > 0.0


def test_quick_match_penalizes_swapped_lines_on_long_page() -> None:
    lines = [f"entry{i} gloss{i}" for i in range(20)]
    gold = _lines_to_rows(lines)
    pred = _lines_to_rows(list(reversed(lines)))
    cq = compute_character_quality(align_lines_quick_match(pred, gold))
    assert cq.gcer == 0.0


def test_case_invariant_character_quality_for_pos() -> None:
    gold = _lines_to_rows(["<b>VERB</b> Tone: H gloss"])
    pred = _lines_to_rows(["<b>verb</b> Tone: H gloss"])
    cq = compute_character_quality(align_rows(pred, gold, threshold=0.6))
    assert cq.gcer == 0.0


def test_adjacent_pred_merge_handles_split_line() -> None:
    gold = _lines_to_rows(["hello world"])
    pred = _lines_to_rows(["hello", "world"])
    cq = compute_character_quality(align_lines_quick_match(pred, gold))
    assert cq.gcer == 0.0
