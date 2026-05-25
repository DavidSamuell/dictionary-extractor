"""Page-collapse alignment for eval-flat character/typography metrics."""

from dictextractor.evaluation.stage1.alignment import (
    align_page_collapsed,
    collapse_rows_to_page,
)
from dictextractor.evaluation.stage1.character_quality import compute_character_quality
from dictextractor.evaluation.stage1.flat_evaluator import _lines_to_rows


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
