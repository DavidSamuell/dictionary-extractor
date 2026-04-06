"""
Read-order (structure preservation) metric for Stage 1 evaluation.

Concatenates all tag-stripped text in canonical reading order into one
continuous string, then computes NED between predicted and gold.
"""

from typing import Dict, List, Tuple

import Levenshtein

from dictextractor.evaluation.stage1.stage1_metrics import ReadOrderMetrics
from dictextractor.evaluation.stage1.tag_parser import (
    normalize_unicode,
    normalize_whitespace,
    strip_tags,
)

Row = Dict[str, str]

# Canonical column ordering for concatenation.
_COLUMN_ORDER = {"single": 0, "left": 1, "center": 2, "right": 3}


def _sort_key(row: Row) -> Tuple[int, int]:
    col_rank = _COLUMN_ORDER.get(row["column_id"], 99)
    return (col_rank, int(row["line_number"]))


def _concat(rows: List[Row]) -> str:
    """Concatenate tag-stripped lines in reading order into one string."""
    sorted_rows = sorted(rows, key=_sort_key)
    parts = [
        normalize_whitespace(normalize_unicode(strip_tags(r["text"])))
        for r in sorted_rows
    ]
    return " ".join(parts)


def compute_read_order(
    pred_rows: List[Row],
    gold_rows: List[Row],
    character_ned: float = 0.0,
) -> ReadOrderMetrics:
    """Compute read-order NED and isolated order error.

    Parameters
    ----------
    pred_rows, gold_rows:
        TSV rows with column_id / line_number / text.
    character_ned:
        The per-line NED from the character-quality metric (used to
        factor out character-level errors from the order signal).
    """
    pred_str = _concat(pred_rows)
    gold_str = _concat(gold_rows)

    dist = Levenshtein.distance(pred_str, gold_str)
    maxlen = max(len(pred_str), len(gold_str), 1)
    ned = dist / maxlen

    # Isolated order error: read-order NED minus the baseline character NED.
    # Clamped to zero since character errors can sometimes inflate the per-line
    # metric slightly more than the concatenated one.
    isolated = max(ned - character_ned, 0.0)

    return ReadOrderMetrics(
        ned=ned,
        edit_distance=dist,
        max_length=maxlen,
        character_ned=character_ned,
        isolated_order_error=isolated,
    )
