"""Read-order metrics for eval-flat."""

from __future__ import annotations

import Levenshtein

from dictextractor.evaluation.stage1.alignment import AlignmentResult, clean_text
from dictextractor.evaluation.stage1.stage1_metrics import ReadOrderMetrics


def _collapse_lines(lines: list[str]) -> str:
    """Join flat lines the same way as page-collapse character alignment."""
    parts = [clean_text(line) for line in lines if clean_text(line)]
    return " ".join(parts)


def _find_anchor(haystack: str, needle: str, start: int) -> int:
    """Find ``needle`` in ``haystack`` at/after ``start`` (exact, then casefold)."""
    if not needle:
        return start
    pos = haystack.find(needle, start)
    if pos >= 0:
        return pos
    from dictextractor.evaluation.stage1.tag_parser import casefold_letters_for_eval

    folded_hay = casefold_letters_for_eval(haystack)
    folded_needle = casefold_letters_for_eval(needle)
    if not folded_needle:
        return start
    return folded_hay.find(folded_needle, start)


def compute_read_order_collapsed(
    gold_lines: list[str],
    pred_lines: list[str],
) -> ReadOrderMetrics:
    """ReadOrderEdit via sequential anchors in the collapsed pred string.

    For each gold line (in order), locate its cleaned text left-to-right in the
    collapsed prediction. Matched gold indices in discovery order are compared
    to ``[0, 1, …, n-1]`` with normalized Levenshtein distance.

    Uses the same ``clean_text`` normalisation as page-collapse alignment.
    Unmatched gold lines are omitted from the pred sequence (counted as deletions).
    """
    n_gold = len(gold_lines)
    if n_gold == 0:
        return ReadOrderMetrics(read_order_edit=0.0, edit_distance=0, max_length=1)

    pred_collapsed = _collapse_lines(pred_lines)
    gt = list(range(n_gold))

    pred_order: list[int] = []
    search_start = 0
    for idx, line in enumerate(gold_lines):
        needle = clean_text(line)
        if not needle:
            continue
        pos = _find_anchor(pred_collapsed, needle, search_start)
        if pos < 0:
            continue
        pred_order.append(idx)
        search_start = pos + len(needle)

    dist = Levenshtein.distance(gt, pred_order)
    maxlen = max(len(gt), len(pred_order), 1)
    return ReadOrderMetrics(
        read_order_edit=dist / maxlen,
        edit_distance=dist,
        max_length=maxlen,
    )


def compute_read_order(alignment: AlignmentResult) -> ReadOrderMetrics:
    """OmniDocBench-style ReadOrderEdit over gold line indices.

    After quick_match alignment:

    1. ``gt = [0, 1, …, n−1]`` for all gold lines.
    2. Sort matched (gold, pred) pairs by predicted line index (``pred.start_index``).
    3. ``pred`` = gold indices in predicted reading order (flattened when a span
       covers multiple lines, as when several gold lines map to one merged pred).
    4. Unmatched gold lines stay in ``gt`` only; unmatched pred lines are omitted.

    Same formula as OmniDocBench ``get_order_paired`` and Stage 2 MDF read order,
    with flat lines in place of annotated layout units.
    """
    n_gold = len(alignment.gold_rows)
    if n_gold == 0:
        return ReadOrderMetrics(read_order_edit=0.0, edit_distance=0, max_length=1)

    gt = list(range(n_gold))

    matched_pairs = [
        pair for pair in alignment.pairs if pair.pred is not None and pair.gold is not None
    ]
    matched_pairs.sort(key=lambda pair: pair.pred.start_index)  # type: ignore[union-attr]

    pred: list[int] = []
    for pair in matched_pairs:
        gold = pair.gold
        assert gold is not None
        pred.extend(range(gold.start_index, gold.end_index + 1))

    dist = Levenshtein.distance(gt, pred)
    maxlen = max(len(pred), len(gt), 1)

    return ReadOrderMetrics(
        read_order_edit=dist / maxlen,
        edit_distance=dist,
        max_length=maxlen,
    )
