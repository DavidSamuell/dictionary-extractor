"""
OmniDocBench-style semantic alignment for Stage 1 TSV rows.

The matcher works on adjacent row spans rather than strict
``(column_id, line_number)`` keys, so harmless line splits/merges do not
dominate text-recognition metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import grapheme
import Levenshtein

from dictextractor.evaluation.stage1.tag_parser import (
    normalize_line_text,
    strip_tags,
)

Row = Dict[str, str]


@dataclass(frozen=True)
class RowSpan:
    """Adjacent TSV rows treated as one semantic alignment unit."""

    span_id: str
    rows: List[Row]
    start_index: int
    end_index: int
    text: str
    tagged_text: str

    @property
    def row_count(self) -> int:
        return self.end_index - self.start_index + 1


@dataclass(frozen=True)
class AlignedSpanPair:
    """One matched or unmatched alignment unit."""

    pred: Optional[RowSpan]
    gold: Optional[RowSpan]
    similarity: float
    text_edit: float


@dataclass(frozen=True)
class AlignmentResult:
    """Semantic alignment output shared by all Stage 1 metrics."""

    pairs: List[AlignedSpanPair]
    pred_rows: List[Row]
    gold_rows: List[Row]

    @property
    def matched_count(self) -> int:
        return sum(1 for pair in self.pairs if pair.pred and pair.gold)

    @property
    def missing_count(self) -> int:
        return sum(1 for pair in self.pairs if pair.gold and not pair.pred)

    @property
    def extra_count(self) -> int:
        return sum(1 for pair in self.pairs if pair.pred and not pair.gold)


@dataclass(frozen=True)
class _Candidate:
    pred: RowSpan
    gold: RowSpan
    similarity: float
    text_edit: float


def clean_text(text: str) -> str:
    """Tag-strip and normalize text for semantic matching and text metrics."""
    return normalize_line_text(strip_tags(text))


def _span_text(rows: List[Row], *, tagged: bool) -> str:
    parts = [r.get("text", "") if tagged else clean_text(r.get("text", "")) for r in rows]
    joined = " ".join(p for p in parts if p)
    return normalize_line_text(joined) if not tagged else joined


def _make_spans(rows: List[Row], prefix: str, max_span_rows: int) -> List[RowSpan]:
    spans: List[RowSpan] = []
    for start in range(len(rows)):
        max_end = min(len(rows), start + max_span_rows)
        for end_exclusive in range(start + 1, max_end + 1):
            span_rows = rows[start:end_exclusive]
            end = end_exclusive - 1
            spans.append(
                RowSpan(
                    span_id=f"{prefix}{start}" if start == end else f"{prefix}{start}-{end}",
                    rows=span_rows,
                    start_index=start,
                    end_index=end,
                    text=_span_text(span_rows, tagged=False),
                    tagged_text=_span_text(span_rows, tagged=True),
                )
            )
    return spans


def _ned(pred: str, gold: str) -> float:
    pred_graphemes = list(grapheme.graphemes(pred))
    gold_graphemes = list(grapheme.graphemes(gold))
    max_len = max(len(pred_graphemes), len(gold_graphemes), 1)
    return Levenshtein.distance(pred_graphemes, gold_graphemes) / max_len


def _row_indexes(span: RowSpan) -> set[int]:
    return set(range(span.start_index, span.end_index + 1))


def collapse_rows_to_page(rows: List[Row]) -> List[Row]:
    """Collapse all rows into one synthetic page row (tagged text in ``text``)."""
    if not rows:
        return []
    tagged = _span_text(rows, tagged=True)
    return [{"column_id": "page", "line_number": "1", "text": tagged}]


def align_page_collapsed(
    pred_rows: List[Row],
    gold_rows: List[Row],
) -> AlignmentResult:
    """Align pred/gold as single collapsed page spans (no multi-line fuzzy search).

    Used for character and typography metrics where line boundaries should not
    affect scoring. Joins rows with spaces and applies the same ``clean_text``
    normalisation as multi-row spans.
    """
    return align_rows(
        collapse_rows_to_page(pred_rows),
        collapse_rows_to_page(gold_rows),
        threshold=0.0,
        max_span_rows=1,
    )


def align_rows(
    pred_rows: List[Row],
    gold_rows: List[Row],
    *,
    threshold: float = 0.5,
    max_span_rows: int = 3,
) -> AlignmentResult:
    """Align predicted/gold TSV rows using adjacent span fuzzy matching.

    Candidate pred/gold spans are scored by grapheme normalized edit distance.
    Non-overlapping candidates are then selected greedily from highest
    similarity to lowest, which approximates OmniDocBench's adjacency search
    while keeping dictionary-page evaluation simple and deterministic.
    """
    if max_span_rows < 1:
        raise ValueError("max_span_rows must be >= 1")

    pred_spans = _make_spans(pred_rows, "p", max_span_rows)
    gold_spans = _make_spans(gold_rows, "g", max_span_rows)

    candidates: List[_Candidate] = []
    for pred in pred_spans:
        for gold in gold_spans:
            text_edit = _ned(pred.text, gold.text)
            similarity = 1.0 - text_edit
            if similarity >= threshold:
                candidates.append(_Candidate(pred, gold, similarity, text_edit))

    candidates.sort(
        key=lambda c: (
            -c.similarity,
            c.pred.row_count + c.gold.row_count,
            c.pred.start_index,
            c.gold.start_index,
        )
    )

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    pairs: List[AlignedSpanPair] = []
    for candidate in candidates:
        pred_indexes = _row_indexes(candidate.pred)
        gold_indexes = _row_indexes(candidate.gold)
        if pred_indexes & used_pred or gold_indexes & used_gold:
            continue
        used_pred.update(pred_indexes)
        used_gold.update(gold_indexes)
        pairs.append(
            AlignedSpanPair(
                pred=candidate.pred,
                gold=candidate.gold,
                similarity=candidate.similarity,
                text_edit=candidate.text_edit,
            )
        )

    for idx, row in enumerate(pred_rows):
        if idx not in used_pred:
            span = _make_spans([row], f"p{idx}_", 1)[0]
            pairs.append(AlignedSpanPair(pred=span, gold=None, similarity=0.0, text_edit=1.0))
    for idx, row in enumerate(gold_rows):
        if idx not in used_gold:
            span = _make_spans([row], f"g{idx}_", 1)[0]
            pairs.append(AlignedSpanPair(pred=None, gold=span, similarity=0.0, text_edit=1.0))

    pairs.sort(
        key=lambda pair: (
            pair.pred.start_index if pair.pred else float("inf"),
            pair.gold.start_index if pair.gold else float("inf"),
        )
    )
    return AlignmentResult(pairs=pairs, pred_rows=pred_rows, gold_rows=gold_rows)
