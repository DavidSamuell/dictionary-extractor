"""Greedy alignment of MDF records and field lines by normalized value."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from dictextractor.evaluation.stage2.mdf_parser import FieldLine, MdfRecord, normalize_field_value
from dictextractor.evaluation.stage2.mdf_similarity import value_similarity


@dataclass(frozen=True)
class RecordMatch:
    """One aligned gold/pred record pair."""

    gold_index: int
    pred_index: int
    similarity: float


@dataclass(frozen=True)
class LineMatch:
    """One aligned gold/pred field line inside a record pair."""

    gold_index: int
    pred_index: int
    similarity: float


@dataclass(frozen=True)
class RecordAlignmentResult:
    """Record-level greedy alignment output."""

    matched: List[RecordMatch]
    missing_gold: List[int]
    extra_pred: List[int]


@dataclass(frozen=True)
class LineAlignmentResult:
    """Line-level greedy alignment output within one record."""

    matched: List[LineMatch]
    missing_gold: List[int]
    extra_pred: List[int]


def _greedy_match(
    candidates: List[Tuple[float, int, int]],
    n_gold: int,
    n_pred: int,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matched: List[Tuple[int, int, float]] = []

    for sim, gold_idx, pred_idx in candidates:
        if gold_idx in used_gold or pred_idx in used_pred:
            continue
        used_gold.add(gold_idx)
        used_pred.add(pred_idx)
        matched.append((gold_idx, pred_idx, sim))

    missing_gold = [idx for idx in range(n_gold) if idx not in used_gold]
    extra_pred = [idx for idx in range(n_pred) if idx not in used_pred]
    return matched, missing_gold, extra_pred


def align_records(
    gold: Sequence[MdfRecord],
    pred: Sequence[MdfRecord],
    *,
    threshold: float = 0.6,
) -> RecordAlignmentResult:
    """Greedy one-to-one record matching by value fingerprint."""
    candidates: List[Tuple[float, int, int]] = []
    for gold_idx, gold_record in enumerate(gold):
        gold_fp = gold_record.fingerprint()
        for pred_idx, pred_record in enumerate(pred):
            sim = value_similarity(gold_fp, pred_record.fingerprint())
            if sim >= threshold:
                candidates.append((sim, gold_idx, pred_idx))

    matched_raw, missing_gold, extra_pred = _greedy_match(candidates, len(gold), len(pred))
    matched = [
        RecordMatch(gold_index=gi, pred_index=pi, similarity=sim)
        for gi, pi, sim in matched_raw
    ]
    return RecordAlignmentResult(
        matched=matched,
        missing_gold=missing_gold,
        extra_pred=extra_pred,
    )


def align_lines(
    gold_lines: Sequence[FieldLine],
    pred_lines: Sequence[FieldLine],
    *,
    threshold: float = 0.7,
) -> LineAlignmentResult:
    """Greedy line alignment by normalized value within a record pair."""
    candidates: List[Tuple[float, int, int]] = []
    for gold_idx, gold_line in enumerate(gold_lines):
        gold_value = normalize_field_value(gold_line.value)
        for pred_idx, pred_line in enumerate(pred_lines):
            sim = value_similarity(gold_value, normalize_field_value(pred_line.value))
            if sim >= threshold:
                candidates.append((sim, gold_idx, pred_idx))

    matched_raw, missing_gold, extra_pred = _greedy_match(
        candidates, len(gold_lines), len(pred_lines)
    )
    matched = [
        LineMatch(gold_index=gi, pred_index=pi, similarity=sim)
        for gi, pi, sim in matched_raw
    ]
    return LineAlignmentResult(
        matched=matched,
        missing_gold=missing_gold,
        extra_pred=extra_pred,
    )
