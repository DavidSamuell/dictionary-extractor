"""Metric dataclasses for Stage 2 MDF evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PrfCounts:
    """Micro precision/recall/F1 counts."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class ReadOrderMetrics:
    """OmniDocBench-style read order over matched record indices."""

    read_order_edit: float = 0.0
    edit_distance: int = 0
    max_length: int = 0


@dataclass
class MarkerErrorSample:
    """One wrong-marker example for reports."""

    headword: str
    gold: str
    pred: str
    value_sim: float


@dataclass
class RecordSample:
    """Diagnostic summary for one matched record pair."""

    gold_index: int
    pred_index: int
    similarity: float
    headword_gold: str
    headword_pred: str
    gold_lines: int
    pred_lines: int
    line_pairs: int
    missing_lines: int
    extra_lines: int


@dataclass
class RecordIndexSample:
    """One unmatched record for reports."""

    index: int
    headword: str
    fingerprint_preview: str


@dataclass
class MdfPageMetrics:
    """Evaluation output for one gold/pred MDF page pair."""

    page_id: str
    gold_path: str
    pred_path: str
    n_pred_records: int = 0
    record: PrfCounts = field(default_factory=PrfCounts)
    marker: PrfCounts = field(default_factory=PrfCounts)
    read_order: ReadOrderMetrics = field(default_factory=ReadOrderMetrics)
    marker_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    record_samples: List[RecordSample] = field(default_factory=list)
    marker_error_samples: List[MarkerErrorSample] = field(default_factory=list)
    missing_record_samples: List[RecordIndexSample] = field(default_factory=list)
    extra_record_samples: List[RecordIndexSample] = field(default_factory=list)

    @property
    def record_accuracy(self) -> float:
        """Fraction of gold records correctly matched (``TP / (TP + FN)``)."""
        return self.record.recall

    @property
    def mdf_fields_f1(self) -> float:
        """F1 over MDF field-line marker assignment within matched records."""
        return self.marker.f1
