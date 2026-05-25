"""Stage 2 MDF evaluation: record detection, marker F1, read order."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence

import Levenshtein

from dictextractor.evaluation.stage2.mdf_align import align_lines, align_records
from dictextractor.evaluation.stage2.mdf_marker_equiv import markers_equivalent
from dictextractor.evaluation.stage2.mdf_metrics import (
    MarkerErrorSample,
    MdfPageMetrics,
    PrfCounts,
    ReadOrderMetrics,
    RecordIndexSample,
    RecordSample,
)
from dictextractor.evaluation.stage2.mdf_parser import parse_mdf

logger = logging.getLogger(__name__)

DEFAULT_RECORD_THRESHOLD = 0.6
DEFAULT_LINE_THRESHOLD = 0.7


@dataclass(frozen=True)
class MdfEvalTask:
    """One Stage 2 MDF evaluation unit."""

    experiment: str
    pred_path: Path
    gold_path: Path
    page_id: str


def compute_read_order_metrics(
    matched_gold_indices: Sequence[int],
    matched_pred_indices: Sequence[int],
    n_gold: int,
) -> ReadOrderMetrics:
    """Compute ReadOrderEdit from matched record index pairs."""
    pairs = sorted(zip(matched_gold_indices, matched_pred_indices), key=lambda x: x[1])
    gt = list(range(n_gold))
    pred_order = [gold_idx for gold_idx, _ in pairs]
    dist = Levenshtein.distance(gt, pred_order)
    maxlen = max(len(gt), len(pred_order), 1)
    return ReadOrderMetrics(
        read_order_edit=dist / maxlen,
        edit_distance=dist,
        max_length=maxlen,
    )


class MdfEvaluator:
    """Evaluate predicted MDF against gold using fuzzy record and line alignment."""

    def __init__(
        self,
        *,
        record_threshold: float = DEFAULT_RECORD_THRESHOLD,
        line_threshold: float = DEFAULT_LINE_THRESHOLD,
        marker_sub_list_path: str | Path | None = None,
    ) -> None:
        self.record_threshold = record_threshold
        self.line_threshold = line_threshold
        self.marker_sub_list_path = str(marker_sub_list_path) if marker_sub_list_path else None

    def evaluate(
        self,
        pred_path: str | Path,
        gold_path: str | Path,
        page_id: str = "",
    ) -> MdfPageMetrics:
        """Evaluate one predicted MDF page against gold."""
        pred_path = Path(pred_path)
        gold_path = Path(gold_path)
        gold_text = gold_path.read_text(encoding="utf-8")
        pred_text = pred_path.read_text(encoding="utf-8")

        gold_records = parse_mdf(gold_text)
        pred_records = parse_mdf(pred_text)
        record_alignment = align_records(
            gold_records,
            pred_records,
            threshold=self.record_threshold,
        )

        marker_counts = PrfCounts()
        confusion: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        marker_errors: List[MarkerErrorSample] = []
        record_samples: List[RecordSample] = []

        for match in record_alignment.matched:
            gold_record = gold_records[match.gold_index]
            pred_record = pred_records[match.pred_index]
            line_alignment = align_lines(
                gold_record.lines,
                pred_record.lines,
                threshold=self.line_threshold,
            )

            for line_match in line_alignment.matched:
                gold_line = gold_record.lines[line_match.gold_index]
                pred_line = pred_record.lines[line_match.pred_index]
                if markers_equivalent(
                    gold_line.marker,
                    pred_line.marker,
                    sub_list_path=self.marker_sub_list_path,
                ):
                    marker_counts.tp += 1
                else:
                    marker_counts.fp += 1
                    marker_counts.fn += 1
                    confusion[gold_line.marker][pred_line.marker] += 1
                    if len(marker_errors) < 15:
                        marker_errors.append(
                            MarkerErrorSample(
                                headword=gold_record.headword,
                                gold=f"\\{gold_line.marker} {gold_line.value}",
                                pred=f"\\{pred_line.marker} {pred_line.value}",
                                value_sim=round(line_match.similarity, 3),
                            )
                        )

            marker_counts.fn += len(line_alignment.missing_gold)
            marker_counts.fp += len(line_alignment.extra_pred)

            if match.similarity < 0.95:
                record_samples.append(
                    RecordSample(
                        gold_index=match.gold_index,
                        pred_index=match.pred_index,
                        similarity=round(match.similarity, 3),
                        headword_gold=gold_record.headword,
                        headword_pred=pred_record.headword,
                        gold_lines=len(gold_record.lines),
                        pred_lines=len(pred_record.lines),
                        line_pairs=len(line_alignment.matched),
                        missing_lines=len(line_alignment.missing_gold),
                        extra_lines=len(line_alignment.extra_pred),
                    )
                )

        read_order = compute_read_order_metrics(
            [m.gold_index for m in record_alignment.matched],
            [m.pred_index for m in record_alignment.matched],
            len(gold_records),
        )

        return MdfPageMetrics(
            page_id=page_id or gold_path.parent.name,
            gold_path=str(gold_path),
            pred_path=str(pred_path),
            n_pred_records=len(pred_records),
            record=PrfCounts(
                tp=len(record_alignment.matched),
                fp=len(record_alignment.extra_pred),
                fn=len(record_alignment.missing_gold),
            ),
            marker=marker_counts,
            read_order=read_order,
            marker_confusion={g: dict(c) for g, c in confusion.items()},
            record_samples=record_samples[:12],
            marker_error_samples=marker_errors,
            missing_record_samples=[
                RecordIndexSample(
                    index=idx,
                    headword=gold_records[idx].headword,
                    fingerprint_preview=gold_records[idx].fingerprint()[:120],
                )
                for idx in record_alignment.missing_gold[:8]
            ],
            extra_record_samples=[
                RecordIndexSample(
                    index=idx,
                    headword=pred_records[idx].headword,
                    fingerprint_preview=pred_records[idx].fingerprint()[:120],
                )
                for idx in record_alignment.extra_pred[:8]
            ],
        )

    @staticmethod
    def discover_tasks(
        samples_dir: str | Path,
        experiments: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
    ) -> List[MdfEvalTask]:
        """Discover gold/pred MDF pairs under a samples tree."""
        samples_dir = Path(samples_dir)
        tasks: List[MdfEvalTask] = []
        selected_languages = set(languages) if languages else None

        golds_by_lang: Dict[Path, List[Path]] = {}
        for gold_path in sorted(samples_dir.glob("*/outputs/stage-2-gold/*/*.mdf.txt")):
            lang_dir = gold_path.parents[3]
            if selected_languages and lang_dir.name not in selected_languages:
                continue
            golds_by_lang.setdefault(lang_dir, []).append(gold_path)

        for lang_dir, gold_paths in sorted(golds_by_lang.items()):
            stage2_root = lang_dir / "outputs" / "stage-2"
            if not stage2_root.is_dir():
                continue
            available = sorted(
                p.name for p in stage2_root.iterdir() if p.is_dir() and not p.name.startswith(".")
            )
            exp_names = available if experiments is None else [e for e in experiments if e in available]
            for exp in exp_names:
                for gold_path in gold_paths:
                    stem = gold_path.parent.name
                    pred_path = stage2_root / exp / stem / f"{stem}.mdf.txt"
                    if not pred_path.is_file():
                        logger.debug("skip missing pred: %s", pred_path)
                        continue
                    tasks.append(
                        MdfEvalTask(
                            experiment=exp,
                            pred_path=pred_path,
                            gold_path=gold_path,
                            page_id=f"{lang_dir.name}/{stem}",
                        )
                    )
        return tasks

    @staticmethod
    def metrics_to_dict(metrics: MdfPageMetrics) -> dict:
        """Serialize page metrics for JSON reports."""
        return {
            "page_id": metrics.page_id,
            "gold_path": metrics.gold_path,
            "pred_path": metrics.pred_path,
            "n_pred_records": metrics.n_pred_records,
            "Record_Accuracy": round(metrics.record_accuracy, 6),
            "MDF_Fields_F1": round(metrics.mdf_fields_f1, 6),
            "record_counts": {
                "tp": metrics.record.tp,
                "fp": metrics.record.fp,
                "fn": metrics.record.fn,
            },
            "marker_counts": {
                "tp": metrics.marker.tp,
                "fp": metrics.marker.fp,
                "fn": metrics.marker.fn,
            },
            "read_order": {
                "ReadOrderEdit": round(metrics.read_order.read_order_edit, 6),
                "edit_distance": metrics.read_order.edit_distance,
                "max_length": metrics.read_order.max_length,
            },
            "marker_confusion": metrics.marker_confusion,
            "record_samples": [asdict(s) for s in metrics.record_samples],
            "marker_error_samples": [asdict(s) for s in metrics.marker_error_samples],
            "missing_record_samples": [asdict(s) for s in metrics.missing_record_samples],
            "extra_record_samples": [asdict(s) for s in metrics.extra_record_samples],
        }

    def generate_json_report(self, results: List[MdfPageMetrics], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "settings": {
                "record_threshold": self.record_threshold,
                "line_threshold": self.line_threshold,
                "marker_sub_list_path": self.marker_sub_list_path,
            },
            "pages": [self.metrics_to_dict(m) for m in results],
        }
        if len(results) > 1:
            payload["aggregate"] = self._aggregate_dict(results)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def generate_text_report(self, results: List[MdfPageMetrics], output_path: Path) -> str:
        lines = [
            "=" * 72,
            "STAGE 2 MDF EVALUATION REPORT",
            "=" * 72,
            f"record_threshold={self.record_threshold}  line_threshold={self.line_threshold}",
            "",
        ]
        for metrics in results:
            lines.extend(self._format_page(metrics))
            lines.append("")
        if len(results) > 1:
            agg = self._aggregate_dict(results)
            lines.extend(
                [
                    "=" * 72,
                    f"AGGREGATE ({len(results)} pages)",
                    "=" * 72,
                    f"  Record Accuracy: {agg['Record_Accuracy']:.4f}",
                    f"  MDF Fields F1:   {agg['MDF_Fields_F1']:.4f}",
                    f"  ReadOrderEdit:   {agg['read_order']['ReadOrderEdit']:.4f}",
                    "",
                ]
            )
        report = "\n".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        return report

    def generate_summary_csv(self, results_by_exp: Dict[str, List[MdfPageMetrics]], output_path: Path) -> None:
        rows: List[dict] = []
        for exp, pages in results_by_exp.items():
            for page in pages:
                rows.append(
                    {
                        "experiment": exp,
                        "page_id": page.page_id,
                        "Record_Accuracy": round(page.record_accuracy, 6),
                        "MDF_Fields_F1": round(page.mdf_fields_f1, 6),
                        "ReadOrderEdit": round(page.read_order.read_order_edit, 6),
                    }
                )
            if len(pages) > 1:
                agg = self._aggregate_dict(pages)
                rows.append(
                    {
                        "experiment": exp,
                        "page_id": "__aggregate__",
                        "Record_Accuracy": agg["Record_Accuracy"],
                        "MDF_Fields_F1": agg["MDF_Fields_F1"],
                        "ReadOrderEdit": agg["read_order"]["ReadOrderEdit"],
                    }
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "experiment",
                    "page_id",
                    "Record_Accuracy",
                    "MDF_Fields_F1",
                    "ReadOrderEdit",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _format_page(metrics: MdfPageMetrics) -> List[str]:
        return [
            f"--- {metrics.page_id} ---",
            f"  Records: matched={metrics.record.tp} missing={metrics.record.fn} "
            f"extra={metrics.record.fp} (pred={metrics.n_pred_records})",
            f"  Record Accuracy: {metrics.record_accuracy:.4f}",
            f"  MDF Fields F1:   {metrics.mdf_fields_f1:.4f}",
            f"  ReadOrderEdit:   {metrics.read_order.read_order_edit:.4f}",
        ]

    @staticmethod
    def _aggregate_dict(results: List[MdfPageMetrics]) -> dict:
        record_tp = sum(m.record.tp for m in results)
        record_fp = sum(m.record.fp for m in results)
        record_fn = sum(m.record.fn for m in results)
        marker_tp = sum(m.marker.tp for m in results)
        marker_fp = sum(m.marker.fp for m in results)
        marker_fn = sum(m.marker.fn for m in results)
        ro_edit = sum(m.read_order.read_order_edit for m in results) / len(results)

        record_accuracy = (
            record_tp / (record_tp + record_fn) if (record_tp + record_fn) else 0.0
        )
        marker_p = marker_tp / (marker_tp + marker_fp) if (marker_tp + marker_fp) else 0.0
        marker_r = marker_tp / (marker_tp + marker_fn) if (marker_tp + marker_fn) else 0.0
        mdf_fields_f1 = (
            2 * marker_p * marker_r / (marker_p + marker_r) if (marker_p + marker_r) else 0.0
        )

        return {
            "Record_Accuracy": round(record_accuracy, 6),
            "MDF_Fields_F1": round(mdf_fields_f1, 6),
            "record_counts": {"tp": record_tp, "fp": record_fp, "fn": record_fn},
            "marker_counts": {"tp": marker_tp, "fp": marker_fp, "fn": marker_fn},
            "read_order": {"ReadOrderEdit": round(ro_edit, 6)},
        }
