"""
Stage1Evaluator: orchestrates all Stage 1 OCR evaluation metrics.

Loads predicted and gold TSV files, runs character quality, markup quality,
and read-order evaluations, then generates JSON + human-readable reports.
"""

import csv
import io
import json
from pathlib import Path
from typing import Dict, List

from dictextractor.evaluation.stage1.character_quality import compute_character_quality
from dictextractor.evaluation.stage1.markup_quality import compute_markup_quality
from dictextractor.evaluation.stage1.read_order import compute_read_order
from dictextractor.evaluation.stage1.stage1_metrics import Stage1Metrics

Row = Dict[str, str]


class Stage1Evaluator:
    """Evaluate a single predicted Stage 1 TSV against a gold TSV."""

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def load_tsv(filepath: str | Path) -> List[Row]:
        rows: List[Row] = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(dict(row))
        return rows

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pred_path: str | Path,
        gold_path: str | Path,
        page_id: str = "",
    ) -> Stage1Metrics:
        pred_rows = self.load_tsv(pred_path)
        gold_rows = self.load_tsv(gold_path)

        char_q = compute_character_quality(pred_rows, gold_rows)
        markup_q = compute_markup_quality(pred_rows, gold_rows)
        read_o = compute_read_order(pred_rows, gold_rows, character_ned=char_q.ned)

        return Stage1Metrics(
            page_id=page_id,
            character_quality=char_q,
            markup_quality=markup_q,
            read_order=read_o,
        )

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(self, samples_dir: str | Path) -> List[Stage1Metrics]:
        """Find all *_stage1.tsv / *_stage1_GOLD.tsv pairs under *samples_dir*."""
        samples_dir = Path(samples_dir)
        results: List[Stage1Metrics] = []

        for gold_path in sorted(samples_dir.rglob("*_stage1_GOLD.tsv")):
            pred_path = gold_path.parent / gold_path.name.replace(
                "_stage1_GOLD.tsv", "_stage1.tsv"
            )
            if not pred_path.exists():
                print(f"  [skip] No predicted file for {gold_path}")
                continue

            # Derive a human-readable page id from the path
            # e.g. "Evenki-Russian/page_1"
            try:
                rel = gold_path.relative_to(samples_dir)
                # pattern: <Language>/outputs/2-stage/<page_dir>/...
                parts = rel.parts
                lang = parts[0]
                page_dir = parts[3] if len(parts) > 3 else parts[-2]
                page_id = f"{lang}/{page_dir}"
            except (IndexError, ValueError):
                page_id = gold_path.stem

            metrics = self.evaluate(pred_path, gold_path, page_id=page_id)
            results.append(metrics)

        return results

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    @staticmethod
    def metrics_to_dict(m: Stage1Metrics) -> dict:
        cq = m.character_quality
        mq = m.markup_quality
        ro = m.read_order
        return {
            "page_id": m.page_id,
            "character_quality": {
                "GCER": round(cq.gcer, 6),
                "WER": round(cq.wer, 6),
                "NED": round(cq.ned, 6),
                "total_chars_gold": cq.total_chars_gold,
                "total_chars_pred": cq.total_chars_pred,
                "total_char_edits": cq.total_char_edits,
                "total_words_gold": cq.total_words_gold,
                "total_word_edits": cq.total_word_edits,
                "matched_lines": cq.matched_lines,
                "missing_lines": cq.missing_lines,
                "extra_lines": cq.extra_lines,
            },
            "markup_quality": {
                "bold": {
                    "precision": round(mq.bold.precision, 4),
                    "recall": round(mq.bold.recall, 4),
                    "f1": round(mq.bold.f1, 4),
                    "tp": mq.bold.true_positives,
                    "fp": mq.bold.false_positives,
                    "fn": mq.bold.false_negatives,
                },
                "italic": {
                    "precision": round(mq.italic.precision, 4),
                    "recall": round(mq.italic.recall, 4),
                    "f1": round(mq.italic.f1, 4),
                    "tp": mq.italic.true_positives,
                    "fp": mq.italic.false_positives,
                    "fn": mq.italic.false_negatives,
                },
            },
            "read_order": {
                "NED": round(ro.ned, 6),
                "edit_distance": ro.edit_distance,
                "max_length": ro.max_length,
                "character_NED_baseline": round(ro.character_ned, 6),
                "isolated_order_error": round(ro.isolated_order_error, 6),
            },
        }

    def generate_json_report(
        self,
        results: list[Stage1Metrics],
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [self.metrics_to_dict(m) for m in results]

        # Compute aggregate across all pages
        if len(results) > 1:
            agg = self._aggregate(results)
            data.append({"page_id": "__aggregate__", **agg})

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # -- Per-component CSV column definitions --

    _CHAR_CSV_COLS = ["page_id", "GCER", "WER", "NED"]

    _MARKUP_CSV_COLS = [
        "page_id",
        "bold_precision", "bold_recall", "bold_f1",
        "italic_precision", "italic_recall", "italic_f1",
    ]

    _ORDER_CSV_COLS = [
        "page_id", "read_order_NED", "character_NED_baseline",
        "isolated_order_error",
    ]

    @staticmethod
    def _char_csv_row(m: Stage1Metrics) -> dict:
        cq = m.character_quality
        return {
            "page_id": m.page_id,
            "GCER": round(cq.gcer, 6),
            "WER": round(cq.wer, 6),
            "NED": round(cq.ned, 6),
        }

    @staticmethod
    def _markup_csv_row(m: Stage1Metrics) -> dict:
        mq = m.markup_quality
        return {
            "page_id": m.page_id,
            "bold_precision": round(mq.bold.precision, 4),
            "bold_recall": round(mq.bold.recall, 4),
            "bold_f1": round(mq.bold.f1, 4),
            "italic_precision": round(mq.italic.precision, 4),
            "italic_recall": round(mq.italic.recall, 4),
            "italic_f1": round(mq.italic.f1, 4),
        }

    @staticmethod
    def _order_csv_row(m: Stage1Metrics) -> dict:
        ro = m.read_order
        return {
            "page_id": m.page_id,
            "read_order_NED": round(ro.ned, 6),
            "character_NED_baseline": round(ro.character_ned, 6),
            "isolated_order_error": round(ro.isolated_order_error, 6),
        }

    def _write_csv(
        self, rows: list[dict], columns: list[str], path: Path
    ) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def generate_csv_reports(
        self,
        results: list[Stage1Metrics],
        output_dir: str | Path,
    ) -> None:
        """Write three CSV files, one per evaluation component."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        char_rows = [self._char_csv_row(m) for m in results]
        markup_rows = [self._markup_csv_row(m) for m in results]
        order_rows = [self._order_csv_row(m) for m in results]

        if len(results) > 1:
            agg = self._aggregate(results)
            char_rows.append({
                "page_id": "__aggregate__",
                "GCER": agg["character_quality"]["GCER"],
                "WER": agg["character_quality"]["WER"],
                "NED": agg["character_quality"]["NED"],
            })
            markup_rows.append({
                "page_id": "__aggregate__",
                "bold_precision": agg["markup_quality"]["bold"]["precision"],
                "bold_recall": agg["markup_quality"]["bold"]["recall"],
                "bold_f1": agg["markup_quality"]["bold"]["f1"],
                "italic_precision": agg["markup_quality"]["italic"]["precision"],
                "italic_recall": agg["markup_quality"]["italic"]["recall"],
                "italic_f1": agg["markup_quality"]["italic"]["f1"],
            })
            order_rows.append({
                "page_id": "__aggregate__",
                "read_order_NED": agg["read_order"]["NED"],
                "character_NED_baseline": agg["read_order"]["character_NED_baseline"],
                "isolated_order_error": agg["read_order"]["isolated_order_error"],
            })

        self._write_csv(char_rows, self._CHAR_CSV_COLS,
                         output_dir / "character_recognition.csv")
        self._write_csv(markup_rows, self._MARKUP_CSV_COLS,
                         output_dir / "markup_preservation.csv")
        self._write_csv(order_rows, self._ORDER_CSV_COLS,
                         output_dir / "structure_preservation.csv")

    def generate_text_report(
        self,
        results: list[Stage1Metrics],
        output_path: str | Path,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "=" * 72,
            "STAGE 1 OCR EVALUATION REPORT",
            "=" * 72,
            "",
        ]

        for m in results:
            lines.extend(self._format_page(m))
            lines.append("")

        if len(results) > 1:
            lines.extend(self._format_aggregate(results))

        report = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        return report

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_page(m: Stage1Metrics) -> list[str]:
        cq = m.character_quality
        mq = m.markup_quality
        ro = m.read_order
        return [
            f"--- {m.page_id} ---",
            "",
            "  Character Recognition Quality:",
            f"    GCER: {cq.gcer:.4f}  ({cq.gcer*100:.2f}%)",
            f"    WER:  {cq.wer:.4f}  ({cq.wer*100:.2f}%)",
            f"    NED:  {cq.ned:.4f}",
            f"    Lines: {cq.matched_lines} matched, {cq.missing_lines} missing, {cq.extra_lines} extra",
            "",
            "  Markup / Typography Preservation:",
            f"    Bold   — P: {mq.bold.precision:.4f}  R: {mq.bold.recall:.4f}  F1: {mq.bold.f1:.4f}  (TP={mq.bold.true_positives} FP={mq.bold.false_positives} FN={mq.bold.false_negatives})",
            f"    Italic — P: {mq.italic.precision:.4f}  R: {mq.italic.recall:.4f}  F1: {mq.italic.f1:.4f}  (TP={mq.italic.true_positives} FP={mq.italic.false_positives} FN={mq.italic.false_negatives})",
            "",
            "  Read Order (Structure Preservation):",
            f"    NED:                    {ro.ned:.4f}",
            f"    Character NED baseline: {ro.character_ned:.4f}",
            f"    Isolated order error:   {ro.isolated_order_error:.4f}",
        ]

    def _format_aggregate(self, results: list[Stage1Metrics]) -> list[str]:
        agg = self._aggregate(results)
        return [
            "=" * 72,
            f"AGGREGATE ({len(results)} pages)",
            "=" * 72,
            "",
            "  Character Recognition Quality:",
            f"    GCER: {agg['character_quality']['GCER']:.4f}",
            f"    WER:  {agg['character_quality']['WER']:.4f}",
            f"    NED:  {agg['character_quality']['NED']:.4f}",
            "",
            "  Markup / Typography Preservation:",
            f"    Bold   — P: {agg['markup_quality']['bold']['precision']:.4f}  R: {agg['markup_quality']['bold']['recall']:.4f}  F1: {agg['markup_quality']['bold']['f1']:.4f}",
            f"    Italic — P: {agg['markup_quality']['italic']['precision']:.4f}  R: {agg['markup_quality']['italic']['recall']:.4f}  F1: {agg['markup_quality']['italic']['f1']:.4f}",
            "",
            "  Read Order (Structure Preservation):",
            f"    NED:                    {agg['read_order']['NED']:.4f}",
            f"    Isolated order error:   {agg['read_order']['isolated_order_error']:.4f}",
        ]

    @staticmethod
    def _aggregate(results: list[Stage1Metrics]) -> dict:
        """Micro-average across all pages."""
        # Character quality: micro-average using totals
        total_char_edits = sum(m.character_quality.total_char_edits for m in results)
        total_chars_gold = sum(m.character_quality.total_chars_gold for m in results)
        total_chars_pred = sum(m.character_quality.total_chars_pred for m in results)
        total_word_edits = sum(m.character_quality.total_word_edits for m in results)
        total_words_gold = sum(m.character_quality.total_words_gold for m in results)

        agg_gcer = total_char_edits / total_chars_gold if total_chars_gold else 0.0
        agg_wer = total_word_edits / total_words_gold if total_words_gold else 0.0
        agg_ned = total_char_edits / max(total_chars_gold, total_chars_pred, 1)

        # Markup: sum TP/FP/FN
        bold_tp = sum(m.markup_quality.bold.true_positives for m in results)
        bold_fp = sum(m.markup_quality.bold.false_positives for m in results)
        bold_fn = sum(m.markup_quality.bold.false_negatives for m in results)
        ital_tp = sum(m.markup_quality.italic.true_positives for m in results)
        ital_fp = sum(m.markup_quality.italic.false_positives for m in results)
        ital_fn = sum(m.markup_quality.italic.false_negatives for m in results)

        def _prf(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            return round(p, 4), round(r, 4), round(f1, 4)

        bp, br, bf = _prf(bold_tp, bold_fp, bold_fn)
        ip, ir, if1 = _prf(ital_tp, ital_fp, ital_fn)

        # Read order: macro-average NED
        ro_ned = sum(m.read_order.ned for m in results) / len(results)
        ro_char_baseline = sum(m.read_order.character_ned for m in results) / len(results)
        ro_iso = sum(m.read_order.isolated_order_error for m in results) / len(results)

        return {
            "character_quality": {
                "GCER": round(agg_gcer, 6),
                "WER": round(agg_wer, 6),
                "NED": round(agg_ned, 6),
            },
            "markup_quality": {
                "bold": {"precision": bp, "recall": br, "f1": bf},
                "italic": {"precision": ip, "recall": ir, "f1": if1},
            },
            "read_order": {
                "NED": round(ro_ned, 6),
                "character_NED_baseline": round(ro_char_baseline, 6),
                "isolated_order_error": round(ro_iso, 6),
            },
        }
