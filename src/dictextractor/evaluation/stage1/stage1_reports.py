"""JSON/CSV/text report generation for Stage 1 evaluation metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Literal, Union

from dictextractor.evaluation.stage1.stage1_metrics import Stage1Metrics

MetricsProfile = Literal["full", "minimal"]


class Stage1ReportWriter:
    """Write Stage 1 evaluation reports from ``Stage1Metrics`` results."""

    _FULL_METRIC_CSV_COLS = [
        "TextEdit",
        "GCER",
        "WER",
        "typography_f1",
        "bold_precision",
        "bold_recall",
        "bold_f1",
        "italic_precision",
        "italic_recall",
        "italic_f1",
    ]
    _MINIMAL_METRIC_CSV_COLS = [
        "TextEdit",
        "GCER",
        "WER",
        "typography_f1",
    ]
    _READ_ORDER_COL = "ReadOrderEdit"

    def __init__(
        self,
        metrics_profile: MetricsProfile = "full",
        *,
        include_read_order: bool = False,
    ) -> None:
        if metrics_profile not in ("full", "minimal"):
            raise ValueError(
                f"metrics_profile must be 'full' or 'minimal', got {metrics_profile!r}"
            )
        self.metrics_profile = metrics_profile
        self.include_read_order = include_read_order

    def _metric_csv_cols(self) -> List[str]:
        cols = (
            list(self._MINIMAL_METRIC_CSV_COLS)
            if self.metrics_profile == "minimal"
            else list(self._FULL_METRIC_CSV_COLS)
        )
        if self.include_read_order:
            cols = [*cols, self._READ_ORDER_COL]
        return cols

    @staticmethod
    def _pick_columns(row: dict, columns: List[str]) -> dict:
        return {col: row[col] for col in columns}

    @staticmethod
    def metrics_to_dict(m: Stage1Metrics, *, include_read_order: bool = False) -> dict:
        cq = m.character_quality
        mq = m.markup_quality
        out: dict = {
            "page_id": m.page_id,
            "character_quality": {
                "TextEdit": round(cq.text_edit, 6),
                "GCER": round(cq.gcer, 6),
                "WER": round(cq.wer, 6),
                "total_graphemes_gold": cq.total_graphemes_gold,
                "total_graphemes_pred": cq.total_graphemes_pred,
                "total_grapheme_edits": cq.total_grapheme_edits,
                "total_words_gold": cq.total_words_gold,
                "total_word_edits": cq.total_word_edits,
                "matched_spans": cq.matched_spans,
                "missing_spans": cq.missing_spans,
                "extra_spans": cq.extra_spans,
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
                "typography": {
                    "precision": round(mq.typography.precision, 4),
                    "recall": round(mq.typography.recall, 4),
                    "f1": round(mq.typography.f1, 4),
                    "tp": mq.typography.true_positives,
                    "fp": mq.typography.false_positives,
                    "fn": mq.typography.false_negatives,
                },
            },
        }
        if include_read_order:
            ro = m.read_order
            out["read_order"] = {
                "ReadOrderEdit": round(ro.read_order_edit, 6),
                "edit_distance": ro.edit_distance,
                "max_length": ro.max_length,
            }
        return out

    def generate_json_report(
        self,
        results: list[Stage1Metrics],
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            self.metrics_to_dict(m, include_read_order=self.include_read_order)
            for m in results
        ]
        if len(results) > 1:
            agg = self._aggregate(results)
            data.append({"page_id": "__aggregate__", **agg})

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _char_csv_cols(self) -> List[str]:
        return ["page_id", "TextEdit", "GCER", "WER"]

    def _markup_csv_cols(self) -> List[str]:
        if self.metrics_profile == "minimal":
            return ["page_id", "typography_f1"]
        return [
            "page_id",
            "typography_f1",
            "bold_precision",
            "bold_recall",
            "bold_f1",
            "italic_precision",
            "italic_recall",
            "italic_f1",
        ]

    def _order_csv_cols(self) -> List[str]:
        return ["page_id", "ReadOrderEdit"]

    @staticmethod
    def _char_csv_row(m: Stage1Metrics) -> dict:
        cq = m.character_quality
        return {
            "page_id": m.page_id,
            "TextEdit": round(cq.text_edit, 6),
            "GCER": round(cq.gcer, 6),
            "WER": round(cq.wer, 6),
        }

    @staticmethod
    def _markup_csv_row(m: Stage1Metrics) -> dict:
        mq = m.markup_quality
        return {
            "page_id": m.page_id,
            "typography_f1": round(mq.typography.f1, 4),
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
            "ReadOrderEdit": round(ro.read_order_edit, 6),
        }

    def _write_csv(self, rows: list[dict], columns: list[str], path: Path) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def generate_csv_reports(
        self,
        results: list[Stage1Metrics],
        output_dir: str | Path,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        char_rows = [self._char_csv_row(m) for m in results]
        markup_rows = [self._markup_csv_row(m) for m in results]

        if len(results) > 1:
            agg = self._aggregate(results)
            char_rows.append(
                self._pick_columns(
                    {"page_id": "__aggregate__", **self._metrics_from_aggregate(agg)},
                    self._char_csv_cols(),
                )
            )
            markup_rows.append(
                self._pick_columns(
                    {"page_id": "__aggregate__", **self._metrics_from_aggregate(agg)},
                    self._markup_csv_cols(),
                )
            )

        char_rows = [self._pick_columns(r, self._char_csv_cols()) for r in char_rows]
        markup_rows = [self._pick_columns(r, self._markup_csv_cols()) for r in markup_rows]

        self._write_csv(char_rows, self._char_csv_cols(), output_dir / "character_recognition.csv")
        self._write_csv(markup_rows, self._markup_csv_cols(), output_dir / "markup_preservation.csv")
        if self.include_read_order:
            order_rows = [self._order_csv_row(m) for m in results]
            if len(results) > 1:
                agg = self._aggregate(results)
                order_rows.append(
                    self._pick_columns(
                        {"page_id": "__aggregate__", **self._metrics_from_aggregate(agg)},
                        self._order_csv_cols(),
                    )
                )
            order_rows = [self._pick_columns(r, self._order_csv_cols()) for r in order_rows]
            self._write_csv(
                order_rows,
                self._order_csv_cols(),
                output_dir / "structure_preservation.csv",
            )

    def _detailed_csv_cols(self) -> List[str]:
        return [
            "experiment",
            "alphabet",
            "ocr-hint",
            "page_id",
            *self._metric_csv_cols(),
        ]

    def _summary_csv_cols(self) -> List[str]:
        return [
            "experiment",
            "language",
            "alphabet",
            "ocr-hint",
            "page_count",
            *self._metric_csv_cols(),
        ]

    @staticmethod
    def _bool_csv(value: bool) -> str:
        return "true" if value else "false"

    @staticmethod
    def _parse_page_id(page_id: str) -> tuple[str, str] | None:
        if page_id == "__aggregate__":
            return None
        lang, _, stem = page_id.partition("/")
        return (lang, stem) if lang else None

    @staticmethod
    def _load_run_config_flags(
        samples_dir: Path,
        language: str,
        experiment: str,
        *,
        stage1_output_subdir: str = "stage-1",
    ) -> tuple[bool, bool]:
        path = (
            samples_dir
            / language
            / "outputs"
            / stage1_output_subdir
            / experiment
            / "run_config.json"
        )
        if not path.is_file():
            return False, False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False, False
        alphabet = bool(data.get("alphabet", {}).get("used", False))
        ocr_hint = bool(data.get("ocr_hint", {}).get("used", False))
        return alphabet, ocr_hint

    def _metrics_from_aggregate(self, agg: dict) -> dict:
        out = {
            "TextEdit": agg["character_quality"]["TextEdit"],
            "GCER": agg["character_quality"]["GCER"],
            "WER": agg["character_quality"]["WER"],
            "typography_f1": agg["markup_quality"]["typography"]["f1"],
            "bold_precision": agg["markup_quality"]["bold"]["precision"],
            "bold_recall": agg["markup_quality"]["bold"]["recall"],
            "bold_f1": agg["markup_quality"]["bold"]["f1"],
            "italic_precision": agg["markup_quality"]["italic"]["precision"],
            "italic_recall": agg["markup_quality"]["italic"]["recall"],
            "italic_f1": agg["markup_quality"]["italic"]["f1"],
        }
        if self.include_read_order:
            out["ReadOrderEdit"] = agg["read_order"]["ReadOrderEdit"]
        return out

    def _metrics_from_stage1(self, m: Stage1Metrics) -> dict:
        cq, mq = m.character_quality, m.markup_quality
        out = {
            "TextEdit": round(cq.text_edit, 6),
            "GCER": round(cq.gcer, 6),
            "WER": round(cq.wer, 6),
            "typography_f1": round(mq.typography.f1, 4),
            "bold_precision": round(mq.bold.precision, 4),
            "bold_recall": round(mq.bold.recall, 4),
            "bold_f1": round(mq.bold.f1, 4),
            "italic_precision": round(mq.italic.precision, 4),
            "italic_recall": round(mq.italic.recall, 4),
            "italic_f1": round(mq.italic.f1, 4),
        }
        if self.include_read_order:
            out["ReadOrderEdit"] = round(m.read_order.read_order_edit, 6)
        return out

    def _detailed_csv_row(
        self,
        experiment: str,
        m: Stage1Metrics,
        samples_dir: Path,
        *,
        stage1_output_subdir: str = "stage-1",
    ) -> dict:
        parsed = self._parse_page_id(m.page_id)
        if parsed is None:
            alphabet_used: Union[bool, str] = ""
            ocr_hint_used: Union[bool, str] = ""
        else:
            language, _ = parsed
            alphabet_used, ocr_hint_used = self._load_run_config_flags(
                samples_dir,
                language,
                experiment,
                stage1_output_subdir=stage1_output_subdir,
            )
        row = {
            "experiment": experiment,
            "alphabet": (
                self._bool_csv(alphabet_used) if isinstance(alphabet_used, bool) else ""
            ),
            "ocr-hint": (
                self._bool_csv(ocr_hint_used) if isinstance(ocr_hint_used, bool) else ""
            ),
            "page_id": m.page_id,
            **self._metrics_from_stage1(m),
        }
        return self._pick_columns(row, self._detailed_csv_cols())

    def generate_detailed_csv(
        self,
        results_by_exp: Dict[str, List[Stage1Metrics]],
        samples_dir: str | Path,
        output_path: str | Path,
        *,
        stage1_output_subdir: str = "stage-1",
    ) -> None:
        samples_dir = Path(samples_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows: List[dict] = []
        for exp, results in results_by_exp.items():
            for m in results:
                rows.append(
                    self._detailed_csv_row(
                        exp,
                        m,
                        samples_dir,
                        stage1_output_subdir=stage1_output_subdir,
                    )
                )
            if len(results) > 1:
                agg = self._aggregate(results)
                rows.append(
                    self._pick_columns(
                        {
                            "experiment": exp,
                            "alphabet": "",
                            "ocr-hint": "",
                            "page_id": "__aggregate__",
                            **self._metrics_from_aggregate(agg),
                        },
                        self._detailed_csv_cols(),
                    )
                )
        self._write_csv(rows, self._detailed_csv_cols(), output_path)

    def generate_summary_csv(
        self,
        results_by_exp: Dict[str, List[Stage1Metrics]],
        samples_dir: str | Path,
        output_path: str | Path,
        *,
        stage1_output_subdir: str = "stage-1",
    ) -> None:
        samples_dir = Path(samples_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows: List[dict] = []
        for exp, results in results_by_exp.items():
            by_lang: Dict[str, List[Stage1Metrics]] = {}
            for m in results:
                parsed = self._parse_page_id(m.page_id)
                if parsed is None:
                    continue
                language, _ = parsed
                by_lang.setdefault(language, []).append(m)

            for language in sorted(by_lang):
                lang_results = by_lang[language]
                alphabet_used, ocr_hint_used = self._load_run_config_flags(
                    samples_dir,
                    language,
                    exp,
                    stage1_output_subdir=stage1_output_subdir,
                )
                row = {
                    "experiment": exp,
                    "language": language,
                    "alphabet": self._bool_csv(alphabet_used),
                    "ocr-hint": self._bool_csv(ocr_hint_used),
                    "page_count": len(lang_results),
                    **self._metrics_from_aggregate(self._aggregate(lang_results)),
                }
                rows.append(self._pick_columns(row, self._summary_csv_cols()))
        self._write_csv(rows, self._summary_csv_cols(), output_path)

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

    def _format_page(self, m: Stage1Metrics) -> list[str]:
        cq = m.character_quality
        mq = m.markup_quality
        lines = [
            f"--- {m.page_id} ---",
            "",
            "  Character Recognition Quality:",
            f"    TextEdit: {cq.text_edit:.4f}",
            f"    GCER: {cq.gcer:.4f}  ({cq.gcer*100:.2f}%)",
            f"    WER:  {cq.wer:.4f}  ({cq.wer*100:.2f}%)",
            f"    Spans: {cq.matched_spans} matched, {cq.missing_spans} missing, {cq.extra_spans} extra",
            "",
            "  Markup / Typography Preservation:",
            f"    Typography (bold+italic pooled) — F1: {mq.typography.f1:.4f}  "
            f"(TP={mq.typography.true_positives} FP={mq.typography.false_positives} "
            f"FN={mq.typography.false_negatives})",
            f"    Bold   — P: {mq.bold.precision:.4f}  R: {mq.bold.recall:.4f}  F1: {mq.bold.f1:.4f}  (TP={mq.bold.true_positives} FP={mq.bold.false_positives} FN={mq.bold.false_negatives})",
            f"    Italic — P: {mq.italic.precision:.4f}  R: {mq.italic.recall:.4f}  F1: {mq.italic.f1:.4f}  (TP={mq.italic.true_positives} FP={mq.italic.false_positives} FN={mq.italic.false_negatives})",
        ]
        if self.include_read_order:
            ro = m.read_order
            lines.extend(
                [
                    "",
                    "  Read Order (Structure Preservation):",
                    f"    ReadOrderEdit: {ro.read_order_edit:.4f}",
                ]
            )
        return lines

    def _format_aggregate(self, results: list[Stage1Metrics]) -> list[str]:
        agg = self._aggregate(results)
        lines = [
            "=" * 72,
            f"AGGREGATE ({len(results)} pages)",
            "=" * 72,
            "",
            "  Character Recognition Quality:",
            f"    TextEdit: {agg['character_quality']['TextEdit']:.4f}",
            f"    GCER: {agg['character_quality']['GCER']:.4f}",
            f"    WER:  {agg['character_quality']['WER']:.4f}",
            "",
            "  Markup / Typography Preservation:",
            f"    Typography (bold+italic pooled) — F1: {agg['markup_quality']['typography']['f1']:.4f}",
            f"    Bold   — P: {agg['markup_quality']['bold']['precision']:.4f}  R: {agg['markup_quality']['bold']['recall']:.4f}  F1: {agg['markup_quality']['bold']['f1']:.4f}",
            f"    Italic — P: {agg['markup_quality']['italic']['precision']:.4f}  R: {agg['markup_quality']['italic']['recall']:.4f}  F1: {agg['markup_quality']['italic']['f1']:.4f}",
        ]
        if self.include_read_order:
            lines.extend(
                [
                    "",
                    "  Read Order (Structure Preservation):",
                    f"    ReadOrderEdit: {agg['read_order']['ReadOrderEdit']:.4f}",
                ]
            )
        return lines

    @staticmethod
    def _aggregate(results: list[Stage1Metrics]) -> dict:
        total_grapheme_edits = sum(m.character_quality.total_grapheme_edits for m in results)
        total_graphemes_gold = sum(m.character_quality.total_graphemes_gold for m in results)
        total_word_edits = sum(m.character_quality.total_word_edits for m in results)
        total_words_gold = sum(m.character_quality.total_words_gold for m in results)
        total_spans = sum(
            m.character_quality.matched_spans
            + m.character_quality.missing_spans
            + m.character_quality.extra_spans
            for m in results
        )
        text_edit_sum = sum(
            m.character_quality.text_edit
            * (
                m.character_quality.matched_spans
                + m.character_quality.missing_spans
                + m.character_quality.extra_spans
            )
            for m in results
        )

        agg_gcer = total_grapheme_edits / total_graphemes_gold if total_graphemes_gold else 0.0
        agg_wer = total_word_edits / total_words_gold if total_words_gold else 0.0
        agg_text_edit = text_edit_sum / total_spans if total_spans else 0.0

        bold_tp = sum(m.markup_quality.bold.true_positives for m in results)
        bold_fp = sum(m.markup_quality.bold.false_positives for m in results)
        bold_fn = sum(m.markup_quality.bold.false_negatives for m in results)
        ital_tp = sum(m.markup_quality.italic.true_positives for m in results)
        ital_fp = sum(m.markup_quality.italic.false_positives for m in results)
        ital_fn = sum(m.markup_quality.italic.false_negatives for m in results)

        def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            return round(p, 4), round(r, 4), round(f1, 4)

        bp, br, bf = _prf(bold_tp, bold_fp, bold_fn)
        ip, ir, if1 = _prf(ital_tp, ital_fp, ital_fn)
        tp, tr, tf = _prf(bold_tp + ital_tp, bold_fp + ital_fp, bold_fn + ital_fn)

        result = {
            "character_quality": {
                "TextEdit": round(agg_text_edit, 6),
                "GCER": round(agg_gcer, 6),
                "WER": round(agg_wer, 6),
            },
            "markup_quality": {
                "bold": {"precision": bp, "recall": br, "f1": bf},
                "italic": {"precision": ip, "recall": ir, "f1": if1},
                "typography": {"precision": tp, "recall": tr, "f1": tf},
            },
        }
        result["read_order"] = {
            "ReadOrderEdit": round(
                sum(m.read_order.read_order_edit for m in results) / len(results),
                6,
            )
        }
        return result
