"""
Stage1Evaluator: orchestrates all Stage 1 OCR evaluation metrics.

Loads predicted and gold TSV files, runs character quality, markup quality,
and read-order evaluations, then generates JSON + human-readable reports.
"""

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

MetricsProfile = Literal["full", "minimal"]

from dictextractor.evaluation.stage1.alignment import align_rows
from dictextractor.evaluation.stage1.character_quality import compute_character_quality
from dictextractor.evaluation.stage1.markup_quality import compute_markup_quality
from dictextractor.evaluation.stage1.read_order import compute_read_order
from dictextractor.evaluation.stage1.stage1_metrics import Stage1Metrics

Row = Dict[str, str]


@dataclass(frozen=True)
class EvalTask:
    """One predicted vs gold evaluation unit under ``samples_dir``."""

    experiment: str
    pred_path: Path
    gold_path: Path
    page_id: str

# column_id values that are page-level metadata, not body content.
# These rows are excluded from every Stage 1 metric — predictions emit them but
# existing gold TSVs predate the change, so including them would inflate
# extra spans / FP counts on predictions only.
_METADATA_COLUMN_IDS = {"header", "footer"}


def _strip_metadata(rows: List[Row]) -> List[Row]:
    """Drop rows whose column_id is `header` or `footer`."""
    return [r for r in rows if r.get("column_id") not in _METADATA_COLUMN_IDS]


class Stage1Evaluator:
    """Evaluate a single predicted Stage 1 TSV against a gold TSV."""

    _FULL_METRIC_CSV_COLS = [
        "TextEdit", "GCER", "WER",
        "typography_f1",
        "bold_precision", "bold_recall", "bold_f1",
        "italic_precision", "italic_recall", "italic_f1",
        "ReadOrderEdit",
    ]

    _MINIMAL_METRIC_CSV_COLS = [
        "TextEdit", "GCER", "WER", "typography_f1", "ReadOrderEdit",
    ]

    def __init__(
        self,
        metrics_profile: MetricsProfile = "full",
        *,
        alignment_threshold: float = 0.5,
        alignment_max_span_rows: int = 3,
    ) -> None:
        if metrics_profile not in ("full", "minimal"):
            raise ValueError(
                f"metrics_profile must be 'full' or 'minimal', got {metrics_profile!r}"
            )
        self.metrics_profile = metrics_profile
        self.alignment_threshold = alignment_threshold
        self.alignment_max_span_rows = alignment_max_span_rows

    def _metric_csv_cols(self) -> List[str]:
        if self.metrics_profile == "minimal":
            return list(self._MINIMAL_METRIC_CSV_COLS)
        return list(self._FULL_METRIC_CSV_COLS)

    @staticmethod
    def _pick_columns(row: dict, columns: List[str]) -> dict:
        return {col: row[col] for col in columns}

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
        pred_rows = _strip_metadata(self.load_tsv(pred_path))
        gold_rows = _strip_metadata(self.load_tsv(gold_path))

        alignment = align_rows(
            pred_rows,
            gold_rows,
            threshold=self.alignment_threshold,
            max_span_rows=self.alignment_max_span_rows,
        )
        char_q = compute_character_quality(alignment)
        markup_q = compute_markup_quality(alignment)
        read_o = compute_read_order(alignment)

        return Stage1Metrics(
            page_id=page_id,
            character_quality=char_q,
            markup_quality=markup_q,
            read_order=read_o,
        )

    # ------------------------------------------------------------------
    # Batch / experiment evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def discover_tasks(
        samples_dir: str | Path,
        experiments: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
    ) -> List[EvalTask]:
        """List every (experiment, page) pair whose gold and prediction both exist.

        Layout (per language root under ``samples_dir``):
            <lang>/outputs/stage-1-gold/<stem>/<stem>_stage1_GOLD.tsv   (gold)
            <lang>/outputs/stage-1/<experiment>/<stem>/<stem>_stage1.tsv (pred)

        When ``experiments`` is ``None``, every non-hidden experiment subfolder
        under each language's ``outputs/stage-1/`` is included.
        """
        samples_dir = Path(samples_dir)
        tasks: List[EvalTask] = []
        selected_languages = set(languages) if languages else None

        golds_by_lang: Dict[Path, List[Path]] = {}
        for gold_path in sorted(
            samples_dir.glob("*/outputs/stage-1-gold/*/*_stage1_GOLD.tsv")
        ):
            parents = list(gold_path.parents)
            if selected_languages and parents[3].name not in selected_languages:
                continue
            golds_by_lang.setdefault(parents[3], []).append(gold_path)

        for lang_dir, gold_paths in sorted(golds_by_lang.items()):
            stage1_root = lang_dir / "outputs" / "stage-1"
            available = (
                sorted(
                    p.name for p in stage1_root.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                )
                if stage1_root.is_dir() else []
            )
            if experiments is None:
                exp_names = available
                if not exp_names:
                    print(f"  [skip] no experiments under {stage1_root}")
                    continue
            else:
                exp_names = [e for e in experiments if e in available]
                for missing in (set(experiments) - set(available)):
                    print(
                        f"  [warn] experiment {missing!r} not found under {stage1_root}"
                    )

            for exp in exp_names:
                for gold_path in gold_paths:
                    stem = gold_path.parent.name
                    pred_path = stage1_root / exp / stem / f"{stem}_stage1.tsv"
                    if not pred_path.exists():
                        print(
                            f"  [skip] no prediction for {lang_dir.name}/{stem} "
                            f"in experiment {exp!r} ({pred_path})"
                        )
                        continue
                    page_id = f"{lang_dir.name}/{stem}"
                    tasks.append(
                        EvalTask(
                            experiment=exp,
                            pred_path=pred_path,
                            gold_path=gold_path,
                            page_id=page_id,
                        )
                    )

        return tasks

    def evaluate_experiments(
        self,
        samples_dir: str | Path,
        experiments: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
    ) -> List[Tuple[str, Stage1Metrics]]:
        """Pair each gold TSV with every requested experiment's prediction and
        evaluate. See ``discover_tasks`` for path conventions.
        """
        results: List[Tuple[str, Stage1Metrics]] = []
        for task in self.discover_tasks(samples_dir, experiments, languages):
            metrics = self.evaluate(
                task.pred_path, task.gold_path, page_id=task.page_id
            )
            results.append((task.experiment, metrics))
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
            "read_order": {
                "ReadOrderEdit": round(ro.read_order_edit, 6),
                "edit_distance": ro.edit_distance,
                "max_length": ro.max_length,
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

    def _char_csv_cols(self) -> List[str]:
        return ["page_id", "TextEdit", "GCER", "WER"]

    def _markup_csv_cols(self) -> List[str]:
        if self.metrics_profile == "minimal":
            return ["page_id", "typography_f1"]
        return [
            "page_id",
            "typography_f1",
            "bold_precision", "bold_recall", "bold_f1",
            "italic_precision", "italic_recall", "italic_f1",
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
            char_rows.append(self._pick_columns({
                "page_id": "__aggregate__",
                **self._metrics_from_aggregate(agg),
            }, self._char_csv_cols()))
            markup_rows.append(self._pick_columns({
                "page_id": "__aggregate__",
                **self._metrics_from_aggregate(agg),
            }, self._markup_csv_cols()))
            order_rows.append(self._pick_columns({
                "page_id": "__aggregate__",
                **self._metrics_from_aggregate(agg),
            }, self._order_csv_cols()))

        char_rows = [self._pick_columns(r, self._char_csv_cols()) for r in char_rows]
        markup_rows = [self._pick_columns(r, self._markup_csv_cols()) for r in markup_rows]
        order_rows = [self._pick_columns(r, self._order_csv_cols()) for r in order_rows]

        self._write_csv(char_rows, self._char_csv_cols(),
                         output_dir / "character_recognition.csv")
        self._write_csv(markup_rows, self._markup_csv_cols(),
                         output_dir / "markup_preservation.csv")
        self._write_csv(order_rows, self._order_csv_cols(),
                         output_dir / "structure_preservation.csv")

    # -- Cross-experiment CSV exports (detailed + per-language summary) --

    def _detailed_csv_cols(self) -> List[str]:
        return [
            "experiment", "alphabet", "ocr-hint", "page_id",
            *self._metric_csv_cols(),
        ]

    def _summary_csv_cols(self) -> List[str]:
        return [
            "experiment", "language", "alphabet", "ocr-hint", "page_count",
            *self._metric_csv_cols(),
        ]

    @staticmethod
    def _bool_csv(value: bool) -> str:
        return "true" if value else "false"

    @staticmethod
    def _parse_page_id(page_id: str) -> Optional[Tuple[str, str]]:
        """Return ``(language, stem)`` or ``None`` for aggregate rows."""
        if page_id == "__aggregate__":
            return None
        lang, _, stem = page_id.partition("/")
        return (lang, stem) if lang else None

    @staticmethod
    def _load_run_config_flags(
        samples_dir: Path, language: str, experiment: str
    ) -> Tuple[bool, bool]:
        """Read ``alphabet.used`` and ``ocr_hint.used`` from a run manifest."""
        path = (
            samples_dir / language / "outputs" / "stage-1"
            / experiment / "run_config.json"
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

    @staticmethod
    def _metrics_from_aggregate(agg: dict) -> dict:
        return {
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
            "ReadOrderEdit": agg["read_order"]["ReadOrderEdit"],
        }

    @staticmethod
    def _metrics_from_stage1(m: Stage1Metrics) -> dict:
        cq, mq, ro = m.character_quality, m.markup_quality, m.read_order
        return {
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
            "ReadOrderEdit": round(ro.read_order_edit, 6),
        }

    def _detailed_csv_row(
        self,
        experiment: str,
        m: Stage1Metrics,
        samples_dir: Path,
    ) -> dict:
        parsed = self._parse_page_id(m.page_id)
        if parsed is None:
            alphabet_used: Union[bool, str] = ""
            ocr_hint_used: Union[bool, str] = ""
        else:
            language, _ = parsed
            alphabet_used, ocr_hint_used = self._load_run_config_flags(
                samples_dir, language, experiment
            )
        row = {
            "experiment": experiment,
            "alphabet": (
                self._bool_csv(alphabet_used)
                if isinstance(alphabet_used, bool) else ""
            ),
            "ocr-hint": (
                self._bool_csv(ocr_hint_used)
                if isinstance(ocr_hint_used, bool) else ""
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
    ) -> None:
        """Long-format CSV: one row per (experiment, page) plus a per-experiment
        ``__aggregate__`` row. ``alphabet`` and ``ocr-hint`` come from each
        language's ``run_config.json``."""
        samples_dir = Path(samples_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows: List[dict] = []
        for exp, results in results_by_exp.items():
            for m in results:
                rows.append(self._detailed_csv_row(exp, m, samples_dir))
            if len(results) > 1:
                agg = self._aggregate(results)
                rows.append(self._pick_columns({
                    "experiment": exp,
                    "alphabet": "",
                    "ocr-hint": "",
                    "page_id": "__aggregate__",
                    **self._metrics_from_aggregate(agg),
                }, self._detailed_csv_cols()))
        self._write_csv(rows, self._detailed_csv_cols(), output_path)

    def generate_summary_csv(
        self,
        results_by_exp: Dict[str, List[Stage1Metrics]],
        samples_dir: str | Path,
        output_path: str | Path,
    ) -> None:
        """Per-(experiment, language) micro-aggregated metrics across pages."""
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
                    samples_dir, language, exp
                )
                row = {
                    "experiment": exp,
                    "language": language,
                    "alphabet": self._bool_csv(alphabet_used),
                    "ocr-hint": self._bool_csv(ocr_hint_used),
                    "page_count": len(lang_results),
                    **self._metrics_from_aggregate(
                        self._aggregate(lang_results)
                    ),
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
            "",
            "  Read Order (Structure Preservation):",
            f"    ReadOrderEdit: {ro.read_order_edit:.4f}",
        ]

    def _format_aggregate(self, results: list[Stage1Metrics]) -> list[str]:
        agg = self._aggregate(results)
        return [
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
            "",
            "  Read Order (Structure Preservation):",
            f"    ReadOrderEdit: {agg['read_order']['ReadOrderEdit']:.4f}",
        ]

    @staticmethod
    def _aggregate(results: list[Stage1Metrics]) -> dict:
        """Micro-average across all pages."""
        # Character quality: micro-average using totals
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
        tp, tr, tf = _prf(
            bold_tp + ital_tp, bold_fp + ital_fp, bold_fn + ital_fn
        )

        # Read order: macro-average by page.
        ro_edit = sum(m.read_order.read_order_edit for m in results) / len(results)

        return {
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
            "read_order": {
                "ReadOrderEdit": round(ro_edit, 6),
            },
        }
