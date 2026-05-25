"""eval-flat: per-page flat text vs gold flat (spec v2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from dictextractor.evaluation.stage1.alignment import (
    align_lines_quick_match,
    align_page_collapsed,
)
from dictextractor.evaluation.stage1.character_quality import compute_character_quality
from dictextractor.evaluation.stage1.flatten import (
    flatten_stage1_tsv,
    load_flat_lines,
)
from dictextractor.evaluation.stage1.markup_quality import compute_markup_quality
from dictextractor.evaluation.stage1.read_order import (
    compute_read_order,
    compute_read_order_collapsed,
)
from dictextractor.evaluation.stage1.stage1_reports import Stage1ReportWriter
from dictextractor.evaluation.stage1.stage1_metrics import Stage1Metrics

Row = Dict[str, str]
MetricsProfile = Literal["full", "minimal"]
CharacterAlignmentMode = Literal["collapsed", "quick_match"]


@dataclass(frozen=True)
class FlatEvalTask:
    """One flat predicted vs flat gold evaluation unit."""

    experiment: str
    pred_path: Path
    gold_path: Path
    page_id: str


def _lines_to_rows(lines: List[str]) -> List[Row]:
    return [
        {"column_id": "single", "line_number": str(i), "text": line}
        for i, line in enumerate(lines, start=1)
    ]


def _load_pred_lines(pred_path: Path) -> List[str]:
    if pred_path.suffix == ".tsv" or pred_path.name.endswith("_stage1.tsv"):
        text = flatten_stage1_tsv(pred_path)
        return text.splitlines() if text else []
    return load_flat_lines(pred_path)


class FlatStage1Evaluator:
    """Evaluate flat stage-1 predictions against flat gold.

    Character and typography default to OmniDocBench quick_match (line-level
    Adjacency Search Match). ReadOrderEdit uses the same matched pairs (quick_match)
    or anchor search in the collapsed pred string (collapsed mode).
    """

    def __init__(
        self,
        metrics_profile: MetricsProfile = "full",
        *,
        character_alignment: CharacterAlignmentMode = "quick_match",
        alignment_threshold: float = 0.6,
    ) -> None:
        self.metrics_profile = metrics_profile
        self.character_alignment = character_alignment
        self.alignment_threshold = alignment_threshold
        self._report_helper = Stage1ReportWriter(
            metrics_profile=metrics_profile,
            include_read_order=True,
        )

    def _metric_csv_cols(self) -> List[str]:
        return self._report_helper._metric_csv_cols()

    def evaluate(
        self,
        pred_path: str | Path,
        gold_path: str | Path,
        page_id: str = "",
    ) -> Stage1Metrics:
        pred_lines = _load_pred_lines(Path(pred_path))
        gold_lines = load_flat_lines(gold_path)
        pred_rows = _lines_to_rows(pred_lines)
        gold_rows = _lines_to_rows(gold_lines)
        if self.character_alignment == "collapsed":
            content_alignment = align_page_collapsed(pred_rows, gold_rows)
            read_order = compute_read_order_collapsed(gold_lines, pred_lines)
        else:
            content_alignment = align_lines_quick_match(pred_rows, gold_rows)
            read_order = compute_read_order(content_alignment)
        char_q = compute_character_quality(content_alignment)
        markup_q = compute_markup_quality(content_alignment)
        return Stage1Metrics(
            page_id=page_id,
            character_quality=char_q,
            markup_quality=markup_q,
            read_order=read_order,
        )

    @staticmethod
    def discover_tasks(
        samples_dir: str | Path,
        experiments: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        stage1_output_subdir: str = "stage-1",
    ) -> List[FlatEvalTask]:
        """
        Discover (experiment, page) pairs with flat gold and a flat or column pred.

        Gold: ``*/outputs/stage-1-gold/*/*_stage1_GOLD_flat.txt``
        Pred: ``*/outputs/<subdir>/<exp>/*/*_stage1_flat.txt`` or ``*_stage1.tsv``
        """
        samples_dir = Path(samples_dir)
        tasks: List[FlatEvalTask] = []
        selected_languages = set(languages) if languages else None

        golds_by_lang: Dict[Path, List[Path]] = {}
        for gold_path in sorted(
            samples_dir.glob("*/outputs/stage-1-gold/*/*_stage1_GOLD_flat.txt")
        ):
            lang = gold_path.parts[-5]
            if selected_languages and lang not in selected_languages:
                continue
            golds_by_lang.setdefault(gold_path.parents[3], []).append(gold_path)

        for lang_dir, gold_paths in sorted(golds_by_lang.items()):
            stage1_root = lang_dir / "outputs" / stage1_output_subdir
            if not stage1_root.is_dir():
                continue
            available = sorted(
                p.name
                for p in stage1_root.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
            exp_names = available if experiments is None else [
                e for e in experiments if e in available
            ]
            for exp in exp_names:
                for gold_path in gold_paths:
                    stem = gold_path.parent.name
                    page_dir = stage1_root / exp / stem
                    pred_flat = page_dir / f"{stem}_stage1_flat.txt"
                    pred_tsv = page_dir / f"{stem}_stage1.tsv"
                    if pred_flat.is_file():
                        pred_path = pred_flat
                    elif pred_tsv.is_file():
                        pred_path = pred_tsv
                    else:
                        continue
                    page_id = f"{lang_dir.name}/{stem}"
                    tasks.append(
                        FlatEvalTask(
                            experiment=exp,
                            pred_path=pred_path,
                            gold_path=gold_path,
                            page_id=page_id,
                        )
                    )
        return tasks

    def generate_text_report(
        self, results: List[Stage1Metrics], output_path: Path
    ) -> str:
        return self._report_helper.generate_text_report(results, output_path)

    def generate_json_report(
        self, results: List[Stage1Metrics], output_path: Path
    ) -> None:
        return self._report_helper.generate_json_report(results, output_path)

    def generate_csv_reports(
        self, results: List[Stage1Metrics], output_dir: Path
    ) -> None:
        return self._report_helper.generate_csv_reports(results, output_dir)

    def generate_detailed_csv(
        self,
        results_by_exp: dict,
        samples_dir: Path,
        output_path: Path,
        *,
        stage1_output_subdir: str = "stage-1",
    ) -> None:
        return self._report_helper.generate_detailed_csv(
            results_by_exp,
            samples_dir,
            output_path,
            stage1_output_subdir=stage1_output_subdir,
        )

    def generate_summary_csv(
        self,
        results_by_exp: dict,
        samples_dir: Path,
        output_path: Path,
        *,
        stage1_output_subdir: str = "stage-1",
    ) -> None:
        return self._report_helper.generate_summary_csv(
            results_by_exp,
            samples_dir,
            output_path,
            stage1_output_subdir=stage1_output_subdir,
        )
