"""
CLI: evaluate Stage 1 flat transcription (eval-flat).

Usage:
  uv run dictextractor-eval-flat \\
      --samples-dir assets/dictionaries/samples \\
      --all-experiments -o evaluations/stage1_flat_eval
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Literal, Tuple

from dictextractor.evaluation.stage1.flat_evaluator import FlatEvalTask, FlatStage1Evaluator
from dictextractor.evaluation.stage1.stage1_eval_cache import (
    Stage1EvalCache,
)
from dictextractor.evaluation.stage1.stage1_metrics import Stage1Metrics
from dictextractor.evaluation.stage1.stage1_reports import Stage1ReportWriter

FLAT_CACHE_FILE_NAME = "stage1_flat_eval_cache.json"
FLAT_OCR_HINT_SUMMARY_FILE_NAME = "stage1_flat_eval_ocr_hint_summary.csv"
FLAT_OCR_HINT_DETAILED_FILE_NAME = "stage1_flat_eval_ocr_hint_detailed.csv"
GEMINI31PRO_FLAT_ALPHA_OCR_SUMMARY_FILE_NAME = (
    "gemini31pro_flat_alpha_ocr_summary.csv"
)

MetricsProfile = Literal["full", "minimal"]
CharacterAlignmentMode = Literal["collapsed", "quick_match"]


def _parallel_eval_worker(
    payload: tuple[
        str,
        str,
        str,
        str,
        str,
        MetricsProfile,
        CharacterAlignmentMode,
        float,
    ],
) -> tuple[str, str, Path, Path, Stage1Metrics]:
    """Evaluate one flat page (picklable entry point for ProcessPoolExecutor)."""
    experiment, pred_s, gold_s, page_id, metrics_profile, character_alignment, ath = (
        payload
    )
    evaluator = FlatStage1Evaluator(
        metrics_profile=metrics_profile,
        character_alignment=character_alignment,
        alignment_threshold=ath,
    )
    pred_path = Path(pred_s)
    gold_path = Path(gold_s)
    metrics = evaluator.evaluate(pred_path, gold_path, page_id=page_id)
    return experiment, page_id, pred_path, gold_path, metrics

# Specialized OCR backends (folder names without "flat"; preds use *_stage1_flat.txt).
DEFAULT_VLM_OCR_EXPERIMENTS: tuple[str, ...] = (
    "MinerU2.5-Pro",
    "PaddleOCR-VL-1.5",
    "GLM-OCR",
    "Mathpix-OCR",
)


def list_stage1_experiments(
    samples_dir: Path,
    *,
    languages: list[str] | None,
    name_contains: str | None,
    stage1_output_subdir: str = "stage-1",
) -> list[str]:
    """Return sorted experiment folder names under ``outputs/<stage1_output_subdir>``."""
    needle = name_contains.lower() if name_contains else None
    names: set[str] = set()
    for lang_dir in sorted(samples_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        if languages and lang_dir.name not in languages:
            continue
        stage1_root = lang_dir / "outputs" / stage1_output_subdir
        if not stage1_root.is_dir():
            continue
        for child in stage1_root.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                if needle is None or needle in child.name.lower():
                    names.add(child.name)
    return sorted(names)


def resolve_flat_and_vlm_ocr_experiments(
    samples: Path,
    *,
    languages: list[str] | None,
    stage1_output_subdir: str = "stage-1",
) -> list[str]:
    """LLM flat ablations plus specialized OCR folders (excludes column Gemini)."""
    flat = list_stage1_experiments(
        samples,
        languages=languages,
        name_contains="flat",
        stage1_output_subdir=stage1_output_subdir,
    )
    on_disk = set(
        list_stage1_experiments(
            samples,
            languages=languages,
            name_contains=None,
            stage1_output_subdir=stage1_output_subdir,
        )
    )
    ocr = [name for name in DEFAULT_VLM_OCR_EXPERIMENTS if name in on_disk]
    return sorted(set(flat) | set(ocr))


def split_results_by_ocr_hint(
    results_by_exp: "OrderedDict[str, list[Stage1Metrics]]",
    samples_dir: Path,
    *,
    stage1_output_subdir: str = "stage-1",
) -> tuple[OrderedDict[str, list[Stage1Metrics]], OrderedDict[str, list[Stage1Metrics]]]:
    """Partition cached results into non-OCR-hint vs OCR-hint LLM experiments."""
    without_hint: OrderedDict[str, list[Stage1Metrics]] = OrderedDict()
    with_hint: OrderedDict[str, list[Stage1Metrics]] = OrderedDict()
    for experiment, metrics in results_by_exp.items():
        languages: set[str] = set()
        for metric in metrics:
            parsed = Stage1ReportWriter._parse_page_id(metric.page_id)
            if parsed is not None:
                languages.add(parsed[0])
        uses_ocr_hint = False
        for language in languages:
            _, ocr_hint = Stage1ReportWriter._load_run_config_flags(
                samples_dir,
                language,
                experiment,
                stage1_output_subdir=stage1_output_subdir,
            )
            if ocr_hint:
                uses_ocr_hint = True
                break
        target = with_hint if uses_ocr_hint else without_hint
        target[experiment] = metrics
    return without_hint, with_hint


def filter_results_by_experiment(
    results_by_exp: "OrderedDict[str, list[Stage1Metrics]]",
    experiment_name: str,
) -> OrderedDict[str, list[Stage1Metrics]]:
    """Return a single-experiment slice (empty when absent)."""
    metrics = results_by_exp.get(experiment_name)
    if not metrics:
        return OrderedDict()
    return OrderedDict([(experiment_name, metrics)])


def resolve_experiment_names(
    args: argparse.Namespace, samples: Path
) -> list[str] | None:
    """Resolve which experiment folders to include in batch eval-flat."""
    subdir = getattr(args, "stage1_output_subdir", "stage-1")
    if args.include_vlm_ocr:
        found = resolve_flat_and_vlm_ocr_experiments(
            samples, languages=args.languages, stage1_output_subdir=subdir
        )
        if args.experiment_names:
            allowed = set(args.experiment_names)
            found = [name for name in found if name in allowed]
        if not found:
            print("No flat or VLM OCR experiments found on disk.")
        else:
            print(f"Experiments (flat + VLM OCR): {found}")
        return found
    if args.experiment_name_contains:
        found = list_stage1_experiments(
            samples,
            languages=args.languages,
            name_contains=args.experiment_name_contains,
            stage1_output_subdir=subdir,
        )
        if args.experiment_names:
            allowed = set(args.experiment_names)
            found = [name for name in found if name in allowed]
        if not found:
            print(
                f"No experiments matching name filter {args.experiment_name_contains!r}."
            )
        else:
            print(f"Experiments (name contains {args.experiment_name_contains!r}): {found}")
        return found
    if args.all_experiments:
        return None
    return args.experiment_names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 flat OCR transcription (eval-flat)",
    )
    parser.add_argument("-p", "--predicted", help="Predicted flat .txt or column .tsv")
    parser.add_argument("-g", "--gold", help="Gold flat .txt")
    parser.add_argument("--samples-dir", help="Samples root for batch mode")
    parser.add_argument(
        "--experiment-name",
        dest="experiment_names",
        action="append",
        default=None,
    )
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--all-experiments", action="store_true")
    parser.add_argument(
        "--experiment-name-contains",
        default=None,
        metavar="SUBSTR",
        help=(
            "Only evaluate experiment folders whose name contains SUBSTR "
            "(case-insensitive), e.g. 'flat' for LLM flat ablations."
        ),
    )
    parser.add_argument(
        "--include-vlm-ocr",
        action="store_true",
        help=(
            "Evaluate gemini*flat* experiments plus MinerU / Paddle / GLM OCR. "
            "Does not include column-mode gemini3flash_* or legacy."
        ),
    )
    parser.add_argument(
        "--stage1-output-subdir",
        default="stage-1",
        dest="stage1_output_subdir",
        help="Prediction root under outputs/ (default: stage-1). Use stage-1-ocr for "
        "per-language best OCR-hint runs.",
    )
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument(
        "--metrics",
        choices=("full", "minimal"),
        default="minimal",
    )
    parser.add_argument(
        "--alignment-threshold",
        type=float,
        default=0.6,
        help="Deprecated; quick_match thresholds are fixed (OmniDocBench defaults).",
    )
    parser.add_argument(
        "--character-alignment",
        choices=("collapsed", "quick_match"),
        default="quick_match",
        help=(
            "How to score character/typography metrics: "
            "'quick_match' (default; OmniDocBench line-level Adjacency Search Match) or "
            "'collapsed' (join whole page into one string per side)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Parallel worker processes for page evaluation (default: 1). "
            "Use cpu_count-2 on batch nodes (e.g. 14 on a 16-core allocation)."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    evaluator = FlatStage1Evaluator(
        metrics_profile=args.metrics,
        character_alignment=args.character_alignment,
        alignment_threshold=args.alignment_threshold,
    )

    if args.predicted:
        pred = Path(args.predicted)
        gold = Path(args.gold) if args.gold else None
        if not pred.exists() or not gold or not gold.exists():
            print("Error: -p and -g must exist for single-file mode")
            return 1
        page_id = pred.stem.replace("_stage1_flat", "").replace("_stage1", "")
        results = [evaluator.evaluate(pred, gold, page_id=page_id)]
        out = Path(args.output_dir) if args.output_dir else pred.parent
        report_path = out / "stage1_flat_evaluation_report.txt"
        text = evaluator.generate_text_report(results, report_path)
        evaluator.generate_json_report(results, out / "stage1_flat_evaluation_report.json")
        evaluator.generate_csv_reports(results, out)
        print(text)
        print(f"\nReports saved to: {out}")
        return 0

    if not args.samples_dir:
        parser.error("Provide -p/-g or --samples-dir")
    samples = Path(args.samples_dir)
    if not samples.is_dir():
        print(f"Error: samples directory not found: {samples}")
        return 1

    mode_flags = sum(
        bool(x)
        for x in (
            args.all_experiments,
            args.experiment_name_contains,
            args.include_vlm_ocr,
        )
    )
    if mode_flags > 1:
        parser.error(
            "Use only one of --all-experiments, --experiment-name-contains, "
            "--include-vlm-ocr."
        )

    out = Path(args.output_dir) if args.output_dir else samples / "stage1_flat_eval"
    out.mkdir(parents=True, exist_ok=True)

    experiment_names = resolve_experiment_names(args, samples)
    if experiment_names is not None and not experiment_names:
        return 1

    tasks_export = evaluator.discover_tasks(
        samples,
        experiments=experiment_names,
        languages=None,
        stage1_output_subdir=args.stage1_output_subdir,
    )
    tasks_eval = evaluator.discover_tasks(
        samples,
        experiments=experiment_names,
        languages=args.languages,
        stage1_output_subdir=args.stage1_output_subdir,
    )
    if not tasks_export:
        print("No flat gold/pred pairs found.")
        return 1

    eval_keys = {(t.experiment, t.page_id) for t in tasks_eval}
    cache = Stage1EvalCache(out / FLAT_CACHE_FILE_NAME)
    cache.load()
    metrics_by_key: Dict[Tuple[str, str], Stage1Metrics] = {}
    n_cached = n_evaled = 0
    ath = evaluator.alignment_threshold
    calign = evaluator.character_alignment

    to_eval: list[FlatEvalTask] = []
    for task in tasks_export:
        key = (task.experiment, task.page_id)
        in_eval = key in eval_keys
        cache_ok = cache.entry_valid(
            task.experiment,
            task.page_id,
            task.pred_path,
            task.gold_path,
            ath,
            calign,
        )
        if cache_ok and not (in_eval and args.overwrite):
            entry = cache.get_entry(task.experiment, task.page_id)
            if entry is not None:
                metrics_by_key[key] = entry.metrics
                n_cached += 1
                continue
        to_eval.append(task)

    def _store_eval_result(
        experiment: str,
        page_id: str,
        pred_path: Path,
        gold_path: Path,
        metrics: Stage1Metrics,
    ) -> None:
        nonlocal n_evaled
        print(f"  [eval-flat] {experiment} :: {page_id}")
        cache.put(experiment, page_id, pred_path, gold_path, ath, calign, metrics)
        metrics_by_key[(experiment, page_id)] = metrics
        n_evaled += 1

    if to_eval and args.workers > 1:
        worker_payloads = [
            (
                task.experiment,
                str(task.pred_path),
                str(task.gold_path),
                task.page_id,
                args.metrics,
                args.character_alignment,
                ath,
            )
            for task in to_eval
        ]
        print(
            f"Evaluating {len(to_eval)} page(s) with {args.workers} worker(s) "
            f"(cpus on node: {os.cpu_count()})"
        )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(_parallel_eval_worker, worker_payloads):
                _store_eval_result(*result)
    else:
        for task in to_eval:
            m = evaluator.evaluate(
                task.pred_path, task.gold_path, page_id=task.page_id
            )
            _store_eval_result(
                task.experiment,
                task.page_id,
                task.pred_path,
                task.gold_path,
                m,
            )

    cache.prune_stale_paths(samples, stage1_output_subdir=args.stage1_output_subdir)
    cache.save()

    paired = []
    for task in tasks_export:
        key = (task.experiment, task.page_id)
        if key in metrics_by_key:
            paired.append((task.experiment, metrics_by_key[key]))

    print(
        f"\nFlat eval cache: {out / FLAT_CACHE_FILE_NAME} — "
        f"reused {n_cached}, computed {n_evaled} page(s)."
    )

    results_by_exp: OrderedDict[str, list[Stage1Metrics]] = OrderedDict()
    for exp, metrics in paired:
        results_by_exp.setdefault(exp, []).append(metrics)

    for exp, results in results_by_exp.items():
        exp_out = out / exp
        text = evaluator.generate_text_report(
            results, exp_out / "stage1_flat_evaluation_report.txt"
        )
        evaluator.generate_json_report(
            results, exp_out / "stage1_flat_evaluation_report.json"
        )
        evaluator.generate_csv_reports(results, exp_out)
        print(f"\n### Experiment: {exp} ({len(results)} page(s)) ###")
        print(text)

    tasks_for_csv = evaluator.discover_tasks(
        samples,
        experiments=None,
        languages=args.languages,
        stage1_output_subdir=args.stage1_output_subdir,
    )
    results_for_csv: OrderedDict[str, list[Stage1Metrics]] = cache.collect_valid_metrics(
        tasks_for_csv,
        alignment_threshold=ath,
        character_alignment=calign,
    )
    main_results, ocr_hint_results = split_results_by_ocr_hint(
        results_for_csv,
        samples,
        stage1_output_subdir=args.stage1_output_subdir,
    )
    split_ocr_hint = args.stage1_output_subdir == "stage-1"
    if split_ocr_hint:
        csv_results = main_results
        n_csv_exps = len(main_results)
        n_csv_pages = sum(len(v) for v in main_results.values())
        print(
            f"Aggregate CSVs: {n_csv_pages} page(s) across {n_csv_exps} experiment(s) "
            f"in main summary (OCR-hint LLM runs excluded)."
        )
        if ocr_hint_results:
            print(
                f"OCR-hint sidecar: {sum(len(v) for v in ocr_hint_results.values())} "
                f"page(s) across {len(ocr_hint_results)} experiment(s) → "
                f"{FLAT_OCR_HINT_SUMMARY_FILE_NAME}"
            )
    else:
        csv_results = results_for_csv
        n_csv_exps = len(results_for_csv)
        n_csv_pages = sum(len(v) for v in results_for_csv.values())
        print(
            f"Aggregate CSVs: {n_csv_pages} page(s) across {n_csv_exps} experiment(s)."
        )
    evaluator.generate_detailed_csv(
        csv_results,
        samples,
        out / "stage1_flat_eval_detailed.csv",
        stage1_output_subdir=args.stage1_output_subdir,
    )
    evaluator.generate_summary_csv(
        csv_results,
        samples,
        out / "stage1_flat_eval_summary.csv",
        stage1_output_subdir=args.stage1_output_subdir,
    )
    if split_ocr_hint and ocr_hint_results:
        evaluator.generate_detailed_csv(
            ocr_hint_results,
            samples,
            out / FLAT_OCR_HINT_DETAILED_FILE_NAME,
            stage1_output_subdir=args.stage1_output_subdir,
        )
        evaluator.generate_summary_csv(
            ocr_hint_results,
            samples,
            out / FLAT_OCR_HINT_SUMMARY_FILE_NAME,
            stage1_output_subdir=args.stage1_output_subdir,
        )
        gemini31pro_results = filter_results_by_experiment(
            ocr_hint_results, "gemini31pro_flat_alpha_ocr"
        )
        if gemini31pro_results:
            evaluator.generate_summary_csv(
                gemini31pro_results,
                samples,
                out / GEMINI31PRO_FLAT_ALPHA_OCR_SUMMARY_FILE_NAME,
                stage1_output_subdir=args.stage1_output_subdir,
            )
    print(f"\nReports under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
