"""
CLI: evaluate Stage 1 flat transcription (eval-flat).

Usage:
  uv run dictextractor-eval-flat \\
      --samples-dir assets/dictionaries/samples \\
      --all-experiments -o evaluations/stage1_flat_eval
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

from dictextractor.evaluation.stage1.flat_evaluator import FlatEvalTask, FlatStage1Evaluator
from dictextractor.evaluation.stage1.stage1_eval_cache import (
    Stage1EvalCache,
)
from dictextractor.evaluation.stage1.stage1_metrics import Stage1Metrics

FLAT_CACHE_FILE_NAME = "stage1_flat_eval_cache.json"

# Specialized OCR backends (folder names without "flat"; preds use *_stage1_flat.txt).
DEFAULT_VLM_OCR_EXPERIMENTS: tuple[str, ...] = (
    "MinerU2.5-Pro",
    "PaddleOCR-VL-1.5",
    "GLM-OCR",
)


def list_stage1_experiments(
    samples_dir: Path,
    *,
    languages: list[str] | None,
    name_contains: str | None,
) -> list[str]:
    """Return sorted experiment folder names under ``outputs/stage-1``."""
    needle = name_contains.lower() if name_contains else None
    names: set[str] = set()
    for lang_dir in sorted(samples_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        if languages and lang_dir.name not in languages:
            continue
        stage1_root = lang_dir / "outputs" / "stage-1"
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
) -> list[str]:
    """LLM flat ablations plus specialized OCR folders (excludes column Gemini)."""
    flat = list_stage1_experiments(
        samples, languages=languages, name_contains="flat"
    )
    on_disk = set(
        list_stage1_experiments(samples, languages=languages, name_contains=None)
    )
    ocr = [name for name in DEFAULT_VLM_OCR_EXPERIMENTS if name in on_disk]
    return sorted(set(flat) | set(ocr))


def resolve_experiment_names(args: argparse.Namespace, samples: Path) -> list[str] | None:
    """Resolve which experiment folders to include in batch eval-flat."""
    if args.include_vlm_ocr:
        found = resolve_flat_and_vlm_ocr_experiments(
            samples, languages=args.languages
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
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument(
        "--metrics",
        choices=("full", "minimal"),
        default="minimal",
    )
    parser.add_argument("--alignment-threshold", type=float, default=0.5)
    parser.add_argument("--alignment-max-span-rows", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    evaluator = FlatStage1Evaluator(
        metrics_profile=args.metrics,
        alignment_threshold=args.alignment_threshold,
        alignment_max_span_rows=args.alignment_max_span_rows,
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
        samples, experiments=experiment_names, languages=None
    )
    tasks_eval = evaluator.discover_tasks(
        samples,
        experiments=experiment_names,
        languages=args.languages,
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
    amax = evaluator.alignment_max_span_rows

    for task in tasks_export:
        key = (task.experiment, task.page_id)
        in_eval = key in eval_keys
        cache_ok = cache.entry_valid(
            task.experiment,
            task.page_id,
            task.pred_path,
            task.gold_path,
            ath,
            amax,
        )
        if cache_ok and not (in_eval and args.overwrite):
            entry = cache.get_entry(task.experiment, task.page_id)
            if entry is not None:
                metrics_by_key[key] = entry.metrics
                n_cached += 1
                continue
        print(f"  [eval-flat] {task.experiment} :: {task.page_id}")
        m = evaluator.evaluate(task.pred_path, task.gold_path, page_id=task.page_id)
        cache.put(
            task.experiment,
            task.page_id,
            task.pred_path,
            task.gold_path,
            ath,
            amax,
            m,
        )
        metrics_by_key[key] = m
        n_evaled += 1

    cache.prune_stale_paths(samples)
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
        samples, experiments=None, languages=args.languages
    )
    results_for_csv = cache.collect_valid_metrics(
        tasks_for_csv,
        alignment_threshold=ath,
        alignment_max_span_rows=amax,
    )
    n_csv_exps = len(results_for_csv)
    n_csv_pages = sum(len(v) for v in results_for_csv.values())
    print(
        f"Aggregate CSVs: {n_csv_pages} page(s) across {n_csv_exps} experiment(s) "
        f"(cached + this run; not limited to experiments selected above)."
    )
    evaluator.generate_detailed_csv(
        results_for_csv, samples, out / "stage1_flat_eval_detailed.csv"
    )
    evaluator.generate_summary_csv(
        results_for_csv, samples, out / "stage1_flat_eval_summary.csv"
    )
    print(f"\nReports under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
