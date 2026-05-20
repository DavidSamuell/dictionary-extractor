"""
CLI: evaluate Stage 1 OCR transcription quality.

Usage:
  # Single file
  python -m dictextractor.cli.evaluate_stage1 -p predicted.tsv -g gold.tsv

  # Batch over every experiment under every language
  python -m dictextractor.cli.evaluate_stage1 \
      --samples-dir assets/dictionaries/samples-2 --all-experiments \
      -o assets/dictionaries/samples-2/stage1_eval

  # Batch over a specific subset of experiments
  python -m dictextractor.cli.evaluate_stage1 \
      --samples-dir assets/dictionaries/samples-2 \
      --experiment-name baseline --experiment-name no_alphabet

Batch mode writes ``stage1_eval_cache.json`` under ``-o``: reuse metrics when
prediction/gold files and alignment settings are unchanged. ``--languages``
limits recomputation for that invocation; use ``--overwrite`` to force it.
"""

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from dictextractor.evaluation.stage1.stage1_eval_cache import (
    CACHE_FILE_NAME,
    Stage1EvalCache,
)
from dictextractor.evaluation.stage1.stage1_evaluator import Stage1Evaluator
from dictextractor.evaluation.stage1.stage1_metrics import Stage1Metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 OCR transcription quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python -m dictextractor.cli.evaluate_stage1 \\
      -p page_1_stage1.tsv -g page_1_stage1_GOLD.tsv

  # Compare every experiment under samples-2/
  python -m dictextractor.cli.evaluate_stage1 \\
      --samples-dir assets/dictionaries/samples-2 \\
      --all-experiments \\
      -o assets/dictionaries/samples-2/stage1_eval
        """,
    )

    # Single-file mode
    parser.add_argument("-p", "--predicted", help="Path to predicted Stage 1 TSV")
    parser.add_argument("-g", "--gold", help="Path to gold Stage 1 TSV")

    # Batch mode
    parser.add_argument(
        "--samples-dir",
        help="Root directory containing per-language samples. Gold TSVs are "
        "expected at <lang>/outputs/stage-1-gold/<stem>/<stem>_stage1_GOLD.tsv "
        "and predictions at <lang>/outputs/stage-1/<experiment>/<stem>/"
        "<stem>_stage1.tsv. Run scripts/migrate_stage1_layout.sh once to "
        "move pre-existing gold + predictions into this layout.",
    )
    parser.add_argument(
        "--experiment-name",
        dest="experiment_names",
        action="append",
        default=None,
        help="Restrict batch evaluation to this experiment name "
        "(repeatable). Defaults to all experiments discovered under each "
        "language's outputs/stage-1/.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Optional list of language subfolder names to evaluate when "
        "--samples-dir is used (e.g. --languages Armenian-English "
        "Yiddish-English). Defaults to every subfolder with stage-1 gold.",
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="Explicit form of the default behaviour: evaluate every "
        "experiment subfolder found under each language's outputs/stage-1/. "
        "Mutually exclusive with --experiment-name (which takes precedence).",
    )

    parser.add_argument("-o", "--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--metrics",
        choices=("full", "minimal"),
        default="full",
        help="CSV column set: 'full' (TextEdit, GCER, WER, typography detail, "
        "ReadOrderEdit) or 'minimal' (TextEdit, GCER, WER, typography_f1, "
        "ReadOrderEdit). JSON/text reports always include full detail.",
    )
    parser.add_argument(
        "--alignment-threshold",
        type=float,
        default=0.5,
        help="Minimum semantic span similarity for adjacency matching "
        "(default: 0.5).",
    )
    parser.add_argument(
        "--alignment-max-span-rows",
        type=int,
        default=3,
        help="Maximum adjacent TSV rows per semantic alignment span "
        "(default: 3).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run evaluation for the current (language × experiment) selection "
        "even when cached metrics are still valid. Other languages stay on cache.",
    )

    args = parser.parse_args()

    if not args.predicted and not args.samples_dir:
        parser.error("Provide either -p/-g for single file or --samples-dir for batch")

    evaluator = Stage1Evaluator(
        metrics_profile=args.metrics,
        alignment_threshold=args.alignment_threshold,
        alignment_max_span_rows=args.alignment_max_span_rows,
    )

    # ---- Single-file mode ----
    if args.predicted:
        pred = Path(args.predicted)
        gold = Path(args.gold) if args.gold else None
        if not pred.exists():
            print(f"Error: predicted file not found: {pred}")
            return 1
        if gold and not gold.exists():
            print(f"Error: gold file not found: {gold}")
            return 1
        if not gold:
            parser.error("-g/--gold is required for single-file mode")

        page_id = pred.stem.replace("_stage1", "")
        single_results = [evaluator.evaluate(pred, gold, page_id=page_id)]

        out = (
            Path(args.output_dir) if args.output_dir
            else Path(args.predicted).parent
        )
        text_report = evaluator.generate_text_report(
            single_results, out / "stage1_evaluation_report.txt"
        )
        evaluator.generate_json_report(
            single_results, out / "stage1_evaluation_report.json"
        )
        evaluator.generate_csv_reports(single_results, out)
        print(text_report)
        print(f"\nReports saved to: {out}")
        return 0

    # ---- Batch / experiment mode ----
    samples = Path(args.samples_dir)
    if not samples.is_dir():
        print(f"Error: samples directory not found: {samples}")
        return 1

    if args.languages:
        available = {p.name for p in samples.iterdir() if p.is_dir()}
        missing = set(args.languages) - available
        if missing:
            parser.error(
                f"--languages references unknown subfolders: {sorted(missing)}. "
                f"Available: {sorted(available)}"
            )

    out = (
        Path(args.output_dir) if args.output_dir
        else samples / "stage1_eval"
    )
    out.mkdir(parents=True, exist_ok=True)

    tasks_export = Stage1Evaluator.discover_tasks(
        samples,
        experiments=args.experiment_names,
        languages=None,
    )
    tasks_eval = Stage1Evaluator.discover_tasks(
        samples,
        experiments=args.experiment_names,
        languages=args.languages,
    )
    if not tasks_export:
        print(
            "No matching predicted/gold pairs found. Did you run "
            "scripts/migrate_stage1_layout.sh and run extraction with "
            "--experiment-name first?"
        )
        return 1

    eval_keys = {(t.experiment, t.page_id) for t in tasks_eval}
    cache = Stage1EvalCache(out / CACHE_FILE_NAME)
    cache.load()

    n_cached = 0
    n_evaled = 0
    metrics_by_key: Dict[Tuple[str, str], Stage1Metrics] = {}
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

        print(f"  [eval] {task.experiment} :: {task.page_id}")
        m = evaluator.evaluate(
            task.pred_path, task.gold_path, page_id=task.page_id
        )
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

    paired: List[Tuple[str, Stage1Metrics]] = []
    for task in tasks_export:
        key = (task.experiment, task.page_id)
        if key in metrics_by_key:
            paired.append((task.experiment, metrics_by_key[key]))

    print(
        f"\nStage 1 eval cache: {out / CACHE_FILE_NAME} — "
        f"reused {n_cached}, computed {n_evaled} page(s)."
    )

    # Group by experiment, preserving the order experiments first appear in.
    results_by_exp: "OrderedDict[str, List[Stage1Metrics]]" = OrderedDict()
    for exp, metrics in paired:
        results_by_exp.setdefault(exp, []).append(metrics)

    for exp, results in results_by_exp.items():
        exp_out = out / exp
        text_report = evaluator.generate_text_report(
            results, exp_out / "stage1_evaluation_report.txt"
        )
        evaluator.generate_json_report(
            results, exp_out / "stage1_evaluation_report.json"
        )
        evaluator.generate_csv_reports(results, exp_out)
        print(f"\n### Experiment: {exp} ({len(results)} page(s)) ###")
        print(text_report)
        print(f"Reports saved to: {exp_out}")

    detailed_path = out / "stage1_eval_detailed.csv"
    summary_path = out / "stage1_eval_summary.csv"
    evaluator.generate_detailed_csv(results_by_exp, samples, detailed_path)
    evaluator.generate_summary_csv(results_by_exp, samples, summary_path)

    n_pages = sum(len(r) for r in results_by_exp.values())
    print(
        f"\nDetailed CSV ({n_pages} rows across "
        f"{len(results_by_exp)} experiment(s)): {detailed_path}"
    )
    print(f"Summary CSV (per language): {summary_path}")
    return 0


if __name__ == "__main__":
    exit(main())
