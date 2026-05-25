"""
CLI: evaluate Stage 2 MDF extraction against gold.

Usage:
  uv run dictextractor-eval-stage2-mdf -p pred.mdf.txt -g gold.mdf.txt
  uv run dictextractor-eval-stage2-mdf \\
      --samples-dir assets/dictionaries/samples \\
      --experiment-name gemini31pro_high_mdf_intro_notoolbox \\
      --experiment-name gemini31pro_high_mdf_intro_toolbox \\
      -o evaluations/stage2_mdf_eval
"""

from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from pathlib import Path

from dictextractor.evaluation.stage2.mdf_evaluator import MdfEvaluator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 MDF against gold")
    parser.add_argument("-p", "--predicted", help="Predicted .mdf.txt")
    parser.add_argument("-g", "--gold", help="Gold .mdf.txt")
    parser.add_argument("--samples-dir", help="Samples root for batch mode")
    parser.add_argument(
        "--experiment-name",
        dest="experiment_names",
        action="append",
        default=None,
    )
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--all-experiments", action="store_true")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument(
        "--record-threshold",
        type=float,
        default=0.6,
        help="Record fingerprint similarity threshold (default: 0.6)",
    )
    parser.add_argument(
        "--line-threshold",
        type=float,
        default=0.7,
        help="Line value similarity threshold (default: 0.7)",
    )
    parser.add_argument(
        "--marker-sub-list",
        dest="marker_sub_list",
        default=None,
        help="Optional path to mdf_marker_sub_list.yaml",
    )
    args = parser.parse_args()

    evaluator = MdfEvaluator(
        record_threshold=args.record_threshold,
        line_threshold=args.line_threshold,
        marker_sub_list_path=args.marker_sub_list,
    )

    if args.predicted:
        pred = Path(args.predicted)
        gold = Path(args.gold) if args.gold else None
        if not pred.is_file() or gold is None or not gold.is_file():
            logger.error("Single-file mode requires existing -p and -g paths")
            return 1
        page_id = pred.parent.name
        results = [evaluator.evaluate(pred, gold, page_id=page_id)]
        out = Path(args.output_dir) if args.output_dir else pred.parent
        text = evaluator.generate_text_report(results, out / "stage2_mdf_evaluation_report.txt")
        evaluator.generate_json_report(results, out / "stage2_mdf_evaluation_report.json")
        print(text)
        print(f"\nReports saved to: {out}")
        return 0

    if not args.samples_dir:
        parser.error("Provide -p/-g or --samples-dir")
    samples = Path(args.samples_dir)
    if not samples.is_dir():
        logger.error("Samples directory not found: %s", samples)
        return 1

    experiments = None if args.all_experiments else args.experiment_names
    tasks = evaluator.discover_tasks(samples, experiments=experiments, languages=args.languages)
    if not tasks:
        logger.error("No stage-2 gold/pred MDF pairs found")
        return 1

    out = Path(args.output_dir) if args.output_dir else samples / "stage2_mdf_eval"
    out.mkdir(parents=True, exist_ok=True)

    results_by_exp: OrderedDict[str, list] = OrderedDict()
    for task in tasks:
        logger.info("  [eval] %s :: %s", task.experiment, task.page_id)
        metrics = evaluator.evaluate(task.pred_path, task.gold_path, page_id=task.page_id)
        results_by_exp.setdefault(task.experiment, []).append(metrics)

    for exp, pages in results_by_exp.items():
        exp_out = out / exp
        text = evaluator.generate_text_report(pages, exp_out / "stage2_mdf_evaluation_report.txt")
        evaluator.generate_json_report(pages, exp_out / "stage2_mdf_evaluation_report.json")
        print(f"\n### Experiment: {exp} ({len(pages)} page(s)) ###")
        print(text)

    summary_csv = out / "stage2_mdf_eval_summary.csv"
    evaluator.generate_summary_csv(results_by_exp, summary_csv)
    print(f"\nSummary CSV: {summary_csv}")
    print(f"Reports under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
