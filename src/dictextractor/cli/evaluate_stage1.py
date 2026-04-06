"""
CLI: evaluate Stage 1 OCR transcription quality.

Usage:
  # Single file
  python -m dictextractor.cli.evaluate_stage1 -p predicted.tsv -g gold.tsv

  # Batch (auto-discovers *_stage1.tsv / *_stage1_GOLD.tsv pairs)
  python -m dictextractor.cli.evaluate_stage1 --samples-dir assets/dictionaries/samples/
"""

import argparse
from pathlib import Path

from dictextractor.evaluation.stage1.stage1_evaluator import Stage1Evaluator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 OCR transcription quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m dictextractor.cli.evaluate_stage1 \\
      -p page_1_stage1.tsv -g page_1_stage1_GOLD.tsv

  python -m dictextractor.cli.evaluate_stage1 \\
      --samples-dir assets/dictionaries/samples/ \\
      -o results/stage1_eval/
        """,
    )

    # Single-file mode
    parser.add_argument("-p", "--predicted", help="Path to predicted Stage 1 TSV")
    parser.add_argument("-g", "--gold", help="Path to gold Stage 1 TSV")

    # Batch mode
    parser.add_argument(
        "--samples-dir",
        help="Root directory to scan for *_stage1.tsv / *_stage1_GOLD.tsv pairs",
    )

    parser.add_argument("-o", "--output-dir", default=None, help="Output directory")

    args = parser.parse_args()

    if not args.predicted and not args.samples_dir:
        parser.error("Provide either -p/-g for single file or --samples-dir for batch")

    evaluator = Stage1Evaluator()

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
        results = [evaluator.evaluate(pred, gold, page_id=page_id)]

    # ---- Batch mode ----
    else:
        samples = Path(args.samples_dir)
        if not samples.is_dir():
            print(f"Error: samples directory not found: {samples}")
            return 1
        results = evaluator.evaluate_batch(samples)
        if not results:
            print("No matching predicted/gold TSV pairs found.")
            return 1

    # ---- Output ----
    if args.output_dir:
        out = Path(args.output_dir)
    elif args.predicted:
        out = Path(args.predicted).parent
    else:
        out = Path(args.samples_dir) / "stage1_eval"

    text_report = evaluator.generate_text_report(
        results, out / "stage1_evaluation_report.txt"
    )
    evaluator.generate_json_report(results, out / "stage1_evaluation_report.json")
    evaluator.generate_csv_reports(results, out)

    print(text_report)
    print(f"\nReports saved to: {out}")
    return 0


if __name__ == "__main__":
    exit(main())
