"""
CLI: evaluate extraction results against ground truth.
Usage: python -m dictextractor.cli.evaluate [options]
"""

import argparse
from pathlib import Path

from dictextractor.evaluation.stage2.evaluator import DictionaryEvaluator
from dictextractor.evaluation.stage2.error_analyzer import DetailedErrorAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dictionary extraction results against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m dictextractor.cli.evaluate -e extracted.tsv -g gold_label.tsv
  python -m dictextractor.cli.evaluate -e extracted.tsv -g gold_label.tsv -o results/ -t 0.90
  python -m dictextractor.cli.evaluate -e extracted.tsv -g gold_label.tsv --no-char-analysis
        """,
    )
    parser.add_argument("-e", "--extracted", required=True, help="Path to extracted TSV file")
    parser.add_argument("-g", "--ground-truth", required=True, help="Path to ground truth TSV file")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory for reports")
    parser.add_argument("-t", "--threshold", type=float, default=0.85, help="Similarity threshold (default: 0.85)")
    parser.add_argument("--no-char-analysis", action="store_true", help="Skip character-level analysis")

    args = parser.parse_args()

    extracted_path = Path(args.extracted)
    if not extracted_path.exists():
        print(f"Error: Extracted TSV not found: {extracted_path}")
        return 1

    ground_truth_path = Path(args.ground_truth)
    if not ground_truth_path.exists():
        print(f"Error: Ground truth TSV not found: {ground_truth_path}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else extracted_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating: {extracted_path}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Threshold: {args.threshold}")
    print("-" * 60)

    evaluator = DictionaryEvaluator(similarity_threshold=args.threshold)
    metrics = evaluator.evaluate(str(ground_truth_path), str(extracted_path))
    evaluator.generate_report(metrics, str(output_dir / "evaluation_report.txt"))

    if not args.no_char_analysis:
        print("\n" + "=" * 60)
        print("Running character-level error analysis...")
        print("=" * 60)
        matched_pairs = [(m["ground_truth"], m["extracted"]) for m in evaluator.matched_entries]
        if matched_pairs:
            gt_entries = evaluator.load_tsv(str(ground_truth_path))
            ext_entries = evaluator.load_tsv(str(extracted_path))
            analyzer = DetailedErrorAnalyzer()
            error_stats = analyzer.calculate_cer_wer(gt_entries, ext_entries, matched_pairs)
            analyzer.generate_detailed_report(error_stats, str(output_dir / "character_error_report.txt"))
        else:
            print("No matched entries — skipping character-level analysis.")
    else:
        print("Skipping character-level analysis (--no-char-analysis)")

    return 0


if __name__ == "__main__":
    exit(main())
