"""
DictionaryEvaluator: entry-level evaluation against ground truth.
Migrated from src/evaluate_extraction.py.
"""

import csv
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dictextractor.evaluation.stage2.metrics import EvaluationMetrics
from dictextractor.utils.text import normalize_text


class DictionaryEvaluator:
    """
    Evaluates extraction quality by matching extracted entries against ground truth
    using weighted string similarity, then computing precision, recall, and F1.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.matched_entries: List[Dict] = []
        self.missing_entries: List[Dict] = []
        self.extra_entries: List[Dict] = []

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_tsv(self, filepath: str) -> List[Dict[str, str]]:
        entries = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                entries.append(dict(row))
        return entries

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0 if (a or b) else 1.0
        return SequenceMatcher(None, a, b).ratio()

    def find_matching_entry(
        self,
        entry: Dict[str, str],
        candidates: List[Dict[str, str]],
        used: Set[int],
    ) -> Tuple[int, float]:
        best_idx, best_score = -1, 0.0
        for idx, candidate in enumerate(candidates):
            if idx in used:
                continue
            hw_sim = self._similarity(
                normalize_text(entry.get("Headword_Phrase", "")),
                normalize_text(candidate.get("Headword_Phrase", "")),
            )
            if hw_sim < 0.7:
                continue
            gram_sim = self._similarity(
                normalize_text(entry.get("POS", "")),
                normalize_text(candidate.get("POS", "")),
            )
            def_sim = self._similarity(
                normalize_text(entry.get("Translation_RU", "")),
                normalize_text(candidate.get("Translation_RU", "")),
            )
            score = hw_sim * 0.5 + gram_sim * 0.15 + def_sim * 0.35
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx, best_score

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, ground_truth_path: str, extracted_path: str) -> EvaluationMetrics:
        ground_truth = self.load_tsv(ground_truth_path)
        extracted = self.load_tsv(extracted_path)

        used: Set[int] = set()
        exact_matches = partial_matches = 0
        hw_scores, gram_scores, def_scores = [], [], []
        self.matched_entries, self.missing_entries = [], []

        for gt_entry in ground_truth:
            idx, score = self.find_matching_entry(gt_entry, extracted, used)
            if idx >= 0:
                used.add(idx)
                ext = extracted[idx]
                hw = self._similarity(
                    normalize_text(gt_entry.get("Headword_Phrase", "")),
                    normalize_text(ext.get("Headword_Phrase", "")),
                )
                gr = self._similarity(
                    normalize_text(gt_entry.get("POS", "")),
                    normalize_text(ext.get("POS", "")),
                )
                df = self._similarity(
                    normalize_text(gt_entry.get("Translation_RU", "")),
                    normalize_text(ext.get("Translation_RU", "")),
                )
                hw_scores.append(hw)
                gram_scores.append(gr)
                def_scores.append(df)

                if hw == 1.0 and gr >= 0.9 and df >= 0.9:
                    exact_matches += 1
                elif score >= self.similarity_threshold:
                    partial_matches += 1

                self.matched_entries.append(
                    {"ground_truth": gt_entry, "extracted": ext, "scores": {"overall": score, "headword": hw, "grammatical": gr, "definition": df}}
                )
            else:
                self.missing_entries.append(gt_entry)

        self.extra_entries = [extracted[i] for i in range(len(extracted)) if i not in used]

        total_matched = exact_matches + partial_matches
        precision = total_matched / len(extracted) if extracted else 0
        recall = total_matched / len(ground_truth) if ground_truth else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return EvaluationMetrics(
            total_ground_truth=len(ground_truth),
            total_extracted=len(extracted),
            exact_matches=exact_matches,
            partial_matches=partial_matches,
            missing_entries=len(self.missing_entries),
            extra_entries=len(self.extra_entries),
            headword_accuracy=sum(hw_scores) / len(hw_scores) if hw_scores else 0,
            grammatical_accuracy=sum(gram_scores) / len(gram_scores) if gram_scores else 0,
            definition_accuracy=sum(def_scores) / len(def_scores) if def_scores else 0,
            overall_f1=f1,
            precision=precision,
            recall=recall,
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, metrics: EvaluationMetrics, output_path: str = "evaluation_report.txt"):
        lines = [
            "=" * 60,
            "DICTIONARY EXTRACTION EVALUATION REPORT",
            "=" * 60,
            "",
            "SUMMARY STATISTICS:",
            "-" * 40,
            f"Total entries in ground truth: {metrics.total_ground_truth}",
            f"Total entries extracted:        {metrics.total_extracted}",
            f"Exact matches:                  {metrics.exact_matches}",
            f"Partial matches:                {metrics.partial_matches}",
            f"Missing entries:                {metrics.missing_entries}",
            f"Extra entries:                  {metrics.extra_entries}",
            "",
            "PERFORMANCE METRICS:",
            "-" * 40,
            f"Overall F1 Score: {metrics.overall_f1:.3f}",
            f"Precision:        {metrics.precision:.3f}",
            f"Recall:           {metrics.recall:.3f}",
            "",
            "FIELD-LEVEL ACCURACY:",
            "-" * 40,
            f"Headword accuracy:         {metrics.headword_accuracy:.3f}",
            f"Grammatical info accuracy: {metrics.grammatical_accuracy:.3f}",
            f"Definition accuracy:       {metrics.definition_accuracy:.3f}",
            "",
        ]

        if self.missing_entries:
            lines += ["MISSING ENTRIES (First 10):", "-" * 40]
            for i, e in enumerate(self.missing_entries[:10]):
                lines.append(f"{i+1}. {e.get('Headword_Phrase', 'N/A')} ({e.get('POS', '')})")
            if len(self.missing_entries) > 10:
                lines.append(f"... and {len(self.missing_entries) - 10} more")
            lines.append("")

        if self.extra_entries:
            lines += ["EXTRA ENTRIES (First 10):", "-" * 40]
            for i, e in enumerate(self.extra_entries[:10]):
                lines.append(f"{i+1}. {e.get('Headword_Phrase', 'N/A')} ({e.get('POS', '')})")
            if len(self.extra_entries) > 10:
                lines.append(f"... and {len(self.extra_entries) - 10} more")
            lines.append("")

        low_score = [m for m in self.matched_entries if m["scores"]["overall"] < 0.9]
        if low_score:
            lines += ["LOW-SCORING MATCHES (First 5):", "-" * 40]
            for i, m in enumerate(low_score[:5]):
                lines.append(f"{i+1}. Ground Truth: {m['ground_truth'].get('Headword_Phrase', 'N/A')}")
                lines.append(f"   Extracted:    {m['extracted'].get('Headword_Phrase', 'N/A')}")
                lines.append(f"   Score: {m['scores']['overall']:.3f}")
                lines.append("")

        report_text = "\n".join(lines)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(report_text)
        print(f"\nReport saved to: {output_path}")

        json_out = output_path.replace(".txt", "_details.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": {
                        "total_ground_truth": metrics.total_ground_truth,
                        "total_extracted": metrics.total_extracted,
                        "exact_matches": metrics.exact_matches,
                        "partial_matches": metrics.partial_matches,
                        "missing_entries": metrics.missing_entries,
                        "extra_entries": metrics.extra_entries,
                        "headword_accuracy": metrics.headword_accuracy,
                        "grammatical_accuracy": metrics.grammatical_accuracy,
                        "definition_accuracy": metrics.definition_accuracy,
                        "overall_f1": metrics.overall_f1,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                    },
                    "missing_entries": self.missing_entries,
                    "extra_entries": self.extra_entries,
                    "low_score_matches": low_score[:10],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Detailed results saved to: {json_out}")
