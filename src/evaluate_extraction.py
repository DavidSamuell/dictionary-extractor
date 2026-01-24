"""
Evaluation script for Chukchi-Russian dictionary extraction
Compares extracted TSV results with ground truth
"""

import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import pandas as pd
from collections import defaultdict, Counter

try:
    import Levenshtein

    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print(
        "Warning: python-Levenshtein not installed. Install it with: pip install python-Levenshtein"
    )
    print("Falling back to basic edit distance calculation.")
    import difflib


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""

    total_ground_truth: int
    total_extracted: int
    exact_matches: int
    partial_matches: int
    missing_entries: int
    extra_entries: int
    headword_accuracy: float
    grammatical_accuracy: float
    definition_accuracy: float
    overall_f1: float
    precision: float
    recall: float


class DetailedErrorAnalyzer:
    """Analyzes character-level errors with detailed statistics"""

    def __init__(self):
        """Initialize the detailed error analyzer"""
        self.has_levenshtein = HAS_LEVENSHTEIN

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison (lowercase, strip whitespace, etc.)
        Also normalizes homoglyphs (visually identical characters from different scripts)

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if text is None:
            return ""

        # Homoglyph mapping: normalize to Cyrillic equivalents
        # These characters look identical but have different Unicode codepoints
        homoglyph_map = {
            # Latin → Cyrillic
            "B": "В",  # Latin B (U+0042) → Cyrillic Ve (U+0412)
            "p": "р",  # Latin p (U+0070) → Cyrillic er (U+0440)
            "ə": "ә",  # Latin schwa (U+0259) → Cyrillic schwa (U+04D9)
        }

        # Apply homoglyph normalization before other normalizations
        for latin_char, cyrillic_char in homoglyph_map.items():
            text = text.replace(latin_char, cyrillic_char)

        # Basic normalization
        # Normalize commas to semicolons for consistent comparison
        return text.strip().lower().replace(",", ";")

    def get_edit_operations(
        self, ground_truth: str, extracted: str
    ) -> List[Tuple[str, str, str, int]]:
        """
        Get detailed edit operations between two strings

        Args:
            ground_truth: Ground truth string
            extracted: Extracted string

        Returns:
            List of tuples: (operation_type, gt_char, ext_char, position)
            operation_type: 'substitution', 'deletion', or 'insertion'
        """
        if self.has_levenshtein:
            # Use Levenshtein library for accurate operations
            ops = Levenshtein.editops(ground_truth, extracted)

            detailed_ops = []
            for op_type, gt_pos, ext_pos in ops:
                if op_type == "replace":
                    # Substitution
                    gt_char = ground_truth[gt_pos] if gt_pos < len(ground_truth) else ""
                    ext_char = extracted[ext_pos] if ext_pos < len(extracted) else ""
                    detailed_ops.append(("substitution", gt_char, ext_char, gt_pos))
                elif op_type == "delete":
                    # Deletion (char was in ground truth but not in extracted)
                    gt_char = ground_truth[gt_pos] if gt_pos < len(ground_truth) else ""
                    detailed_ops.append(("deletion", gt_char, "", gt_pos))
                elif op_type == "insert":
                    # Insertion (char was added in extracted that wasn't in ground truth)
                    ext_char = extracted[ext_pos] if ext_pos < len(extracted) else ""
                    detailed_ops.append(("insertion", "", ext_char, ext_pos))

            return detailed_ops
        else:
            # Fallback to SequenceMatcher
            matcher = SequenceMatcher(None, ground_truth, extracted)
            detailed_ops = []

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "replace":
                    # Handle substitutions
                    for k in range(max(i2 - i1, j2 - j1)):
                        gt_char = ground_truth[i1 + k] if i1 + k < i2 else ""
                        ext_char = extracted[j1 + k] if j1 + k < j2 else ""
                        if gt_char and ext_char:
                            detailed_ops.append(
                                ("substitution", gt_char, ext_char, i1 + k)
                            )
                        elif gt_char:
                            detailed_ops.append(("deletion", gt_char, "", i1 + k))
                        elif ext_char:
                            detailed_ops.append(("insertion", "", ext_char, j1 + k))
                elif tag == "delete":
                    for k in range(i1, i2):
                        detailed_ops.append(("deletion", ground_truth[k], "", k))
                elif tag == "insert":
                    for k in range(j1, j2):
                        detailed_ops.append(("insertion", "", extracted[k], k))

            return detailed_ops

    def calculate_cer_wer(
        self,
        ground_truth_entries: List[Dict[str, str]],
        extracted_entries: List[Dict[str, str]],
        matched_pairs: List[Tuple[Dict[str, str], Dict[str, str]]],
    ) -> Dict:
        """
        Calculate CER and WER with detailed error statistics

        Args:
            ground_truth_entries: List of all ground truth entries
            extracted_entries: List of all extracted entries
            matched_pairs: List of (gt_entry, extracted_entry) tuples that were matched

        Returns:
            Dictionary with detailed error statistics
        """
        # Character-level statistics
        total_chars = 0
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0

        # Track which characters are involved in errors
        substitution_pairs = Counter()  # (original_char, wrong_char): count
        deletion_chars = Counter()  # deleted_char: count
        insertion_chars = Counter()  # inserted_char: count

        # Track word-level context for character errors
        substitution_word_examples = defaultdict(
            list
        )  # (char1, char2): [(gt_word, ext_word)]
        deletion_word_examples = defaultdict(list)  # char: [(gt_word, ext_word)]
        insertion_word_examples = defaultdict(list)  # char: [(gt_word, ext_word)]

        # Word-level statistics
        total_words = 0
        total_word_substitutions = 0
        total_word_deletions = 0
        total_word_insertions = 0

        # Fields to analyze
        fields_to_analyze = [
            "Headword_Phrase",
            "POS",
            "Translation_RU",
            "Grammar_Notes",
        ]

        # Field-specific statistics
        field_stats = {
            field: {"chars": 0, "subs": 0, "dels": 0, "ins": 0}
            for field in fields_to_analyze
        }

        for gt_entry, ext_entry in matched_pairs:
            for field in fields_to_analyze:
                gt_text = str(gt_entry.get(field, "")).strip()
                ext_text = str(ext_entry.get(field, "")).strip()

                if not gt_text:  # Skip empty ground truth fields
                    continue

                # Normalize texts for character-level comparison
                # This applies homoglyph normalization and other text standardization
                gt_text_normalized = self.normalize_text(gt_text)
                ext_text_normalized = self.normalize_text(ext_text)

                # Character-level analysis (use normalized text)
                gt_chars = len(gt_text_normalized)
                total_chars += gt_chars
                field_stats[field]["chars"] += gt_chars

                ops = self.get_edit_operations(gt_text_normalized, ext_text_normalized)

                for op_type, char1, char2, pos in ops:
                    if op_type == "substitution":
                        total_substitutions += 1
                        field_stats[field]["subs"] += 1
                        substitution_pairs[(char1, char2)] += 1
                        # Store all word examples (no limit)
                        substitution_word_examples[(char1, char2)].append(
                            (gt_text, ext_text)
                        )
                    elif op_type == "deletion":
                        total_deletions += 1
                        field_stats[field]["dels"] += 1
                        deletion_chars[char1] += 1
                        # Store all word examples (no limit)
                        deletion_word_examples[char1].append((gt_text, ext_text))
                    elif op_type == "insertion":
                        total_insertions += 1
                        field_stats[field]["ins"] += 1
                        insertion_chars[char2] += 1
                        # Store all word examples (no limit)
                        insertion_word_examples[char2].append((gt_text, ext_text))

                # Word-level analysis (use normalized text)
                gt_words = gt_text_normalized.split()
                ext_words = ext_text_normalized.split()
                total_words += len(gt_words)

                # Calculate word-level edit distance
                if self.has_levenshtein:
                    word_ops = Levenshtein.editops(gt_words, ext_words)
                    for op_type, _, _ in word_ops:
                        if op_type == "replace":
                            total_word_substitutions += 1
                        elif op_type == "delete":
                            total_word_deletions += 1
                        elif op_type == "insert":
                            total_word_insertions += 1
                else:
                    # Fallback: approximate word errors
                    matcher = SequenceMatcher(None, gt_words, ext_words)
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == "replace":
                            total_word_substitutions += max(i2 - i1, j2 - j1)
                        elif tag == "delete":
                            total_word_deletions += i2 - i1
                        elif tag == "insert":
                            total_word_insertions += j2 - j1

        # Calculate rates
        total_char_errors = total_substitutions + total_deletions + total_insertions
        cer = total_char_errors / total_chars if total_chars > 0 else 0

        total_word_errors = (
            total_word_substitutions + total_word_deletions + total_word_insertions
        )
        wer = total_word_errors / total_words if total_words > 0 else 0

        return {
            "cer": cer,
            "wer": wer,
            "total_chars": total_chars,
            "total_words": total_words,
            "total_char_errors": total_char_errors,
            "total_word_errors": total_word_errors,
            "substitutions": {
                "count": total_substitutions,
                "rate": total_substitutions / total_chars if total_chars > 0 else 0,
                "most_common": substitution_pairs.most_common(30),
                "word_examples": dict(substitution_word_examples),
            },
            "deletions": {
                "count": total_deletions,
                "rate": total_deletions / total_chars if total_chars > 0 else 0,
                "most_common": deletion_chars.most_common(30),
                "word_examples": dict(deletion_word_examples),
            },
            "insertions": {
                "count": total_insertions,
                "rate": total_insertions / total_chars if total_chars > 0 else 0,
                "most_common": insertion_chars.most_common(30),
                "word_examples": dict(insertion_word_examples),
            },
            "word_errors": {
                "substitutions": total_word_substitutions,
                "deletions": total_word_deletions,
                "insertions": total_word_insertions,
            },
            "field_stats": field_stats,
        }

    def generate_detailed_report(
        self, error_stats: Dict, output_path: str = "character_error_report.txt"
    ):
        """
        Generate a human-readable report of error statistics

        Args:
            error_stats: Dictionary containing error statistics
            output_path: Path to save the report
        """
        report = []
        report.append("=" * 80)
        report.append("DETAILED CHARACTER-LEVEL ERROR ANALYSIS")
        report.append("=" * 80)
        report.append("")

        # Overall metrics
        report.append(
            f"Character Error Rate (CER): {error_stats['cer']:.4f} ({error_stats['cer']*100:.2f}%)"
        )
        report.append(
            f"Word Error Rate (WER): {error_stats['wer']:.4f} ({error_stats['wer']*100:.2f}%)"
        )
        report.append("")
        report.append(f"Total characters analyzed: {error_stats['total_chars']:,}")
        report.append(f"Total character errors: {error_stats['total_char_errors']:,}")
        report.append("")

        # Character-level error breakdown
        subs = error_stats["substitutions"]
        dels = error_stats["deletions"]
        ins = error_stats["insertions"]
        report.append("CHARACTER-LEVEL ERROR BREAKDOWN:")
        report.append("-" * 80)
        report.append(
            f"  Substitutions: {subs['count']:,} ({subs['rate']*100:.2f}% of all characters)"
        )
        report.append(
            f"  Deletions: {dels['count']:,} ({dels['rate']*100:.2f}% of all characters)"
        )
        report.append(
            f"  Insertions: {ins['count']:,} ({ins['rate']*100:.2f}% of all characters)"
        )
        report.append("")

        # Substitutions
        subs = error_stats["substitutions"]
        report.append("=" * 80)
        report.append(
            f"SUBSTITUTIONS: {subs['count']:,} ({subs['rate']*100:.2f}% of all characters)"
        )
        report.append("=" * 80)
        if subs["most_common"]:
            report.append(
                f"{'Original → Wrong':<35} {'Count':<10} {'% of Subs':<12} {'% of Total'}"
            )
            report.append("-" * 80)
            for (orig, wrong), count in subs["most_common"]:
                pct_subs = (count / subs["count"] * 100) if subs["count"] > 0 else 0
                pct_total = (
                    (count / error_stats["total_chars"] * 100)
                    if error_stats["total_chars"] > 0
                    else 0
                )
                orig_repr = repr(orig) if orig else "''"
                wrong_repr = repr(wrong) if wrong else "''"
                display = f"{orig_repr} → {wrong_repr}"
                report.append(
                    f"{display:<35} {count:<10} {pct_subs:>6.2f}%       {pct_total:>6.2f}%"
                )
                # Add word examples
                if (orig, wrong) in subs["word_examples"]:
                    examples = subs["word_examples"][(orig, wrong)]
                    for gt_word, ext_word in examples:  # Show all examples
                        report.append(f"    Example: {gt_word} → {ext_word}")
        else:
            report.append("No substitutions found.")
        report.append("")

        # Deletions
        dels = error_stats["deletions"]
        report.append("=" * 80)
        report.append(
            f"DELETIONS: {dels['count']:,} ({dels['rate']*100:.2f}% of all characters)"
        )
        report.append("=" * 80)
        if dels["most_common"]:
            report.append(
                f"{'Deleted Character':<35} {'Count':<10} {'% of Dels':<12} {'% of Total'}"
            )
            report.append("-" * 80)
            for char, count in dels["most_common"]:
                pct_dels = (count / dels["count"] * 100) if dels["count"] > 0 else 0
                pct_total = (
                    (count / error_stats["total_chars"] * 100)
                    if error_stats["total_chars"] > 0
                    else 0
                )
                char_repr = repr(char) if char else "''"
                report.append(
                    f"{char_repr:<35} {count:<10} {pct_dels:>6.2f}%       {pct_total:>6.2f}%"
                )
                # Add word examples
                if char in dels["word_examples"]:
                    examples = dels["word_examples"][char]
                    for gt_word, ext_word in examples:  # Show all examples
                        report.append(f"    Example: {gt_word} → {ext_word}")
        else:
            report.append("No deletions found.")
        report.append("")

        # Insertions
        ins = error_stats["insertions"]
        report.append("=" * 80)
        report.append(
            f"INSERTIONS: {ins['count']:,} ({ins['rate']*100:.2f}% of all characters)"
        )
        report.append("=" * 80)
        if ins["most_common"]:
            report.append(
                f"{'Inserted Character':<35} {'Count':<10} {'% of Ins':<12} {'% of Total'}"
            )
            report.append("-" * 80)
            for char, count in ins["most_common"]:
                pct_ins = (count / ins["count"] * 100) if ins["count"] > 0 else 0
                pct_total = (
                    (count / error_stats["total_chars"] * 100)
                    if error_stats["total_chars"] > 0
                    else 0
                )
                char_repr = repr(char) if char else "''"
                report.append(
                    f"{char_repr:<35} {count:<10} {pct_ins:>6.2f}%       {pct_total:>6.2f}%"
                )
                # Add word examples
                if char in ins["word_examples"]:
                    examples = ins["word_examples"][char]
                    for gt_word, ext_word in examples:  # Show all examples
                        report.append(f"    Example: {gt_word} → {ext_word}")
        else:
            report.append("No insertions found.")
        report.append("")

        # Field-level statistics
        report.append("=" * 80)
        report.append("ERROR RATES BY FIELD:")
        report.append("=" * 80)
        report.append(
            f"{'Field':<25} {'Chars':<10} {'Subs':<8} {'Dels':<8} {'Ins':<8} {'CER'}"
        )
        report.append("-" * 80)
        for field, stats in error_stats["field_stats"].items():
            if stats["chars"] > 0:
                field_cer = (stats["subs"] + stats["dels"] + stats["ins"]) / stats[
                    "chars"
                ]
                report.append(
                    f"{field:<25} {stats['chars']:<10} {stats['subs']:<8} "
                    f"{stats['dels']:<8} {stats['ins']:<8} {field_cer:.4f}"
                )
        report.append("")

        # Save report
        report_text = "\n".join(report)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print("\n")
        print(report_text)
        print(f"\nCharacter-level error report saved to: {output_path}")

        # Save detailed JSON
        json_output = output_path.replace(".txt", "_details.json")
        with open(json_output, "w", encoding="utf-8") as f:
            # Convert Counter objects to regular dicts for JSON serialization
            json_data = {
                "cer": error_stats["cer"],
                "wer": error_stats["wer"],
                "total_chars": error_stats["total_chars"],
                "total_words": error_stats["total_words"],
                "total_char_errors": error_stats["total_char_errors"],
                "total_word_errors": error_stats["total_word_errors"],
                "substitutions": {
                    "count": error_stats["substitutions"]["count"],
                    "rate": error_stats["substitutions"]["rate"],
                    "most_common": [
                        {
                            "from": orig,
                            "to": wrong,
                            "count": count,
                            "examples": [
                                {"ground_truth": gt, "extracted": ext}
                                for gt, ext in error_stats["substitutions"][
                                    "word_examples"
                                ].get((orig, wrong), [])
                            ],
                        }
                        for (orig, wrong), count in error_stats["substitutions"][
                            "most_common"
                        ]
                    ],
                },
                "deletions": {
                    "count": error_stats["deletions"]["count"],
                    "rate": error_stats["deletions"]["rate"],
                    "most_common": [
                        {
                            "character": char,
                            "count": count,
                            "examples": [
                                {"ground_truth": gt, "extracted": ext}
                                for gt, ext in error_stats["deletions"][
                                    "word_examples"
                                ].get(char, [])
                            ],
                        }
                        for char, count in error_stats["deletions"]["most_common"]
                    ],
                },
                "insertions": {
                    "count": error_stats["insertions"]["count"],
                    "rate": error_stats["insertions"]["rate"],
                    "most_common": [
                        {
                            "character": char,
                            "count": count,
                            "examples": [
                                {"ground_truth": gt, "extracted": ext}
                                for gt, ext in error_stats["insertions"][
                                    "word_examples"
                                ].get(char, [])
                            ],
                        }
                        for char, count in error_stats["insertions"]["most_common"]
                    ],
                },
                "word_errors": error_stats["word_errors"],
                "field_stats": error_stats["field_stats"],
            }
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"Detailed JSON data saved to: {json_output}")


class DictionaryEvaluator:
    """Evaluates dictionary extraction quality against ground truth"""

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the evaluator

        Args:
            similarity_threshold: Minimum similarity score for partial matches (0-1)
        """
        self.similarity_threshold = similarity_threshold

    def load_tsv(self, filepath: str) -> List[Dict[str, str]]:
        """
        Load a TSV file into a list of dictionaries

        Args:
            filepath: Path to the TSV file

        Returns:
            List of dictionaries representing entries
        """
        entries = []
        with open(filepath, "r", encoding="utf-8") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            for row in reader:
                entries.append(row)
        return entries

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison (lowercase, strip whitespace, etc.)
        Also normalizes homoglyphs (visually identical characters from different scripts)

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if text is None:
            return ""

        # Homoglyph mapping: normalize to Cyrillic equivalents
        # These characters look identical but have different Unicode codepoints
        homoglyph_map = {
            # Latin → Cyrillic
            "B": "В",  # Latin B (U+0042) → Cyrillic Ve (U+0412)
            "p": "р",  # Latin p (U+0070) → Cyrillic er (U+0440)
            "ə": "ә",  # Latin schwa (U+0259) → Cyrillic schwa (U+04D9)
        }

        # Apply homoglyph normalization before other normalizations
        for latin_char, cyrillic_char in homoglyph_map.items():
            text = text.replace(latin_char, cyrillic_char)

        # Basic normalization
        # Normalize commas to semicolons for consistent comparison
        return text.strip().lower().replace(",", ";")

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (0-1)
        """
        if not str1 or not str2:
            return 0.0 if (str1 or str2) else 1.0
        return SequenceMatcher(None, str1, str2).ratio()

    def find_matching_entry(
        self,
        entry: Dict[str, str],
        candidates: List[Dict[str, str]],
        used_indices: Set[int],
    ) -> Tuple[int, float]:
        """
        Find the best matching entry from candidates

        Args:
            entry: Entry to match
            candidates: List of candidate entries
            used_indices: Indices already matched

        Returns:
            Tuple of (best_match_index, similarity_score)
        """
        best_match_idx = -1
        best_score = 0.0

        for idx, candidate in enumerate(candidates):
            if idx in used_indices:
                continue

            # Calculate similarity for headword (most important)
            headword_sim = self.calculate_similarity(
                self.normalize_text(entry.get("Headword_Phrase", "")),
                self.normalize_text(candidate.get("Headword_Phrase", "")),
            )

            # If headword doesn't match well enough, skip
            if headword_sim < 0.7:
                continue

            # Calculate similarity for other fields
            gram_sim = self.calculate_similarity(
                self.normalize_text(entry.get("POS", "")),
                self.normalize_text(candidate.get("POS", "")),
            )

            def_sim = self.calculate_similarity(
                self.normalize_text(entry.get("Translation_RU", "")),
                self.normalize_text(candidate.get("Translation_RU", "")),
            )

            # Weight headword more heavily
            total_score = headword_sim * 0.5 + gram_sim * 0.15 + def_sim * 0.35

            if total_score > best_score:
                best_score = total_score
                best_match_idx = idx

        return best_match_idx, best_score

    def evaluate(
        self, ground_truth_path: str, extracted_path: str
    ) -> EvaluationMetrics:
        """
        Evaluate extracted dictionary against ground truth

        Args:
            ground_truth_path: Path to ground truth TSV file
            extracted_path: Path to extracted TSV file

        Returns:
            EvaluationMetrics object with detailed results
        """
        # Load data
        ground_truth = self.load_tsv(ground_truth_path)
        extracted = self.load_tsv(extracted_path)

        # Track matches
        used_extracted_indices = set()
        exact_matches = 0
        partial_matches = 0
        headword_scores = []
        grammatical_scores = []
        definition_scores = []
        matched_entries = []
        missing_entries = []

        # Match each ground truth entry
        for gt_entry in ground_truth:
            match_idx, match_score = self.find_matching_entry(
                gt_entry, extracted, used_extracted_indices
            )

            if match_idx >= 0:
                used_extracted_indices.add(match_idx)
                extracted_entry = extracted[match_idx]

                # Calculate field-level scores
                headword_score = self.calculate_similarity(
                    self.normalize_text(gt_entry.get("Headword_Phrase", "")),
                    self.normalize_text(extracted_entry.get("Headword_Phrase", "")),
                )
                gram_score = self.calculate_similarity(
                    self.normalize_text(gt_entry.get("POS", "")),
                    self.normalize_text(extracted_entry.get("POS", "")),
                )
                def_score = self.calculate_similarity(
                    self.normalize_text(gt_entry.get("Translation_RU", "")),
                    self.normalize_text(extracted_entry.get("Translation_RU", "")),
                )

                headword_scores.append(headword_score)
                grammatical_scores.append(gram_score)
                definition_scores.append(def_score)

                # Check if exact or partial match
                if headword_score == 1.0 and gram_score >= 0.9 and def_score >= 0.9:
                    exact_matches += 1
                elif match_score >= self.similarity_threshold:
                    partial_matches += 1

                matched_entries.append(
                    {
                        "ground_truth": gt_entry,
                        "extracted": extracted_entry,
                        "scores": {
                            "overall": match_score,
                            "headword": headword_score,
                            "grammatical": gram_score,
                            "definition": def_score,
                        },
                    }
                )
            else:
                missing_entries.append(gt_entry)

        # Find extra entries (in extracted but not in ground truth)
        extra_entries = [
            extracted[i]
            for i in range(len(extracted))
            if i not in used_extracted_indices
        ]

        # Calculate metrics
        total_matched = exact_matches + partial_matches
        precision = total_matched / len(extracted) if extracted else 0
        recall = total_matched / len(ground_truth) if ground_truth else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics = EvaluationMetrics(
            total_ground_truth=len(ground_truth),
            total_extracted=len(extracted),
            exact_matches=exact_matches,
            partial_matches=partial_matches,
            missing_entries=len(missing_entries),
            extra_entries=len(extra_entries),
            headword_accuracy=(
                sum(headword_scores) / len(headword_scores) if headword_scores else 0
            ),
            grammatical_accuracy=(
                sum(grammatical_scores) / len(grammatical_scores)
                if grammatical_scores
                else 0
            ),
            definition_accuracy=(
                sum(definition_scores) / len(definition_scores)
                if definition_scores
                else 0
            ),
            overall_f1=f1,
            precision=precision,
            recall=recall,
        )

        # Store detailed results for analysis
        self.matched_entries = matched_entries
        self.missing_entries = missing_entries
        self.extra_entries = extra_entries

        return metrics

    def generate_report(
        self, metrics: EvaluationMetrics, output_path: str = "evaluation_report.txt"
    ):
        """
        Generate a detailed evaluation report

        Args:
            metrics: EvaluationMetrics object
            output_path: Path to save the report
        """
        report = []
        report.append("=" * 60)
        report.append("DICTIONARY EXTRACTION EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("SUMMARY STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total entries in ground truth: {metrics.total_ground_truth}")
        report.append(f"Total entries extracted: {metrics.total_extracted}")
        report.append(f"Exact matches: {metrics.exact_matches}")
        report.append(f"Partial matches: {metrics.partial_matches}")
        report.append(f"Missing entries: {metrics.missing_entries}")
        report.append(f"Extra entries: {metrics.extra_entries}")
        report.append("")

        report.append("PERFORMANCE METRICS:")
        report.append("-" * 40)
        report.append(f"Overall F1 Score: {metrics.overall_f1:.3f}")
        report.append(f"Precision: {metrics.precision:.3f}")
        report.append(f"Recall: {metrics.recall:.3f}")
        report.append("")

        report.append("FIELD-LEVEL ACCURACY:")
        report.append("-" * 40)
        report.append(f"Headword accuracy: {metrics.headword_accuracy:.3f}")
        report.append(f"Grammatical info accuracy: {metrics.grammatical_accuracy:.3f}")
        report.append(f"Definition accuracy: {metrics.definition_accuracy:.3f}")
        report.append("")

        # Add details about missing entries
        if self.missing_entries:
            report.append("MISSING ENTRIES (First 10):")
            report.append("-" * 40)
            for i, entry in enumerate(self.missing_entries[:10]):
                report.append(
                    f"{i+1}. {entry.get('Headword_Phrase', 'N/A')} ({entry.get('POS', '')})"
                )
            if len(self.missing_entries) > 10:
                report.append(f"... and {len(self.missing_entries) - 10} more")
            report.append("")

        # Add details about extra entries
        if self.extra_entries:
            report.append("EXTRA ENTRIES (First 10):")
            report.append("-" * 40)
            for i, entry in enumerate(self.extra_entries[:10]):
                report.append(
                    f"{i+1}. {entry.get('Headword_Phrase', 'N/A')} ({entry.get('POS', '')})"
                )
            if len(self.extra_entries) > 10:
                report.append(f"... and {len(self.extra_entries) - 10} more")
            report.append("")

        # Add examples of partial matches with low scores
        low_score_matches = [
            m for m in self.matched_entries if m["scores"]["overall"] < 0.9
        ]
        if low_score_matches:
            report.append("LOW-SCORING MATCHES (First 5):")
            report.append("-" * 40)
            for i, match in enumerate(low_score_matches[:5]):
                report.append(
                    f"{i+1}. Ground Truth: {match['ground_truth'].get('Headword_Phrase', 'N/A')}"
                )
                report.append(
                    f"   Extracted: {match['extracted'].get('Headword_Phrase', 'N/A')}"
                )
                report.append(f"   Score: {match['scores']['overall']:.3f}")
                report.append("")

        # Save report
        report_text = "\n".join(report)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to: {output_path}")

        # Also save detailed results as JSON
        json_output = output_path.replace(".txt", "_details.json")
        with open(json_output, "w", encoding="utf-8") as f:
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
                    "low_score_matches": low_score_matches[:10],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Detailed results saved to: {json_output}")


def create_sample_ground_truth():
    """
    Create a sample ground truth file for testing
    This should be replaced with actual ground truth data
    """
    sample_entries = [
        {
            "Headword_Phrase": "aacek",
            "Entry_Type": "word",
            "POS": "сущ.",
            "Translation_RU": "молодой человек, юноша, парень",
            "Literal_Meaning": "",
            "Grammar_Notes": "",
        },
        {
            "Headword_Phrase": "ac-úkwʌn",
            "Entry_Type": "word",
            "POS": "сущ.",
            "Translation_RU": "кремень",
            "Literal_Meaning": "жирный камень",
            "Grammar_Notes": "см. æc",
        },
        # Add more entries based on the actual dictionary page
    ]

    output_path = "/Users/davidsamuel/Documents/Github/dictionary-extractor/ground_truth_sample.tsv"
    with open(output_path, "w", newline="", encoding="utf-8") as tsvfile:
        fieldnames = [
            "Headword_Phrase",
            "Entry_Type",
            "POS",
            "Translation_RU",
            "Literal_Meaning",
            "Grammar_Notes",
        ]
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for entry in sample_entries:
            writer.writerow(entry)
    print(f"Sample ground truth created at: {output_path}")


def main():
    """
    Main function to run the evaluation
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate dictionary extraction results against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_extraction.py -e extracted.tsv -g ground_truth.tsv
  
  # With custom output directory
  python evaluate_extraction.py -e extracted.tsv -g ground_truth.tsv -o results/
  
  # With custom similarity threshold
  python evaluate_extraction.py -e extracted.tsv -g ground_truth.tsv -t 0.90
  
  # Disable character-level error analysis
  python evaluate_extraction.py -e extracted.tsv -g ground_truth.tsv --no-char-analysis
        """,
    )

    parser.add_argument(
        "-e",
        "--extracted",
        type=str,
        required=True,
        help="Path to the extracted TSV file",
    )

    parser.add_argument(
        "-g",
        "--ground-truth",
        type=str,
        required=True,
        help="Path to the ground truth TSV file",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: same as extracted file)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for partial matches (default: 0.85)",
    )

    parser.add_argument(
        "--no-char-analysis",
        action="store_true",
        help="Skip character-level error analysis",
    )

    args = parser.parse_args()

    # Validate paths
    extracted_path = Path(args.extracted)
    if not extracted_path.exists():
        print(f"Error: Extracted TSV file not found at {extracted_path}")
        return 1

    ground_truth_path = Path(args.ground_truth)
    if not ground_truth_path.exists():
        print(f"Error: Ground truth TSV file not found at {ground_truth_path}")
        print("\nPlease create a ground truth file with actual dictionary entries")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = extracted_path.parent

    # Generate output paths
    report_path = output_dir / "evaluation_report.txt"
    char_error_report_path = output_dir / "character_error_report.txt"

    # Create evaluator and run evaluation
    evaluator = DictionaryEvaluator(similarity_threshold=args.threshold)

    print(f"Evaluating extraction results...")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Extracted: {extracted_path}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    metrics = evaluator.evaluate(str(ground_truth_path), str(extracted_path))
    evaluator.generate_report(metrics, str(report_path))

    # Run detailed character-level error analysis
    if not args.no_char_analysis:
        print("\n" + "=" * 60)
        print("Running detailed character-level error analysis...")
        print("=" * 60)

        # Create matched pairs for detailed analysis
        ground_truth = evaluator.load_tsv(str(ground_truth_path))
        extracted = evaluator.load_tsv(str(extracted_path))

        # Get matched pairs from the evaluator's stored results
        matched_pairs = [
            (match["ground_truth"], match["extracted"])
            for match in evaluator.matched_entries
        ]

        if matched_pairs:
            error_analyzer = DetailedErrorAnalyzer()
            error_stats = error_analyzer.calculate_cer_wer(
                ground_truth, extracted, matched_pairs
            )
            error_analyzer.generate_detailed_report(
                error_stats, str(char_error_report_path)
            )
        else:
            print(
                "\nNo matched entries found. Skipping character-level error analysis."
            )
    else:
        print("\nSkipping character-level error analysis (--no-char-analysis flag)")

    return 0


if __name__ == "__main__":
    main()
