"""
DetailedErrorAnalyzer: character-level CER and WER analysis.
Migrated from src/evaluate_extraction.py.
"""

import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

from dictextractor.utils.text import normalize_text

try:
    import Levenshtein
    _HAS_LEVENSHTEIN = True
except ImportError:
    _HAS_LEVENSHTEIN = False
    print("Warning: python-Levenshtein not installed. Falling back to difflib.")


class DetailedErrorAnalyzer:
    """
    Computes character-level (CER) and word-level (WER) error statistics
    across matched entry pairs, with per-field breakdowns and concrete examples.
    """

    # Fields compared during CER/WER analysis
    FIELDS = ["Headword_Phrase", "POS", "Translation_RU", "Grammar_Notes"]

    def get_edit_operations(self, gt: str, ext: str) -> List[Tuple[str, str, str, int]]:
        """
        Return detailed edit operations between two strings.

        Returns:
            List of (op_type, gt_char, ext_char, position).
            op_type is one of: 'substitution', 'deletion', 'insertion'.
        """
        if _HAS_LEVENSHTEIN:
            ops = Levenshtein.editops(gt, ext)
            result = []
            for op, gp, ep in ops:
                if op == "replace":
                    result.append(("substitution", gt[gp] if gp < len(gt) else "", ext[ep] if ep < len(ext) else "", gp))
                elif op == "delete":
                    result.append(("deletion", gt[gp] if gp < len(gt) else "", "", gp))
                elif op == "insert":
                    result.append(("insertion", "", ext[ep] if ep < len(ext) else "", ep))
            return result
        else:
            matcher = SequenceMatcher(None, gt, ext)
            result = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "replace":
                    for k in range(max(i2 - i1, j2 - j1)):
                        gc = gt[i1 + k] if i1 + k < i2 else ""
                        ec = ext[j1 + k] if j1 + k < j2 else ""
                        if gc and ec:
                            result.append(("substitution", gc, ec, i1 + k))
                        elif gc:
                            result.append(("deletion", gc, "", i1 + k))
                        elif ec:
                            result.append(("insertion", "", ec, j1 + k))
                elif tag == "delete":
                    for k in range(i1, i2):
                        result.append(("deletion", gt[k], "", k))
                elif tag == "insert":
                    for k in range(j1, j2):
                        result.append(("insertion", "", ext[k], k))
            return result

    def calculate_cer_wer(
        self,
        ground_truth_entries: List[Dict],
        extracted_entries: List[Dict],
        matched_pairs: List[Tuple[Dict, Dict]],
    ) -> Dict:
        total_chars = total_subs = total_dels = total_ins = 0
        total_words = total_wsubs = total_wdels = total_wins = 0

        sub_pairs: Counter = Counter()
        del_chars: Counter = Counter()
        ins_chars: Counter = Counter()
        sub_examples: dict = defaultdict(list)
        del_examples: dict = defaultdict(list)
        ins_examples: dict = defaultdict(list)

        field_stats = {f: {"chars": 0, "subs": 0, "dels": 0, "ins": 0} for f in self.FIELDS}

        for gt_entry, ext_entry in matched_pairs:
            for field in self.FIELDS:
                gt_text = normalize_text(str(gt_entry.get(field, "")))
                ext_text = normalize_text(str(ext_entry.get(field, "")))
                if not gt_text:
                    continue

                n = len(gt_text)
                total_chars += n
                field_stats[field]["chars"] += n

                for op, c1, c2, _ in self.get_edit_operations(gt_text, ext_text):
                    if op == "substitution":
                        total_subs += 1
                        field_stats[field]["subs"] += 1
                        sub_pairs[(c1, c2)] += 1
                        sub_examples[(c1, c2)].append((str(gt_entry.get(field, "")), str(ext_entry.get(field, ""))))
                    elif op == "deletion":
                        total_dels += 1
                        field_stats[field]["dels"] += 1
                        del_chars[c1] += 1
                        del_examples[c1].append((str(gt_entry.get(field, "")), str(ext_entry.get(field, ""))))
                    elif op == "insertion":
                        total_ins += 1
                        field_stats[field]["ins"] += 1
                        ins_chars[c2] += 1
                        ins_examples[c2].append((str(gt_entry.get(field, "")), str(ext_entry.get(field, ""))))

                gt_words = gt_text.split()
                ext_words = ext_text.split()
                total_words += len(gt_words)

                if _HAS_LEVENSHTEIN:
                    for op, _, _ in Levenshtein.editops(gt_words, ext_words):
                        if op == "replace": total_wsubs += 1
                        elif op == "delete": total_wdels += 1
                        elif op == "insert": total_wins += 1
                else:
                    matcher = SequenceMatcher(None, gt_words, ext_words)
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == "replace": total_wsubs += max(i2 - i1, j2 - j1)
                        elif tag == "delete": total_wdels += i2 - i1
                        elif tag == "insert": total_wins += j2 - j1

        total_char_errors = total_subs + total_dels + total_ins
        total_word_errors = total_wsubs + total_wdels + total_wins
        cer = total_char_errors / total_chars if total_chars else 0
        wer = total_word_errors / total_words if total_words else 0

        return {
            "cer": cer, "wer": wer,
            "total_chars": total_chars, "total_words": total_words,
            "total_char_errors": total_char_errors, "total_word_errors": total_word_errors,
            "substitutions": {
                "count": total_subs,
                "rate": total_subs / total_chars if total_chars else 0,
                "most_common": sub_pairs.most_common(30),
                "word_examples": dict(sub_examples),
            },
            "deletions": {
                "count": total_dels,
                "rate": total_dels / total_chars if total_chars else 0,
                "most_common": del_chars.most_common(30),
                "word_examples": dict(del_examples),
            },
            "insertions": {
                "count": total_ins,
                "rate": total_ins / total_chars if total_chars else 0,
                "most_common": ins_chars.most_common(30),
                "word_examples": dict(ins_examples),
            },
            "word_errors": {"substitutions": total_wsubs, "deletions": total_wdels, "insertions": total_wins},
            "field_stats": field_stats,
        }

    def generate_detailed_report(self, error_stats: Dict, output_path: str = "character_error_report.txt"):
        report = [
            "=" * 80,
            "DETAILED CHARACTER-LEVEL ERROR ANALYSIS",
            "=" * 80,
            "",
            f"Character Error Rate (CER): {error_stats['cer']:.4f} ({error_stats['cer']*100:.2f}%)",
            f"Word Error Rate      (WER): {error_stats['wer']:.4f} ({error_stats['wer']*100:.2f}%)",
            "",
            f"Total characters analysed: {error_stats['total_chars']:,}",
            f"Total character errors:    {error_stats['total_char_errors']:,}",
            "",
        ]

        for section, label in [
            ("substitutions", "SUBSTITUTIONS"),
            ("deletions",     "DELETIONS"),
            ("insertions",    "INSERTIONS"),
        ]:
            s = error_stats[section]
            report += [
                "=" * 80,
                f"{label}: {s['count']:,} ({s['rate']*100:.2f}% of all characters)",
                "=" * 80,
            ]
            if s["most_common"]:
                if section == "substitutions":
                    report.append(f"{'Original → Wrong':<35} {'Count':<10} {'% of Subs':<12} {'% of Total'}")
                    report.append("-" * 80)
                    for (c1, c2), cnt in s["most_common"]:
                        pct_s = (cnt / s["count"] * 100) if s["count"] else 0
                        pct_t = (cnt / error_stats["total_chars"] * 100) if error_stats["total_chars"] else 0
                        display = f"{repr(c1)} → {repr(c2)}"
                        report.append(f"{display:<35} {cnt:<10} {pct_s:>6.2f}%       {pct_t:>6.2f}%")
                        for gt_w, ext_w in s["word_examples"].get((c1, c2), []):
                            report.append(f"    Example: {gt_w} → {ext_w}")
                else:
                    char_key = "character"
                    header_label = "Deleted Character" if section == "deletions" else "Inserted Character"
                    report.append(f"{header_label:<35} {'Count':<10} {'%':<12} {'% of Total'}")
                    report.append("-" * 80)
                    for char, cnt in s["most_common"]:
                        pct_s = (cnt / s["count"] * 100) if s["count"] else 0
                        pct_t = (cnt / error_stats["total_chars"] * 100) if error_stats["total_chars"] else 0
                        report.append(f"{repr(char):<35} {cnt:<10} {pct_s:>6.2f}%       {pct_t:>6.2f}%")
                        for gt_w, ext_w in s["word_examples"].get(char, []):
                            report.append(f"    Example: {gt_w} → {ext_w}")
            else:
                report.append(f"No {section} found.")
            report.append("")

        report += [
            "=" * 80,
            "ERROR RATES BY FIELD:",
            "=" * 80,
            f"{'Field':<25} {'Chars':<10} {'Subs':<8} {'Dels':<8} {'Ins':<8} {'CER'}",
            "-" * 80,
        ]
        for field, stats in error_stats["field_stats"].items():
            if stats["chars"] > 0:
                field_cer = (stats["subs"] + stats["dels"] + stats["ins"]) / stats["chars"]
                report.append(f"{field:<25} {stats['chars']:<10} {stats['subs']:<8} {stats['dels']:<8} {stats['ins']:<8} {field_cer:.4f}")
        report.append("")

        report_text = "\n".join(report)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print("\n")
        print(report_text)
        print(f"\nCharacter-level error report saved to: {output_path}")

        json_out = output_path.replace(".txt", "_details.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
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
                            {"from": c1, "to": c2, "count": cnt,
                             "examples": [{"ground_truth": g, "extracted": e}
                                          for g, e in error_stats["substitutions"]["word_examples"].get((c1, c2), [])]}
                            for (c1, c2), cnt in error_stats["substitutions"]["most_common"]
                        ],
                    },
                    "deletions": {
                        "count": error_stats["deletions"]["count"],
                        "rate": error_stats["deletions"]["rate"],
                        "most_common": [
                            {"character": char, "count": cnt,
                             "examples": [{"ground_truth": g, "extracted": e}
                                          for g, e in error_stats["deletions"]["word_examples"].get(char, [])]}
                            for char, cnt in error_stats["deletions"]["most_common"]
                        ],
                    },
                    "insertions": {
                        "count": error_stats["insertions"]["count"],
                        "rate": error_stats["insertions"]["rate"],
                        "most_common": [
                            {"character": char, "count": cnt,
                             "examples": [{"ground_truth": g, "extracted": e}
                                          for g, e in error_stats["insertions"]["word_examples"].get(char, [])]}
                            for char, cnt in error_stats["insertions"]["most_common"]
                        ],
                    },
                    "word_errors": error_stats["word_errors"],
                    "field_stats": error_stats["field_stats"],
                },
                f, ensure_ascii=False, indent=2,
            )
        print(f"Detailed JSON data saved to: {json_out}")
