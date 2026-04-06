"""
Character recognition quality metrics for Stage 1 evaluation.

Computes GCER, WER, and NED between tag-stripped predicted and gold text,
aligned by (column_id, line_number).

GCER (Grapheme Character Error Rate) operates on UAX #29 grapheme clusters
rather than raw Unicode code points, ensuring fair measurement for scripts
where a single visual character spans multiple code points (e.g. Devanagari,
Arabic with diacritics, Thai).
"""

from typing import Dict, List, Tuple

import grapheme
import Levenshtein

from dictextractor.evaluation.stage1.stage1_metrics import CharacterQualityMetrics
from dictextractor.evaluation.stage1.tag_parser import (
    normalize_unicode,
    normalize_whitespace,
    strip_tags,
)

# Type alias for a TSV row: {column_id, line_number, text}
Row = Dict[str, str]


def _key(row: Row) -> Tuple[str, int]:
    return (row["column_id"], int(row["line_number"]))


def _clean(text: str) -> str:
    """Strip tags, NFC-normalise, collapse whitespace."""
    return normalize_whitespace(normalize_unicode(strip_tags(text)))


def compute_character_quality(
    pred_rows: List[Row],
    gold_rows: List[Row],
) -> CharacterQualityMetrics:
    """Compare predicted vs gold rows on pure character recognition.

    Rows are aligned by ``(column_id, line_number)``.  Missing rows on either
    side count as full-length errors.
    """
    pred_map = {_key(r): _clean(r["text"]) for r in pred_rows}
    gold_map = {_key(r): _clean(r["text"]) for r in gold_rows}

    all_keys = sorted(set(pred_map) | set(gold_map))

    total_char_edits = 0
    total_chars_gold = 0
    total_chars_pred = 0
    total_word_edits = 0
    total_words_gold = 0
    matched = missing = extra = 0

    for key in all_keys:
        g = gold_map.get(key, "")
        p = pred_map.get(key, "")

        if key not in pred_map:
            missing += 1
        elif key not in gold_map:
            extra += 1
        else:
            matched += 1

        # Character-level (grapheme-cluster aware)
        pg = list(grapheme.graphemes(p))
        gg = list(grapheme.graphemes(g))
        char_dist = Levenshtein.distance(pg, gg)
        total_char_edits += char_dist
        total_chars_gold += len(gg)
        total_chars_pred += len(pg)

        # Word-level
        gw = g.split()
        pw = p.split()
        word_dist = Levenshtein.distance(gw, pw)
        total_word_edits += word_dist
        total_words_gold += len(gw)

    gcer = total_char_edits / total_chars_gold if total_chars_gold else 0.0
    wer = total_word_edits / total_words_gold if total_words_gold else 0.0
    max_chars = max(total_chars_gold, total_chars_pred, 1)
    ned = total_char_edits / max_chars

    return CharacterQualityMetrics(
        gcer=gcer,
        wer=wer,
        ned=ned,
        total_chars_gold=total_chars_gold,
        total_chars_pred=total_chars_pred,
        total_char_edits=total_char_edits,
        total_words_gold=total_words_gold,
        total_word_edits=total_word_edits,
        matched_lines=matched,
        missing_lines=missing,
        extra_lines=extra,
    )
