"""
Markup / typography preservation metrics for Stage 1 evaluation.

For each aligned span, aligns words between predicted and gold using fuzzy matching,
then checks whether bold/italic tags are preserved correctly.
"""

from difflib import SequenceMatcher
from typing import FrozenSet, List, Tuple

import Levenshtein

from dictextractor.evaluation.stage1.alignment import AlignmentResult
from dictextractor.evaluation.stage1.stage1_metrics import MarkupQualityMetrics, TagMetrics
from dictextractor.evaluation.stage1.tag_parser import (
    normalize_line_for_markup,
    normalize_word_for_markup_align,
    parse_tagged_words,
    strip_tags,
)

TaggedWord = Tuple[str, FrozenSet[str]]

# Minimum character-level similarity to accept a word alignment pair.
_ALIGN_THRESHOLD = 0.5


def _word_similarity(a: str, b: str) -> float:
    """Normalised character-level similarity between two aligned word keys."""
    a = normalize_word_for_markup_align(a)
    b = normalize_word_for_markup_align(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    maxlen = max(len(a), len(b))
    return 1.0 - Levenshtein.distance(a, b) / maxlen


def _align_words(
    pred_words: List[TaggedWord],
    gold_words: List[TaggedWord],
) -> List[Tuple[TaggedWord | None, TaggedWord | None]]:
    """Align predicted words to gold words within a single line.

    Uses SequenceMatcher on stripped word text for alignment, then filters
    by a minimum character similarity threshold.

    Returns a list of (pred_word, gold_word) pairs.  Either element may be
    ``None`` if there is no acceptable match (insertion / deletion).
    """
    pred_stripped = [
        normalize_word_for_markup_align(strip_tags(w)) for w, _ in pred_words
    ]
    gold_stripped = [
        normalize_word_for_markup_align(strip_tags(w)) for w, _ in gold_words
    ]

    matcher = SequenceMatcher(None, pred_stripped, gold_stripped)
    pairs: List[Tuple[TaggedWord | None, TaggedWord | None]] = []

    for op, pi1, pi2, gi1, gi2 in matcher.get_opcodes():
        if op == "equal":
            for pi, gi in zip(range(pi1, pi2), range(gi1, gi2)):
                pairs.append((pred_words[pi], gold_words[gi]))
        elif op == "replace":
            # Try to pair up words by position; check similarity.
            plen = pi2 - pi1
            glen = gi2 - gi1
            for k in range(max(plen, glen)):
                pw = pred_words[pi1 + k] if k < plen else None
                gw = gold_words[gi1 + k] if k < glen else None
                if pw and gw:
                    sim = _word_similarity(pw[0], gw[0])
                    if sim >= _ALIGN_THRESHOLD:
                        pairs.append((pw, gw))
                    else:
                        # Treat as separate insertion + deletion
                        pairs.append((pw, None))
                        pairs.append((None, gw))
                elif pw:
                    pairs.append((pw, None))
                elif gw:
                    pairs.append((None, gw))
        elif op == "delete":
            for pi in range(pi1, pi2):
                pairs.append((pred_words[pi], None))
        elif op == "insert":
            for gi in range(gi1, gi2):
                pairs.append((None, gold_words[gi]))

    return pairs


def _evaluate_tag(
    pairs: List[Tuple[TaggedWord | None, TaggedWord | None]],
    tag: str,
) -> TagMetrics:
    """Count TP / FP / FN for a single tag across aligned word pairs."""
    tp = fp = fn = 0
    for pw, gw in pairs:
        pred_has = tag in pw[1] if pw else False
        gold_has = tag in gw[1] if gw else False

        if gold_has and pred_has:
            tp += 1
        elif gold_has and not pred_has:
            fn += 1
        elif not gold_has and pred_has:
            fp += 1
        # both False → true negative, not counted
    return TagMetrics(true_positives=tp, false_positives=fp, false_negatives=fn)


def compute_markup_quality(alignment: AlignmentResult) -> MarkupQualityMetrics:
    """Evaluate markup preservation across semantically aligned spans."""
    all_pairs: List[Tuple[TaggedWord | None, TaggedWord | None]] = []

    for pair in alignment.pairs:
        if pair.pred and pair.gold:
            pred_words = parse_tagged_words(
                normalize_line_for_markup(pair.pred.tagged_text)
            )
            gold_words = parse_tagged_words(
                normalize_line_for_markup(pair.gold.tagged_text)
            )
            all_pairs.extend(_align_words(pred_words, gold_words))
        elif pair.gold:
            gold_words = parse_tagged_words(
                normalize_line_for_markup(pair.gold.tagged_text)
            )
            all_pairs.extend((None, gw) for gw in gold_words)
        elif pair.pred:
            pred_words = parse_tagged_words(
                normalize_line_for_markup(pair.pred.tagged_text)
            )
            all_pairs.extend((pw, None) for pw in pred_words)

    bold = _evaluate_tag(all_pairs, "b")
    italic = _evaluate_tag(all_pairs, "i")

    return MarkupQualityMetrics(bold=bold, italic=italic)
