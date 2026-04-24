"""
Character recognition quality metrics for Stage 1 evaluation.

Computes GCER, CER, WER, BLEU, and NED between tag-stripped predicted and
gold text, aligned by (column_id, line_number).

- GCER (Grapheme Character Error Rate) uses UAX #29 grapheme clusters as the
  unit of comparison via the ``grapheme`` library + ``python-Levenshtein``.
  This is the OCR-D standard definition of CER and is appropriate for scripts
  where a single visual character spans multiple Unicode code points
  (Devanagari, Arabic with diacritics, Thai, etc.).

- CER (Character Error Rate) operates on raw Unicode code points, computed via
  ``jiwer.process_characters``.  Kept for comparability with benchmarks that
  use the traditional ISRI/ocreval definition.

- WER (Word Error Rate) is computed via ``jiwer.process_words`` (whitespace
  tokenisation, consistent with the ISRI/ocreval word boundary convention).

- BLEU is computed corpus-level via ``sacrebleu`` with ``tokenize="intl"``
  (Moses v14 international tokeniser).  The ``intl`` tokeniser handles
  Latin, Cyrillic, Arabic, and CJK scripts without relying on whitespace,
  making it language-agnostic and reproducible across benchmarks.
"""

from typing import Dict, List, Tuple

import grapheme
import jiwer
import Levenshtein
from sacrebleu.compat import corpus_bleu as sacrebleu_corpus_bleu

from dictextractor.evaluation.stage1.stage1_metrics import CharacterQualityMetrics
from dictextractor.evaluation.stage1.tag_parser import (
    normalize_unicode,
    normalize_whitespace,
    strip_tags,
)

# Type alias for a TSV row: {column_id, line_number, text}
Row = Dict[str, str]

# jiwer transforms that match our pre-cleaned strings exactly:
# - text is already NFC-normalised and whitespace-collapsed by _clean()
# - we only need the final reduction steps, no additional stripping
_JIWER_CHAR_TRANSFORM = jiwer.Compose([jiwer.ReduceToListOfListOfChars()])
_JIWER_WORD_TRANSFORM = jiwer.Compose([jiwer.ReduceToListOfListOfWords()])


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

    Metrics:
        - GCER / NED: custom, grapheme-cluster level (``grapheme`` + ``Levenshtein``)
        - CER: ``jiwer.process_characters`` (raw Unicode code points)
        - WER: ``jiwer.process_words`` (whitespace tokenisation)
        - BLEU: ``sacrebleu`` corpus BLEU with ``tokenize="intl"``
    """
    pred_map = {_key(r): _clean(r["text"]) for r in pred_rows}
    gold_map = {_key(r): _clean(r["text"]) for r in gold_rows}

    all_keys = sorted(set(pred_map) | set(gold_map))

    total_grapheme_edits = 0
    total_graphemes_gold = 0
    total_graphemes_pred = 0
    total_chars_gold = 0
    total_chars_pred = 0
    total_words_gold = 0
    matched = missing = extra = 0

    gold_lines: List[str] = []
    pred_lines: List[str] = []

    for key in all_keys:
        g = gold_map.get(key, "")
        p = pred_map.get(key, "")

        if key not in pred_map:
            missing += 1
        elif key not in gold_map:
            extra += 1
        else:
            matched += 1

        # Grapheme-level (GCER + NED) — no library equivalent
        pg = list(grapheme.graphemes(p))
        gg = list(grapheme.graphemes(g))
        total_grapheme_edits += Levenshtein.distance(pg, gg)
        total_graphemes_gold += len(gg)
        total_graphemes_pred += len(pg)

        # Code-point totals for CER denominator reporting
        total_chars_gold += len(g)
        total_chars_pred += len(p)

        # Word totals for WER denominator reporting
        gw = g.split()
        total_words_gold += len(gw)

        gold_lines.append(g)
        pred_lines.append(p)

    # CER via jiwer (code-point level, corpus-level micro-average)
    char_output = jiwer.process_characters(
        gold_lines,
        pred_lines,
        reference_transform=_JIWER_CHAR_TRANSFORM,
        hypothesis_transform=_JIWER_CHAR_TRANSFORM,
    )
    cer: float = char_output.cer

    # WER via jiwer (whitespace tokenisation, corpus-level micro-average)
    word_output = jiwer.process_words(
        gold_lines,
        pred_lines,
        reference_transform=_JIWER_WORD_TRANSFORM,
        hypothesis_transform=_JIWER_WORD_TRANSFORM,
    )
    wer: float = word_output.wer

    # Corpus-level BLEU via sacrebleu, intl tokeniser
    bleu_result = sacrebleu_corpus_bleu(
        pred_lines,
        [gold_lines],
        tokenize="intl",
    )
    bleu = bleu_result.score / 100.0  # sacrebleu returns 0–100

    gcer = total_grapheme_edits / total_graphemes_gold if total_graphemes_gold else 0.0
    max_graphemes = max(total_graphemes_gold, total_graphemes_pred, 1)
    ned = total_grapheme_edits / max_graphemes

    # Derive edit counts from jiwer rates for use in aggregate reporting.
    # These are approximate (rounding) but consistent with jiwer's own rate
    # calculations and sufficient for micro-average aggregation.
    total_char_edits = round(cer * total_chars_gold)
    total_word_edits = round(wer * total_words_gold)

    return CharacterQualityMetrics(
        gcer=gcer,
        cer=cer,
        wer=wer,
        bleu=bleu,
        ned=ned,
        total_graphemes_gold=total_graphemes_gold,
        total_graphemes_pred=total_graphemes_pred,
        total_grapheme_edits=total_grapheme_edits,
        total_chars_gold=total_chars_gold,
        total_chars_pred=total_chars_pred,
        total_char_edits=total_char_edits,
        total_words_gold=total_words_gold,
        total_word_edits=total_word_edits,
        matched_lines=matched,
        missing_lines=missing,
        extra_lines=extra,
    )
