"""Character recognition quality for semantically aligned Stage 1 spans."""

import grapheme
import jiwer
import Levenshtein

from dictextractor.evaluation.stage1.alignment import AlignmentResult
from dictextractor.evaluation.stage1.tag_parser import casefold_letters_for_eval
from dictextractor.evaluation.stage1.stage1_metrics import CharacterQualityMetrics

_JIWER_WORD_TRANSFORM = jiwer.Compose([jiwer.ReduceToListOfListOfWords()])


def compute_character_quality(alignment: AlignmentResult) -> CharacterQualityMetrics:
    """Compute TextEdit, GCER, and WER after semantic span alignment."""
    total_grapheme_edits = 0
    total_graphemes_gold = 0
    total_graphemes_pred = 0
    total_words_gold = 0
    matched = missing = extra = 0
    text_edit_sum = 0.0

    gold_spans: list[str] = []
    pred_spans: list[str] = []

    for pair in alignment.pairs:
        g = pair.gold.text if pair.gold else ""
        p = pair.pred.text if pair.pred else ""
        text_edit_sum += pair.text_edit

        if pair.gold and pair.pred:
            matched += 1
        elif pair.gold:
            missing += 1
        elif pair.pred:
            extra += 1

        pg = list(grapheme.graphemes(casefold_letters_for_eval(p)))
        gg = list(grapheme.graphemes(casefold_letters_for_eval(g)))
        total_grapheme_edits += Levenshtein.distance(pg, gg)
        total_graphemes_gold += len(gg)
        total_graphemes_pred += len(pg)
        total_words_gold += len(g.split())
        gold_spans.append(casefold_letters_for_eval(g))
        pred_spans.append(casefold_letters_for_eval(p))

    wer = 0.0
    if gold_spans or pred_spans:
        word_output = jiwer.process_words(
            gold_spans,
            pred_spans,
            reference_transform=_JIWER_WORD_TRANSFORM,
            hypothesis_transform=_JIWER_WORD_TRANSFORM,
        )
        wer = float(word_output.wer)

    text_edit = text_edit_sum / len(alignment.pairs) if alignment.pairs else 0.0
    gcer = (
        total_grapheme_edits / total_graphemes_gold
        if total_graphemes_gold else 0.0
    )
    total_word_edits = round(wer * total_words_gold)

    return CharacterQualityMetrics(
        text_edit=text_edit,
        gcer=gcer,
        wer=wer,
        total_graphemes_gold=total_graphemes_gold,
        total_graphemes_pred=total_graphemes_pred,
        total_grapheme_edits=total_grapheme_edits,
        total_words_gold=total_words_gold,
        total_word_edits=total_word_edits,
        matched_spans=matched,
        missing_spans=missing,
        extra_spans=extra,
    )
