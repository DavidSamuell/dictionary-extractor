"""ReadOrderEdit for semantically aligned Stage 1 spans."""

import Levenshtein

from dictextractor.evaluation.stage1.alignment import AlignmentResult
from dictextractor.evaluation.stage1.stage1_metrics import ReadOrderMetrics


def compute_read_order(alignment: AlignmentResult) -> ReadOrderMetrics:
    """Compute NED over matched gold span IDs in predicted reading order."""
    gold_pairs = [pair for pair in alignment.pairs if pair.gold]
    gold_pairs.sort(key=lambda pair: pair.gold.start_index)  # type: ignore[union-attr]
    gold_sequence = [pair.gold.span_id for pair in gold_pairs if pair.gold]

    pred_pairs = [pair for pair in alignment.pairs if pair.pred]
    pred_pairs.sort(key=lambda pair: pair.pred.start_index)  # type: ignore[union-attr]
    pred_sequence: list[str] = []
    for idx, pair in enumerate(pred_pairs):
        if pair.gold:
            pred_sequence.append(pair.gold.span_id)
        else:
            pred_sequence.append(f"extra:{idx}")

    dist = Levenshtein.distance(pred_sequence, gold_sequence)
    maxlen = max(len(pred_sequence), len(gold_sequence), 1)
    read_order_edit = dist / maxlen

    return ReadOrderMetrics(
        read_order_edit=read_order_edit,
        edit_distance=dist,
        max_length=maxlen,
    )
