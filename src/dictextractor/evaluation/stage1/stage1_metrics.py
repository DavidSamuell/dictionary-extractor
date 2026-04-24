"""
Stage1Metrics: dataclass for all Stage 1 OCR evaluation results.

Three evaluation dimensions:
1. Character recognition quality  (GCER, CER, WER, BLEU, NED on tag-stripped text)
2. Markup/typography preservation (bold/italic precision, recall, F1)
3. Structure preservation         (read-order NED)
"""

from dataclasses import dataclass, field


@dataclass
class TagMetrics:
    """Precision / recall / F1 for a single tag type (e.g. bold or italic)."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class CharacterQualityMetrics:
    """Aggregated character-recognition quality."""

    gcer: float = 0.0  # Grapheme Character Error Rate (UAX #29 grapheme clusters)
    cer: float = 0.0  # Character Error Rate (raw Unicode code points)
    wer: float = 0.0  # Word Error Rate
    bleu: float = 0.0  # BLEU score (word-level, corpus-level smoothed)
    ned: float = 0.0  # Normalized Edit Distance

    total_graphemes_gold: int = 0
    total_graphemes_pred: int = 0
    total_grapheme_edits: int = 0
    total_chars_gold: int = 0
    total_chars_pred: int = 0
    total_char_edits: int = 0
    total_words_gold: int = 0
    total_word_edits: int = 0

    matched_lines: int = 0
    missing_lines: int = 0  # in gold but not in pred
    extra_lines: int = 0  # in pred but not in gold


@dataclass
class MarkupQualityMetrics:
    """Markup preservation quality, per tag type."""

    bold: TagMetrics = field(default_factory=TagMetrics)
    italic: TagMetrics = field(default_factory=TagMetrics)


@dataclass
class ReadOrderMetrics:
    """Structure / reading-order preservation."""

    ned: float = 0.0  # NED on the concatenated strings
    edit_distance: int = 0
    max_length: int = 0
    character_ned: float = 0.0  # per-line NED (from CharacterQuality) for reference
    isolated_order_error: float = 0.0  # ned - character_ned (order-attributable error)


@dataclass
class Stage1Metrics:
    """Top-level container for all Stage 1 evaluation results."""

    page_id: str = ""
    character_quality: CharacterQualityMetrics = field(
        default_factory=CharacterQualityMetrics
    )
    markup_quality: MarkupQualityMetrics = field(
        default_factory=MarkupQualityMetrics
    )
    read_order: ReadOrderMetrics = field(default_factory=ReadOrderMetrics)
