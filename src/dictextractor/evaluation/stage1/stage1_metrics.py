"""
Stage1Metrics: dataclass for all Stage 1 OCR evaluation results.

Three evaluation dimensions (flat eval reports all three):
1. Character recognition quality  (TextEdit, GCER, WER on aligned spans)
2. Markup/typography preservation (bold/italic precision, recall, F1)
3. Structure preservation (ReadOrderEdit) — OmniDocBench-style over gold line indices
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

    text_edit: float = 0.0  # Mean NED over aligned/unmatched spans.
    gcer: float = 0.0  # Grapheme Character Error Rate (UAX #29 grapheme clusters)
    wer: float = 0.0  # Word Error Rate

    total_graphemes_gold: int = 0
    total_graphemes_pred: int = 0
    total_grapheme_edits: int = 0
    total_words_gold: int = 0
    total_word_edits: int = 0

    matched_spans: int = 0
    missing_spans: int = 0  # in gold but not in pred
    extra_spans: int = 0  # in pred but not in gold


@dataclass
class MarkupQualityMetrics:
    """Markup preservation quality, per tag type."""

    bold: TagMetrics = field(default_factory=TagMetrics)
    italic: TagMetrics = field(default_factory=TagMetrics)

    @property
    def typography(self) -> TagMetrics:
        """Pooled bold + italic counts (micro-average F1 source)."""
        return TagMetrics(
            true_positives=(
                self.bold.true_positives + self.italic.true_positives
            ),
            false_positives=(
                self.bold.false_positives + self.italic.false_positives
            ),
            false_negatives=(
                self.bold.false_negatives + self.italic.false_negatives
            ),
        )


@dataclass
class ReadOrderMetrics:
    """Structure / reading-order preservation."""

    read_order_edit: float = 0.0
    edit_distance: int = 0
    max_length: int = 0


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
