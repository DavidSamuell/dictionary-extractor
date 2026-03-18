"""
EvaluationMetrics dataclass — the single container for all evaluation results.
"""

from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for entry-level evaluation results."""

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
