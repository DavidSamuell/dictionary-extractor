"""String similarity helpers for MDF record and line alignment."""

from __future__ import annotations

from difflib import SequenceMatcher


def value_similarity(a: str, b: str) -> float:
    """Return normalized Levenshtein ratio proxy via SequenceMatcher.

    Args:
        a: First normalized value string.
        b: Second normalized value string.

    Returns:
        Similarity in ``[0, 1]``.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()
