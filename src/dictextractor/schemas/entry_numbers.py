"""Normalise homograph and sense labels to integers 1, 2, 3, …"""

from __future__ import annotations

import re
from typing import Optional

_ROMAN_VALUES: dict[str, int] = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
}


def normalize_entry_number(value: object) -> Optional[int]:
    """
    Convert a printed sense/homograph label to a 1-based integer.

    Accepts Arabic digits, Roman numerals, and common punctuation wrappers
    (``1)``, ``2.``, ``(3)``). Returns ``None`` when empty or unrecognised.

    Time: O(1); Space: O(1).
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None

    text = str(value).strip()
    if not text:
        return None

    text = re.sub(r"^[\(\[\s]+", "", text)
    text = re.sub(r"[\.\)\]\s]+$", "", text)
    text = text.strip()
    if not text:
        return None

    if text.isdigit():
        number = int(text)
        return number if number >= 1 else None

    roman = _ROMAN_VALUES.get(text.upper())
    if roman is not None:
        return roman

    return None
