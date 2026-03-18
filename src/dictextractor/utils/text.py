"""
Text normalization utilities shared across evaluation and extraction modules.
"""


# Canonical homoglyph map: visually-identical characters normalised to Cyrillic.
# Latin → Cyrillic
HOMOGLYPH_MAP: dict[str, str] = {
    "B": "В",   # Latin B  (U+0042) → Cyrillic Ve   (U+0412)
    "p": "р",   # Latin p  (U+0070) → Cyrillic er    (U+0440)
    "ə": "ә",   # Latin schwa (U+0259) → Cyrillic schwa (U+04D9)
}


def normalize_text(text: str) -> str:
    """
    Normalise text for comparison.

    Steps applied in order:
    1. Replace homoglyphs with their Cyrillic equivalents.
    2. Strip leading/trailing whitespace.
    3. Lowercase.
    4. Normalise commas to semicolons for consistent field comparison.

    Args:
        text: Raw text string (may be None).

    Returns:
        Normalised string, or empty string if input is None.
    """
    if text is None:
        return ""

    for latin_char, cyrillic_char in HOMOGLYPH_MAP.items():
        text = text.replace(latin_char, cyrillic_char)

    return text.strip().lower().replace(",", ";")
