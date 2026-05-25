"""
Utilities for parsing inline HTML tags (<b>, <i>, etc.) in dictionary text.

Provides:
- strip_tags()                    : remove all HTML tags, return plain text
- parse_tagged_words()            : split text into (word, frozenset_of_tags) tuples
- normalize_unicode()             : NFC-normalise for consistent comparison
- normalize_line_text()           : NFC, collapse whitespace, space before punct
- normalize_line_for_markup()     : tagged line normalisation before word parse
- normalize_word_for_markup_align : word key for fuzzy typography alignment
"""

import re
import unicodedata
from typing import List, Tuple, FrozenSet

# Matches opening or closing HTML-style tags: <b>, </i>, <sup>, </sup>, etc.
_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*>")


def strip_tags(text: str) -> str:
    """Remove all inline HTML tags from *text*."""
    return _TAG_RE.sub("", text)


def normalize_unicode(text: str) -> str:
    """Apply NFC normalization so combining diacritics match precomposed forms."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


# Punctuation ignored when aligning words for typography metrics only.
_MARKUP_ALIGN_PUNCT_RE = re.compile(r"[,.\:;!?\'\"()\[\]]")

# Remove whitespace immediately before punctuation (eval + markup normalisation).
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.\:;!?\'\"()\[\]])")


def casefold_letters_for_eval(text: str) -> str:
    """Fold letter case for eval comparison only (not flat export).

    Applied in alignment and character metrics so ``VERB`` vs ``verb`` does not
    inflate GCER/WER. Non-letters (digits, IPA, CJK) are unchanged.
    """
    return "".join(ch.casefold() if ch.isalpha() else ch for ch in text)


def normalize_line_text(text: str) -> str:
    """NFC, collapse whitespace, and remove spaces before punctuation.

    Applied to both pred and gold for semantic alignment (TextEdit, GCER, WER)
    and as the first step before typography word parsing.
    """
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def normalize_line_for_markup(text: str) -> str:
    """Prepare a tagged line for ``parse_tagged_words`` (tags preserved)."""
    return normalize_line_text(text)


def normalize_word_for_markup_align(word: str) -> str:
    """Normalised word surface for typography alignment / similarity.

    Strips punctuation characters used in dictionary typography (``, . : ;``
    etc.) so ``hello.,`` and ``hello`` align as the same slot. Original word
    text and tags are unchanged for TP/FP/FN decisions.
    """
    return _MARKUP_ALIGN_PUNCT_RE.sub("", casefold_letters_for_eval(normalize_unicode(word)))


# ---------------------------------------------------------------------------
# Tagged-word parser
# ---------------------------------------------------------------------------

# Tokenises into tags and non-tag chunks.
_TOKEN_RE = re.compile(r"(</?[a-zA-Z][a-zA-Z0-9]*>)")


def parse_tagged_words(text: str) -> List[Tuple[str, FrozenSet[str]]]:
    """Parse *text* into a list of ``(word, tags)`` tuples.

    *tags* is a frozenset of active tag names (e.g. ``{"b", "i"}``) at the
    point where the word appears.  Words are split on whitespace; tags are
    tracked with a stack so that ``<b>hello world</b>`` yields two words both
    tagged ``{"b"}``.

    Returns an empty list for blank input.
    """
    tokens = _TOKEN_RE.split(text)
    active_tags: dict[str, int] = {}  # tag_name -> nesting depth
    result: List[Tuple[str, FrozenSet[str]]] = []

    for token in tokens:
        if not token:
            continue

        # Opening tag
        m_open = re.fullmatch(r"<([a-zA-Z][a-zA-Z0-9]*)>", token)
        if m_open:
            tag = m_open.group(1).lower()
            active_tags[tag] = active_tags.get(tag, 0) + 1
            continue

        # Closing tag
        m_close = re.fullmatch(r"</([a-zA-Z][a-zA-Z0-9]*)>", token)
        if m_close:
            tag = m_close.group(1).lower()
            if tag in active_tags:
                active_tags[tag] -= 1
                if active_tags[tag] <= 0:
                    del active_tags[tag]
            continue

        # Plain text — split into words
        current_tags = frozenset(active_tags.keys())
        for word in token.split():
            if word:
                result.append((word, current_tags))

    return result
