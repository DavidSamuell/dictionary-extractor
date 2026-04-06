"""
Utilities for parsing inline HTML tags (<b>, <i>, etc.) in dictionary text.

Provides:
- strip_tags()          : remove all HTML tags, return plain text
- parse_tagged_words()  : split text into (word, frozenset_of_tags) tuples
- normalize_unicode()   : NFC-normalise for consistent comparison
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
