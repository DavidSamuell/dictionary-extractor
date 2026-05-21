"""Frozen v1 typography normalization for eval-flat and OCR adapters.

Converts markdown/HTML/LaTeX wrappers to gold-style ``<b>`` / ``<i>`` tags.
Idempotent on already-normalized lines.
"""

from __future__ import annotations

import re

from dictextractor.evaluation.stage1.tag_parser import normalize_unicode

TYPOGRAPHY_SPEC_VERSION = "v1"

# Markdown bold/italic (non-greedy; processed bold before italic).
_MD_BOLD_STAR = re.compile(r"\*\*(.+?)\*\*")
_MD_BOLD_UNDER = re.compile(r"__(.+?)__")
_MD_ITALIC_STAR = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_MD_ITALIC_UNDER = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")

_HTML_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)>", re.IGNORECASE)
_BOLD_HTML = frozenset({"b", "strong"})
_ITALIC_HTML = frozenset({"i", "em"})

# LaTeX inline delimiters — keep inner text, drop markers.
_LATEX_INLINE = re.compile(
    r"\\\((.+?)\\\)|\$(?!\$)(.+?)(?<!\$)\$(?!\$)|\\\[(.+?)\\\]",
    re.DOTALL,
)

# Residual HTML-like tags after conversion (preserve dictionary <b>/<i>).
_RESIDUAL_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)>")
_ALLOWED_INLINE_TAGS = frozenset({"b", "i"})


def _strip_unknown_tags(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1).lower()
        if name in _ALLOWED_INLINE_TAGS:
            return match.group(0)
        return ""

    return _RESIDUAL_TAG_RE.sub(repl, text)


def _html_to_dictionary_tags(text: str) -> str:
    """Map HTML bold/italic tags to ``<b>`` / ``<i>``; strip other tags."""

    def repl(match: re.Match[str]) -> str:
        name = match.group(1).lower()
        closing = match.group(0).startswith("</")
        if name in _BOLD_HTML:
            return "</b>" if closing else "<b>"
        if name in _ITALIC_HTML:
            return "</i>" if closing else "<i>"
        return ""

    return _HTML_TAG_RE.sub(repl, text)


def _markdown_to_dictionary_tags(text: str) -> str:
    """Convert markdown emphasis to ``<b>`` / ``<i>``."""
    text = _MD_BOLD_STAR.sub(r"<b>\1</b>", text)
    text = _MD_BOLD_UNDER.sub(r"<b>\1</b>", text)
    text = _MD_ITALIC_STAR.sub(r"<i>\1</i>", text)
    text = _MD_ITALIC_UNDER.sub(r"<i>\1</i>", text)
    return text


def _strip_latex_delimiters(text: str) -> str:
    """Remove common LaTeX math delimiters while keeping content."""

    def repl(match: re.Match[str]) -> str:
        for group in match.groups():
            if group is not None:
                return group
        return ""

    return _LATEX_INLINE.sub(repl, text)


def normalize_line(line: str) -> str:
    """
    Normalize one line of OCR or transcript text to dictionary tag conventions.

    Order: LaTeX delimiters → markdown → HTML mapping → strip unknown tags → NFC.
    Internal whitespace is preserved (no collapse) so printed line breaks stay faithful.
    """
    if not line:
        return ""
    text = _strip_latex_delimiters(line)
    text = _markdown_to_dictionary_tags(text)
    text = _html_to_dictionary_tags(text)
    text = _strip_unknown_tags(text)
    # Remove empty bold/italic pairs left after stripping.
    text = re.sub(r"<b></b>|<i></i>", "", text)
    return normalize_unicode(text)


def normalize_lines(lines: list[str]) -> list[str]:
    """Apply :func:`normalize_line` to each line."""
    return [normalize_line(line) for line in lines]
