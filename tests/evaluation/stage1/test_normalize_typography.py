"""Tests for typography normalization v1."""

from dictextractor.evaluation.stage1.normalize_typography import (
    TYPOGRAPHY_SPEC_VERSION,
    normalize_line,
)


def test_spec_version() -> None:
    assert TYPOGRAPHY_SPEC_VERSION == "v1"


def test_markdown_bold_italic() -> None:
    assert normalize_line("**head** and *tail*") == "<b>head</b> and <i>tail</i>"


def test_html_mapping() -> None:
    assert normalize_line("<strong>x</strong> <em>y</em>") == "<b>x</b> <i>y</i>"


def test_idempotent_on_dictionary_tags() -> None:
    line = "<b>ээттик</b> 1) на́рта"
    assert normalize_line(line) == line


def test_strip_unknown_html() -> None:
    assert normalize_line("<table><tr><td>cell</td></tr></table>") == "cell"


def test_latex_inline() -> None:
    out = normalize_line(r"\(\lambda\)")
    assert "lambda" in out
    assert r"\(" not in out
