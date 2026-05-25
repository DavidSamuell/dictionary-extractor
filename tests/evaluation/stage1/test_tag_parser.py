"""Tests for stage-1 text normalisation helpers."""

from __future__ import annotations

from dictextractor.evaluation.stage1.tag_parser import (
    normalize_line_text,
    strip_tags,
)


def test_normalize_line_text_removes_space_before_punctuation() -> None:
    assert normalize_line_text("hello , world ; ok") == "hello, world; ok"
    assert normalize_line_text('word . "quote"') == 'word."quote"'


def test_clean_text_path_strips_tags_and_normalizes_punctuation() -> None:
    assert normalize_line_text(strip_tags("<b>hello ,</b>")) == "hello,"


def test_normalize_line_text_collapses_whitespace() -> None:
    assert normalize_line_text("a   b") == "a b"
    assert normalize_line_text("a  \t  b") == "a b"


def test_alignment_clean_text_collapses_whitespace() -> None:
    from dictextractor.evaluation.stage1.alignment import clean_text

    assert clean_text("<b>hello</b>   world") == "hello world"
