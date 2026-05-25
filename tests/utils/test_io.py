"""Tests for file I/O helpers."""

from __future__ import annotations

from pathlib import Path

from dictextractor.utils.io import read_docx_text

_CIRCASSIAN_PAGE_1 = (
    Path(__file__).resolve().parents[2]
    / "assets/dictionaries/samples/Circassian-English-Turkish/mathpix/page_1.docx"
)


def test_read_docx_text_extracts_table_only_mathpix_docx() -> None:
    if not _CIRCASSIAN_PAGE_1.is_file():
        return

    text = read_docx_text(str(_CIRCASSIAN_PAGE_1))

    assert text.strip()
    assert "English" in text
    assert "Circassian" in text
    assert "Able" in text
