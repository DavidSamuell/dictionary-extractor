"""Tests for PDF rasterization helpers."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from dictextractor.utils.pdf_render import needs_pdf_rasterization, render_pdf_pages


def test_needs_pdf_rasterization_by_model() -> None:
    assert needs_pdf_rasterization("openrouter/openai/gpt-5.5") is True
    assert needs_pdf_rasterization("gemini/gemini-3.1-pro-preview") is False
    assert needs_pdf_rasterization("any-model", preprocess=True) is True


def test_render_pdf_pages_caches_png(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()

    cache_dir = tmp_path / "cache"
    first = render_pdf_pages(pdf, cache_dir)
    second = render_pdf_pages(pdf, cache_dir)

    assert len(first) == 1
    assert first[0].suffix == ".png"
    assert first == second
    assert first[0].stat().st_mtime >= pdf.stat().st_mtime
