"""Tests for Mathpix snippet upload preparation."""

from pathlib import Path

import pytest

from dictextractor.ocr.mathpix_convert import MathpixConvertError, prepare_mathpix_upload


def test_prepare_mathpix_upload_passes_pdf_through(tmp_path: Path) -> None:
    pdf = tmp_path / "page_1.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert prepare_mathpix_upload(pdf, tmp_path / "cache") == pdf


def test_prepare_mathpix_upload_wraps_png(tmp_path: Path) -> None:
    png_src = (
        Path("assets/dictionaries/samples/Evenki-Russian/snippets/page_1.png")
    )
    if not png_src.is_file():
        pytest.skip("Evenki PNG sample missing")

    upload_pdf = prepare_mathpix_upload(png_src, tmp_path / "cache")
    assert upload_pdf.suffix == ".pdf"
    assert upload_pdf.is_file()
    assert upload_pdf.stat().st_size > 1000
    # Cached on second call
    assert prepare_mathpix_upload(png_src, tmp_path / "cache") == upload_pdf


def test_prepare_mathpix_upload_rejects_unknown_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "page_1.xyz"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(MathpixConvertError, match="Unsupported"):
        prepare_mathpix_upload(bad, tmp_path / "cache")
