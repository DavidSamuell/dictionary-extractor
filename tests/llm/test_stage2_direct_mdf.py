"""Tests for Pass 2 direct MDF message building."""

from pathlib import Path

from dictextractor.llm.stage2_direct_mdf import build_direct_mdf_messages
from dictextractor.schemas.field_cheatsheet import DictionaryMarkerCheatsheet, MarkerLine


def _minimal_cheatsheet() -> DictionaryMarkerCheatsheet:
    return DictionaryMarkerCheatsheet(
        dictionary_name="Chukchi-Russian",
        markers=[
            MarkerLine(marker="lx", description="headword"),
            MarkerLine(marker="gn", description="Russian gloss"),
        ],
    )


def test_build_messages_without_toolbox_pdf() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "page.png"
        png.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        messages = build_direct_mdf_messages(
            transcription="line",
            image_path=str(png),
            intro_image_paths=[],
            field_map=_minimal_cheatsheet(),
            model="gemini/gemini-3.1-pro-preview",
        )
    user_content = messages[1]["content"]
    text_part = next(p for p in user_content if p["type"] == "text")
    assert "Toolbox MDF Reference Manual" not in text_part["text"]
    assert "gold-label OCR" in messages[0]["content"]
    assert len(user_content) == 2  # text + page image only


def test_build_messages_with_toolbox_pdf() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "page.png"
        png.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        pdf = Path(tmp) / "manual.pdf"
        pdf.write_bytes(b"%PDF-1.4 minimal")
        messages = build_direct_mdf_messages(
            transcription="line",
            image_path=str(png),
            intro_image_paths=[],
            field_map=_minimal_cheatsheet(),
            model="gemini/gemini-3.1-pro-preview",
            toolbox_pdf=pdf,
        )
    user_content = messages[1]["content"]
    text_part = next(p for p in user_content if p["type"] == "text")
    assert "Toolbox MDF Reference Manual" in text_part["text"]
    assert len(user_content) == 3  # text + page image + pdf
    toolbox_part = user_content[-1]
    assert toolbox_part["image_url"]["url"].startswith("data:application/pdf;")


def test_build_messages_uses_toolbox_text_reference_for_gpt() -> None:
    import tempfile

    import pymupdf

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        png = tmp_path / "page.png"
        png.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        pdf = tmp_path / "manual.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "MDF marker reference")
        doc.save(str(pdf))
        doc.close()

        messages = build_direct_mdf_messages(
            transcription="line",
            image_path=str(png),
            intro_image_paths=[],
            field_map=_minimal_cheatsheet(),
            model="openrouter/openai/gpt-5.5",
            toolbox_pdf=pdf,
        )

        user_content = messages[1]["content"]
        text_part = next(p for p in user_content if p["type"] == "text")
        assert "text excerpt" in text_part["text"]
        assert "MDF marker vocabulary" in text_part["text"]
        assert len(user_content) == 2  # text + page image only
