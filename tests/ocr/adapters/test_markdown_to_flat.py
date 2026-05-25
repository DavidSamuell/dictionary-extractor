"""Tests for OCR markdown → flat line normalization."""

from pathlib import Path

from dictextractor.ocr.adapters.markdown_to_flat import (
    find_markdown_source,
    markdown_text_to_flat_lines,
    strip_markdown_line_artifacts,
    unwrap_markdown_fences,
)


def test_unwrap_glm_markdown_fence() -> None:
    raw = "```markdown\nline one\nline two\n```"
    assert unwrap_markdown_fences(raw) == "line one\nline two"


def test_strip_mathpix_image_and_header() -> None:
    line = "## η"
    assert strip_markdown_line_artifacts(line) == "η"

    mixed = (
        "![](https://cdn.mathpix.com/cropped/x.jpg?height=59) "
        "PRONOUN Tone: LM Second person"
    )
    assert strip_markdown_line_artifacts(mixed) == "PRONOUN Tone: LM Second person"


def test_drop_empty_markdown_header_line() -> None:
    from dictextractor.ocr.adapters.markdown_to_flat import markdown_text_to_flat_lines

    assert markdown_text_to_flat_lines("###\nhello\n") == ["hello"]


def test_drop_hathitrust_line() -> None:
    from dictextractor.ocr.adapters.markdown_to_flat import markdown_text_to_flat_lines

    raw = (
        "Generated at University of Melbourne through HathiTrust on 2026-05-18\n"
        "real entry line\n"
    )
    assert markdown_text_to_flat_lines(raw) == ["real entry line"]


def test_strip_html_image_div_line_dropped() -> None:
    line = '<div style="text-align: center;"><img src="imgs/x.jpg" alt="Image" /></div>'
    assert strip_markdown_line_artifacts(line) is None


def test_markdown_bold_preserved_as_dictionary_tags() -> None:
    lines = markdown_text_to_flat_lines("**headword** and *gloss*")
    assert lines == ["<b>headword</b> and <i>gloss</i>"]


def test_paddle_list_indent_stripped() -> None:
    raw = "  -  $ \\eta_{w_{2}} / - / $ nee"
    prepared = strip_markdown_line_artifacts(raw)
    assert prepared is not None
    assert prepared.startswith("$")


def test_page_186_glm_sample_has_multiple_lines() -> None:
    glm_md = Path(
        "assets/dictionaries/samples/Na-English-Chinese-French/outputs/stage-1/"
        "GLM-OCR-flat_alpha/page_186/output.md"
    )
    if not glm_md.is_file():
        return
    text = glm_md.read_text(encoding="utf-8")
    lines = markdown_text_to_flat_lines(text)
    assert len(lines) >= 10
    assert any("VERB Tone" in ln for ln in lines)


def test_page_186_backend_samples_produce_dictionary_lines() -> None:
    """Smoke-test markdown normalization on real OCR outputs when samples exist."""
    samples = {
        "mathpix": (
            "assets/dictionaries/samples/Na-English-Chinese-French/outputs/stage-1/"
            "Mathpix-OCR/page_186/output.md"
        ),
        "glm": (
            "assets/dictionaries/samples/Na-English-Chinese-French/outputs/stage-1/"
            "GLM-OCR-flat_alpha/page_186/output.md"
        ),
        "mineru": (
            "assets/dictionaries/samples/Na-English-Chinese-French/outputs/stage-1/"
            "MinerU2.5-Pro/page_186/output.md"
        ),
        "paddle": (
            "assets/dictionaries/samples/Na-English-Chinese-French/outputs/stage-1/"
            "PaddleOCR-VL-1.5/page_186/page_186.md"
        ),
    }
    for backend, rel_path in samples.items():
        path = Path(rel_path)
        if not path.is_file():
            continue
        lines = markdown_text_to_flat_lines(path.read_text(encoding="utf-8"))
        assert len(lines) >= 5, f"{backend} produced too few lines"
        joined = "\n".join(lines)
        assert "Tone" in joined, f"{backend} missing dictionary content"
        assert "![](" not in joined, f"{backend} still has image markdown"
        assert "<img" not in joined.lower(), f"{backend} still has HTML images"


def test_find_markdown_source_prefers_output_md(tmp_path: Path) -> None:
    page_dir = tmp_path / "page_1"
    page_dir.mkdir()
    (page_dir / "page_1.md").write_text("from stem\n", encoding="utf-8")
    (page_dir / "output.md").write_text("from output\n", encoding="utf-8")
    assert find_markdown_source(page_dir, stem="page_1") == page_dir / "output.md"


def test_strip_unicode_bullet_prefix_and_separator() -> None:
    assert strip_markdown_line_artifacts("•mbiɛkə n") == "mbiɛkə n"
    assert strip_markdown_line_artifacts("wasp •fibiəkə n • kiŋkwɛiŋ n") == (
        "wasp fibiəkə n kiŋkwɛiŋ n"
    )


def test_split_concatenated_markdown_headers() -> None:
    raw = "### money ### capital, principal ### cashier, treasurer\n"
    lines = markdown_text_to_flat_lines(raw)
    assert lines == ["money", "capital, principal", "cashier, treasurer"]


def test_strip_display_math_blocks() -> None:
    raw = "olwiylwiy\n$$ left(***right) $$\nsound made by a bird\n"
    lines = markdown_text_to_flat_lines(raw)
    assert lines == ["olwiylwiy", "sound made by a bird"]
    assert "$$" not in "\n".join(lines)
