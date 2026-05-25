"""Tests for per-entry sample folder wiring and validation."""

from __future__ import annotations

import shutil
from argparse import Namespace
from pathlib import Path

from dictextractor.extraction.sample_entry import (
    configure_sample_entry_args,
    preflight_validate_sample_entries,
    validate_alphabet_file,
    validate_configured_sample_entry,
    validate_ocr_hints_for_snippets,
)


def _make_entry(tmp_path: Path, *, alphabet: str = "a\nb", ocr_pages: list[str] | None = None) -> Path:
    entry = tmp_path / "Canala-English"
    snippets = entry / "snippets"
    snippets.mkdir(parents=True)
    (snippets / "page_12.pdf").write_bytes(b"%PDF-1.4")
    (entry / "mathpix").mkdir()
    (entry / "introduction").mkdir()
    (entry / "alphabet.txt").write_text(alphabet, encoding="utf-8")
    if ocr_pages is None:
        ocr_pages = ["page_12"]
    for stem in ocr_pages:
        (entry / "mathpix" / f"{stem}.txt").write_text("ocr hint", encoding="utf-8")
    return entry


def test_configure_sample_entry_args_discovers_alphabet_and_mathpix(
    tmp_path: Path,
) -> None:
    entry = _make_entry(tmp_path)

    args = Namespace(
        no_alphabet=False,
        no_ocr_hint=False,
        strategy="two_stage",
        stage="1",
    )
    snippets_dir, output_dir = configure_sample_entry_args(args, entry)

    assert snippets_dir == entry / "snippets"
    assert output_dir == entry / "outputs"
    assert args.alphabet == str(entry / "alphabet.txt")
    assert args.ocr_text == str(entry / "mathpix")
    assert args.intro == str(entry / "introduction")
    assert args.input_image == str(entry / "snippets")
    assert args.output == str(entry / "outputs")
    assert args.entry_dir == str(entry)


def test_configure_sample_entry_args_respects_ablation_flags(tmp_path: Path) -> None:
    entry = _make_entry(tmp_path)

    args = Namespace(
        no_alphabet=True,
        no_ocr_hint=True,
        no_intro=True,
        strategy="vlm_ocr",
        stage="1",
    )
    configure_sample_entry_args(args, entry)

    assert args.alphabet is None
    assert args.ocr_text is None
    assert args.intro is None


def test_configure_sample_entry_args_no_intro_only(tmp_path: Path) -> None:
    entry = _make_entry(tmp_path)
    args = Namespace(
        no_alphabet=False,
        no_ocr_hint=False,
        no_intro=True,
        strategy="two_stage",
        stage="2",
    )
    configure_sample_entry_args(args, entry)
    assert args.intro is None


def test_validate_alphabet_file_rejects_missing_and_empty(tmp_path: Path) -> None:
    missing = tmp_path / "alphabet.txt"
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")

    assert validate_alphabet_file(missing) == [f"alphabet file not found: {missing}"]
    assert validate_alphabet_file(empty) == [f"alphabet file is empty: {empty}"]


def test_validate_ocr_hints_for_snippets_requires_each_page(tmp_path: Path) -> None:
    entry = _make_entry(tmp_path, ocr_pages=[])
    snippets_dir = entry / "snippets"
    ocr_dir = entry / "mathpix"

    errors = validate_ocr_hints_for_snippets(ocr_dir, snippets_dir)
    assert any("page_12" in err for err in errors)


def test_preflight_aborts_alpha_ocr_when_alphabet_missing(tmp_path: Path) -> None:
    entry = tmp_path / "Canala-English"
    (entry / "snippets").mkdir(parents=True)
    (entry / "snippets" / "page_12.pdf").write_bytes(b"%PDF-1.4")
    (entry / "mathpix").mkdir()
    (entry / "mathpix" / "page_12.txt").write_text("hint", encoding="utf-8")

    args = Namespace(
        no_alphabet=False,
        no_ocr_hint=False,
        strategy="two_stage",
        stage="1",
        experiment_name="test_alpha_ocr",
    )
    failures = preflight_validate_sample_entries(args, [entry])
    assert failures
    assert failures[0][0] == "Canala-English"
    assert any("alphabet" in err for err in failures[0][1])


def test_validate_ocr_hints_for_snippets_accepts_table_docx(tmp_path: Path) -> None:
    source_docx = (
        Path(__file__).resolve().parents[2]
        / "assets/dictionaries/samples/Circassian-English-Turkish/mathpix/page_1.docx"
    )
    if not source_docx.is_file():
        return

    entry = tmp_path / "Circassian-English-Turkish"
    snippets = entry / "snippets"
    snippets.mkdir(parents=True)
    (snippets / "page_1.pdf").write_bytes(b"%PDF-1.4")
    mathpix = entry / "mathpix"
    mathpix.mkdir()
    shutil.copy2(source_docx, mathpix / "page_1.docx")

    errors = validate_ocr_hints_for_snippets(mathpix, snippets)
    assert errors == []


def test_validate_skipped_when_no_alphabet_flag(tmp_path: Path) -> None:
    entry = tmp_path / "Canala-English"
    (entry / "snippets").mkdir(parents=True)
    (entry / "snippets" / "page_12.pdf").write_bytes(b"%PDF-1.4")

    args = Namespace(
        no_alphabet=True,
        no_ocr_hint=True,
        alphabet=None,
        ocr_text=None,
        strategy="two_stage",
        stage="1",
    )
    errors = validate_configured_sample_entry(args, entry, entry / "snippets")
    assert errors == []
