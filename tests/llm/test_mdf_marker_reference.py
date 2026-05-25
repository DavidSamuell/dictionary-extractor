"""Tests for MDF marker reference used in Pass 1 field discovery."""

from dictextractor.llm.mdf_marker_reference import MDF_MARKER_REFERENCE


def test_reference_includes_core_dictionary_markers() -> None:
    for marker in ("lx", "gn", "ge", "sn", "hm", "se", "un", "de", "dn", "cf", "ps", "ph"):
        assert marker in MDF_MARKER_REFERENCE


def test_reference_omits_legacy_inflection_slots() -> None:
    assert "\\1s" not in MDF_MARKER_REFERENCE
    assert "\\2p" not in MDF_MARKER_REFERENCE


def test_reference_includes_toolbox_descriptions() -> None:
    assert "Record marker" in MDF_MARKER_REFERENCE
    assert "Reversal form (English)" in MDF_MARKER_REFERENCE
    assert "usage or restriction" in MDF_MARKER_REFERENCE.lower()
    assert "Fully expresses meaning" in MDF_MARKER_REFERENCE
    assert "polymorphemic form or phrase" in MDF_MARKER_REFERENCE


def test_reference_documents_language_tiers() -> None:
    assert "e=English, n=national, r=regional, v=vernacular" in MDF_MARKER_REFERENCE
    assert "Gloss (national)" in MDF_MARKER_REFERENCE
    assert "Definition (national)" in MDF_MARKER_REFERENCE
    assert "Usage (national)" in MDF_MARKER_REFERENCE


def test_field_discovery_does_not_accept_toolbox_pdf() -> None:
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "src/dictextractor/llm/field_discovery.py").read_text(
        encoding="utf-8"
    )
    fn_block = src.split("def discover_field_cheatsheet", 1)[1].split("\ndef ", 1)[0]
    assert "toolbox_pdf" not in fn_block
