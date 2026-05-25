"""Tests for compact marker cheat sheet schema."""

from dictextractor.schemas.field_cheatsheet import DictionaryMarkerCheatsheet, MarkerLine


def test_format_prompt_block() -> None:
    sheet = DictionaryMarkerCheatsheet(
        dictionary_name="Test Dict",
        markers=[
            MarkerLine(marker="lx", description="headword"),
            MarkerLine(marker="gn", description="Russian gloss"),
        ],
        rules=["One \\lx per main entry."],
        abbreviations={"мн.": "plural"},
    )
    block = sheet.format_prompt_block()
    assert "\\lx   headword" in block
    assert "\\gn   Russian gloss" in block
    assert "Rules:" in block
    assert "мн. → plural" in block
    assert "<field_profile>" not in block
