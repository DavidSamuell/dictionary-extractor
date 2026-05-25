"""Tests for MDF grouping, export, and validation."""

from __future__ import annotations

from dictextractor.schemas.dictionary_languages import (
    DictionaryLanguagesConfig,
    SourceLanguageConfig,
    TargetLanguageConfig,
)
from dictextractor.schemas.entry import DictionaryEntry
from dictextractor.utils.mdf_export import (
    block_ids_for_entries,
    entries_to_mdf_text,
    group_entries_to_toolbox_records,
    normalize_mdf_text,
    normalize_stage2_entries,
    strip_end_of_sentence_punctuation,
    toolbox_to_mdf_text,
    validate_stage2_entries,
)

SYNTHETIC_GOLD_MDF = """\
\\lx alpha
\\gn first gloss

\\lx beta
\\hm 1
\\gn second gloss

\\lx gamma
\\gn third gloss
"""


def _chukchi_config() -> DictionaryLanguagesConfig:
    return DictionaryLanguagesConfig(
        layout="bilingual",
        source=SourceLanguageConfig(language="Chukchi", code="chukchi"),
        targets=[
            TargetLanguageConfig(language="Russian", code="ru"),
        ],
    )


def _sample_block_with_senses() -> list[DictionaryEntry]:
    return [
        DictionaryEntry(
            entry_type="main",
            headword="аройвыԓьын",
        ),
        DictionaryEntry(
            entry_type="sense",
            headword="аройвыԓьын",
            parent_lexeme="аройвыԓьын",
            sense_number=1,
            gloss="мо́щный",
        ),
        DictionaryEntry(
            entry_type="sense",
            headword="аройвыԓьын",
            parent_lexeme="аройвыԓьын",
            sense_number=2,
            gloss="богаты́рь",
        ),
    ]


def test_normalize_entry_number_from_string_label() -> None:
    entry = DictionaryEntry(
        entry_type="sense",
        headword="lemma",
        parent_lexeme="lemma",
        sense_number="2)",
        gloss="gloss",
    )
    assert entry.sense_number == 2


def test_normalize_roman_homonym_number() -> None:
    entry = DictionaryEntry(
        entry_type="main",
        headword="ачыӈык",
        homonym_number="II",
        gloss="просить",
    )
    assert entry.homonym_number == 2


def test_block_ids_group_subentry_with_main() -> None:
    rows = [
        DictionaryEntry(entry_type="main", headword="аркычеты", gloss="a"),
        DictionaryEntry(
            entry_type="subentry",
            headword="аркычеты ваԓьын",
            parent_lexeme="аркычеты",
            gloss="b",
        ),
    ]
    ids = block_ids_for_entries(rows)
    assert ids[0] == "аркычеты:0"
    assert ids[1] == ids[0]


def test_group_entries_to_toolbox_records() -> None:
    records = group_entries_to_toolbox_records(_sample_block_with_senses())
    assert len(records) == 1
    assert records[0].main.headword == "аройвыԓьын"
    assert len(records[0].children) == 2


def test_toolbox_to_mdf_text_blank_line_records() -> None:
    rows = _sample_block_with_senses() + [
        DictionaryEntry(
            entry_type="main",
            headword="аплёратвака",
            gloss="gloss",
        )
    ]
    text = entries_to_mdf_text(rows, _chukchi_config())
    assert "\\lx аройвыԓьын" in text
    assert "\\sn 1" in text
    assert "\\gn мо́щный" in text
    assert "\n\n" in text


def test_validator_warns_gloss_in_usage_note_only() -> None:
    entry = DictionaryEntry(
        entry_type="main",
        headword="аплёратвака",
        usage_note="че́рез не́которое вре́мя",
    )
    report = validate_stage2_entries([entry], _chukchi_config())
    assert any("gloss is empty" in i.message for i in report.warnings)


def test_validator_warns_empty_gloss() -> None:
    entry = DictionaryEntry(entry_type="main", headword="lemma")
    report = validate_stage2_entries([entry], _chukchi_config())
    assert any("gloss is empty" in i.message for i in report.warnings)


def _normalise_mdf(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for block in text.strip().split("\n\n"):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if lines:
            blocks.append(lines)
    return blocks


def test_gold_fixture_structure() -> None:
    """MDF has one record per headword/homograph block."""
    gold = _normalise_mdf(SYNTHETIC_GOLD_MDF)
    assert len(gold) == 3
    homograph_blocks = [b for b in gold if any(l.startswith("\\hm ") for l in b)]
    assert len(homograph_blocks) == 1


def test_semicolon_gloss_splits_to_multiple_ge_lines() -> None:
    rows = [
        DictionaryEntry(
            entry_type="main",
            headword="армаԓтаттыргын",
            gloss="побе́да; превосхо́дство",
        )
    ]
    text = entries_to_mdf_text(rows, _chukchi_config())
    assert "\\gn побе́да" in text
    assert "\\gn превосхо́дство" in text


def test_usage_note_emits_de_marker() -> None:
    rows = [
        DictionaryEntry(
            entry_type="main",
            headword="аркычыткок",
            gloss="кача́ться",
            usage_note="о ло́дке, самолёте",
        )
    ]
    text = entries_to_mdf_text(rows, _chukchi_config())
    assert "\\gn кача́ться" in text
    assert "\\de о ло́дке, самолёте" in text


def test_gold_sample_round_trip_subset() -> None:
    """Subset of gold entries round-trip through grouping/export."""
    entries = _sample_block_with_senses()
    produced = entries_to_mdf_text(entries, _chukchi_config()).strip()
    expected = (
        "\\lx аройвыԓьын\n\\sn 1\n\\gn мо́щный\n\\sn 2\n\\gn богаты́рь"
    )
    assert produced == expected


def test_strip_end_of_sentence_punctuation() -> None:
    assert strip_end_of_sentence_punctuation("Let's go out!") == "Let's go out"
    assert strip_end_of_sentence_punctuation("Really?") == "Really"
    assert strip_end_of_sentence_punctuation("Done.") == "Done"
    assert strip_end_of_sentence_punctuation("What?!") == "What"
    assert strip_end_of_sentence_punctuation("ellipsis…") == "ellipsis…"
    assert strip_end_of_sentence_punctuation("no change") == "no change"


def test_normalize_mdf_text_strips_trailing_punctuation() -> None:
    raw = (
        "\\lx lemma\n"
        "\\gn Goose.\n"
        "\\xe Let's go out!\n"
        "\\xr On sort !\n"
        "\\sn 1.\n"
    )
    normalized = normalize_mdf_text(raw)
    assert "\\gn Goose" in normalized
    assert "\\xe Let's go out" in normalized
    assert "\\xr On sort" in normalized
    assert "\\sn 1" in normalized


def test_entries_to_mdf_text_strips_gloss_punctuation() -> None:
    rows = [
        DictionaryEntry(
            entry_type="main",
            headword="lemma",
            gloss="Goose.",
            examples=["Let's go out!"],
            example_glosses=["Let us leave."],
        )
    ]
    text = entries_to_mdf_text(rows, _chukchi_config())
    assert "\\gn Goose\n" in text
    assert "\\xv Let's go out\n" in text
    assert "\\xe Let us leave" in text
