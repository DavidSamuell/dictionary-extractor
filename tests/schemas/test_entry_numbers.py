"""Tests for homonym_number and sense_number normalization on DictionaryEntry."""

from dictextractor.schemas.entry import DictionaryEntry


def test_homonym_number_integer_on_main() -> None:
    entry = DictionaryEntry(
        entry_type="main",
        headword="ачыӈык",
        homonym_number=2,
        gloss="просить",
    )
    assert entry.homonym_number == 2


def test_homonym_number_null_by_default() -> None:
    entry = DictionaryEntry(entry_type="main", headword="lemma", gloss="gloss")
    assert entry.homonym_number is None


def test_homonym_number_empty_string_becomes_null() -> None:
    entry = DictionaryEntry(
        entry_type="main",
        headword="lemma",
        homonym_number="",
        gloss="gloss",
    )
    assert entry.homonym_number is None


def test_homonym_number_normalizes_roman_string() -> None:
    entry = DictionaryEntry(
        entry_type="main",
        headword="ачыӈык",
        homonym_number="II",
        gloss="gloss",
    )
    assert entry.homonym_number == 2


def test_sense_number_strips_trailing_punctuation() -> None:
    entry = DictionaryEntry(
        entry_type="sense",
        headword="lemma",
        parent_lexeme="lemma",
        sense_number="1)",
        gloss="gloss",
    )
    assert entry.sense_number == 1


def test_homonym_number_cleared_on_sense_row() -> None:
    entry = DictionaryEntry(
        entry_type="sense",
        headword="lemma",
        parent_lexeme="lemma",
        sense_number=1,
        homonym_number=1,
        gloss="gloss",
    )
    assert entry.homonym_number is None
