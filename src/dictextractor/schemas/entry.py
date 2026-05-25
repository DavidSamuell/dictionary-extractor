"""
Canonical Pydantic schemas for structured dictionary entries.

Stage 2 uses a deliberately small ``DictionaryEntry`` surface so the LLM fills
plain string fields. MDF export adds block grouping and language codes downstream.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from dictextractor.schemas.entry_numbers import normalize_entry_number

EntryType = Literal["main", "subentry", "sense"]


class DictionaryEntry(BaseModel):
    """One row in the Stage 2 JSON array.

    Hierarchy:
      main — new bold headword (or homograph main)
      subentry — bold run-on form under a parent lemma
      sense — numbered meaning under one lemma

    ``block_id`` is assigned in post-processing, not by the LLM.
    """

    entry_type: EntryType = Field(
        default="main",
        description=(
            "'main' = headword block; 'subentry' = run-on compound under parent_lexeme; "
            "'sense' = numbered sense under parent_lexeme."
        ),
    )
    headword: str = Field(
        ...,
        description=(
            "Chukchi/source lemma only — bold text, no homograph index, no POS, "
            "no trailing punctuation."
        ),
    )
    parent_lexeme: str = Field(
        default="",
        description=(
            "Parent headword when entry_type is 'subentry' or 'sense'; empty for 'main'."
        ),
    )
    homonym_number: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Homograph index (1, 2, 3, …) on entry_type='main' only. "
            "Convert Roman numerals (I, II) to integers. Null if not a homograph."
        ),
    )
    sense_number: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Sense index (1, 2, 3, …) on entry_type='sense' only. "
            "Strip trailing ')' or '.' from printed labels. Null otherwise."
        ),
    )
    gloss: str = Field(
        default="",
        description=(
            "Primary target-language translation — all non-italic wording after the "
            "headword. Semicolon-separated synonyms allowed in one string."
        ),
    )
    gloss_secondary: str = Field(
        default="",
        description=(
            "Second target-language translation when the dictionary has two targets "
            "(see <dictionary_languages>); empty for single-target dictionaries."
        ),
    )
    usage_note: str = Field(
        default="",
        description=(
            "Italic or parenthetical domain/usage expansion only — not the main translation."
        ),
    )
    pos: str = Field(
        default="",
        description="Part-of-speech abbreviation exactly as printed; empty if not shown.",
    )
    phonetic: str = Field(
        default="",
        description="Phonetic pronunciation when marked; otherwise empty.",
    )
    cross_references: List[str] = Field(
        default_factory=list,
        description="Cross-reference headwords only — no 'see' / 'cf.' prose.",
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Example phrases in source language — one string per example.",
    )
    example_glosses: List[str] = Field(
        default_factory=list,
        description="Example translations parallel to examples; empty if monolingual.",
    )
    extra_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional marked fields (gender, dialect, …) when discovery is on.",
    )

    @field_validator("homonym_number", "sense_number", mode="before")
    @classmethod
    def _normalize_number_fields(cls, value: object) -> Optional[int]:
        """Accept integers, Arabic strings, Roman labels; normalise to int or null."""
        return normalize_entry_number(value)

    @field_validator("homonym_number")
    @classmethod
    def _homonym_only_on_main(cls, value: Optional[int], info) -> Optional[int]:
        if info.data.get("entry_type") != "main":
            return None
        return value

    @field_validator("sense_number")
    @classmethod
    def _sense_only_on_sense(cls, value: Optional[int], info) -> Optional[int]:
        if info.data.get("entry_type") != "sense":
            return None
        return value

    @field_validator("parent_lexeme")
    @classmethod
    def _parent_only_on_child(cls, value: str, info) -> str:
        if info.data.get("entry_type") == "main":
            return ""
        return value


class DictionaryPage(BaseModel):
    """Container for all extracted entries from a single page."""

    entries: List[DictionaryEntry]
    page_number: int
    source_file: str
    mdf_text: str = Field(
        default="",
        description="Direct MDF output when stage2_mode=direct_mdf.",
    )


# ---------------------------------------------------------------------------
# Stage 1 structured output (unchanged)
# ---------------------------------------------------------------------------

class ColumnTranscription(BaseModel):
    """Lines from a single detected column, read top-to-bottom."""

    column_id: str = Field(
        description=(
            "Column identifier. Use 'left', 'center', 'right' for multi-column pages, "
            "or 'single' when the page has only one column."
        )
    )
    lines: List[str] = Field(
        description=(
            "Every visible line of text in this column exactly as it appears, "
            "one string per line, top to bottom. Preserve all diacritics, stress marks, "
            "and special characters. Do not merge, skip, or paraphrase any line. "
            "Wrap bold text in <b>...</b> and italic text in <i>...</i> tags."
        )
    )


class FlatTranscriptionResponse(BaseModel):
    """Structured output for flat Stage 1 transcription."""

    header: List[str] = Field(default=[])
    lines: List[str] = Field(
        description=(
            "Every visible body line in reading order. Wrap bold in <b>...</b> and "
            "italic in <i>...</i>."
        )
    )
    footer: List[str] = Field(default=[])


class TranscriptionResponse(BaseModel):
    """Structured output schema for Stage 1 column transcription."""

    header: List[str] = Field(default=[])
    columns: List[ColumnTranscription] = Field(
        description="Body columns left → right; transcribe each fully top → bottom."
    )
    footer: List[str] = Field(default=[])


class EntriesResponse(BaseModel):
    """Stage 2 structured output — array of DictionaryEntry rows."""

    entries: List[DictionaryEntry]
