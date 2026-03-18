"""
Canonical Pydantic schemas for structured dictionary entries.
All modules that produce or consume dictionary entries import from here.
"""

from typing import List
from pydantic import BaseModel, Field


class DictionaryEntry(BaseModel):
    """Schema for a dictionary entry. Not all dictionary entries have all fields."""

    headword: str = Field(
        ...,
        description="The headword word/phrase with all diacritical marks preserved",
    )
    entry_type: str = Field(
        default="word", description="Type of entry: 'word' or 'phrase'"
    )
    pos: str = Field(
        default="", description="Part of speech"
    )
    target_translation: str = Field(..., description="The target language translation/definition")
    literal_definition: str = Field(
        default="", description="Literal definition of the headword word/phrase"
    )
    grammar_notes: str = Field(
        default="",
        description="Grammatical forms, tense markers, or cross-references",
    )
    examples: List[str] = Field(
        default=[],
        description="Examples sentences with the headword word/phrase in use"
    )


class DictionaryPage(BaseModel):
    """Container for all extracted entries from a single page."""

    entries: List[DictionaryEntry]
    page_number: int
    source_file: str


# ---------------------------------------------------------------------------
# Structured output response schemas (used as response_format targets)
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
            "and special characters. Do not merge, skip, or paraphrase any line."
        )
    )


class TranscriptionResponse(BaseModel):
    """
    Structured output schema for Stage 1 transcription.

    Columns are ordered left → right. Within each column, lines are ordered
    top → bottom. This column-first ordering ensures that continuation lines
    of the same entry stay together, regardless of page layout.

    For single-column pages use one column with column_id='single'.
    """

    columns: List[ColumnTranscription] = Field(
        description=(
            "Columns detected on the page, ordered left to right. "
            "Transcribe each column fully (top to bottom) before moving to the next. "
            "Never mix lines from different columns in the same column entry."
        )
    )


class EntriesResponse(BaseModel):
    """
    Structured output schema for Stage 2 structuring.
    The LLM fills only fields that are actually present in each entry.
    """

    entries: List[DictionaryEntry]
