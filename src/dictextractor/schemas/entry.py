"""
Canonical Pydantic schemas for structured dictionary entries.
All modules that produce or consume dictionary entries import from here.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class DictionaryEntry(BaseModel):
    """Schema for a dictionary entry. Not all dictionary entries have all fields.

    A single visual entry block in the source page may contain multiple subwords
    or subentries (variants, derived forms, sense splits). Each subword must be
    emitted as its own DictionaryEntry — never collapse them into one.
    """

    headword: str = Field(
        ...,
        description="The headword word/phrase with all diacritical marks preserved",
    )
    pos: str = Field(
        default="", description="Part-of-speech tag/abbreviation if present, else ''"
    )
    meaning_description: str = Field(
        ...,
        description=(
            "The primary meaning/definition of the headword in the target language. "
            "Join near-synonymous glosses of one sense with '; '. Join distinct "
            "minor sub-meanings of the same sense with ' | '. Truly distinct senses "
            "that the dictionary treats as separate subentries must become separate "
            "DictionaryEntry items, not be joined here."
        ),
    )
    semantic_domain: str = Field(
        default="",
        description=(
            "A short domain/register label, ONLY when the dictionary explicitly "
            "marks one with a fixed convention (e.g. an italic abbreviation like "
            "'bot.', 'astr.', 'colloq.', 'arch.', or a dedicated symbol). "
            "Acceptable values are short tokens such as 'botany', 'astronomy', "
            "'colloquial', 'archaic'. If no such marker is present, or you are "
            "uncertain, return ''. Never write reasoning, hedging, or commentary "
            "in this field — it is either a short label or empty."
        ),
    )
    examples: List[str] = Field(
        default=[],
        description=(
            "Example sentences/phrases showing the headword in use. "
            "Always a list — one string per distinct example. "
            "Empty list if the entry has no examples."
        ),
    )
    extra_fields: Dict[str, str] = Field(
        default={},
        description=(
            "Optional discovery slot for any structurally-marked fields beyond "
            "the canonical schema (e.g. etymology, ipa, plural_form, gender, "
            "register, tone_class). Keys are snake_case English labels chosen "
            "by the model; values are the extracted text. Only populate when "
            "discovery mode is explicitly requested in the user prompt; "
            "otherwise leave as {}."
        ),
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
            "and special characters. Do not merge, skip, or paraphrase any line. "
            "Wrap bold text in <b>...</b> and italic text in <i>...</i> tags."
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
