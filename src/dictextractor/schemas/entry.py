"""
Canonical Pydantic schemas for structured dictionary entries.
All modules that produce or consume dictionary entries import from here.
"""

from typing import Dict, List, Literal
from pydantic import BaseModel, Field

EntryType = Literal["main", "subentry", "sense"]


class DictionaryEntry(BaseModel):
    """Structured dictionary entry aligned with SIL Toolbox / MDF export.

    Use entry_type and parent_lexeme to encode hierarchy (\\lx, \\se, \\sn).
    Populate target_glosses (\\ge, \\gn, …) and definition (\\de) separately when the source distinguishes them.
    """

    entry_type: EntryType = Field(
        default="main",
        description=(
            "MDF record role: 'main' = new \\lx headword block; 'subentry' = run-on "
            "derivative/compound under a parent (\\se); 'sense' = numbered sense under "
            "a lemma (\\sn). Every row must set this explicitly."
        ),
    )
    headword: str = Field(
        ...,
        description=(
            "Surface lemma for \\lx — the headword only, with all diacritics preserved. "
            "Do NOT include POS, commas, or trailing line punctuation (put POS in pos)."
        ),
    )
    parent_lexeme: str = Field(
        default="",
        description=(
            "Parent lemma headword when entry_type is 'subentry' or 'sense' (links \\se/\\sn "
            "to the main \\lx). Must be empty when entry_type is 'main'."
        ),
    )
    sense_number: str = Field(
        default="",
        description=(
            "Sense index for \\sn (e.g. '1', '2', 'I') when entry_type is 'sense'; "
            "otherwise empty."
        ),
    )
    homonym_number: str = Field(
        default="",
        description=(
            "Homonym discriminator for \\hm (e.g. '1', '2') when the dictionary marks "
            "homographs; otherwise empty."
        ),
    )
    pos: str = Field(
        default="",
        description=(
            "Part-of-speech tag for \\ps — abbreviation exactly as printed (e.g. n., "
            "сущ., nn.); empty if not shown."
        ),
    )
    target_glosses: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Short glosses per target language — keys must match dictionary_languages.yaml "
            "(e.g. en, zh, tr). Maps to MDF \\ge, \\gn, etc. Leave empty {} when none. "
            "Do not duplicate into gloss."
        ),
    )
    gloss: str = Field(
        default="",
        description=(
            "Legacy single gloss field — leave empty. Use target_glosses[code] from "
            "the dictionary language config instead."
        ),
    )
    definition: str = Field(
        default="",
        description=(
            "Longer definitional text for \\de — explanatory wording beyond a short gloss; "
            "join minor sub-meanings of the same sense with ' | '. Empty if none."
        ),
    )
    semantic_domain: str = Field(
        default="",
        description=(
            "Semantic/register label for \\sd — short token only when explicitly marked "
            "(e.g. bot., colloq., archaic); never commentary or reasoning; else ''."
        ),
    )
    citation_form: str = Field(
        default="",
        description=(
            "Lexical citation form for \\lc when the printed headword differs from headword "
            "(e.g. bound roots); otherwise empty."
        ),
    )
    phonetic: str = Field(
        default="",
        description=(
            "Phonetic pronunciation for \\ph when marked on the entry; otherwise empty."
        ),
    )
    cross_references: List[str] = Field(
        default=[],
        description=(
            "Cross-reference target lemmas for \\cf — headword strings only, no 'see' or "
            "'cf.' prose; empty list if none."
        ),
    )
    examples: List[str] = Field(
        default=[],
        description=(
            "Example phrases/sentences for \\xv — one string per example in order; "
            "vernacular/source-language text when bilingual."
        ),
    )
    example_glosses: List[str] = Field(
        default=[],
        description=(
            "Translations of examples for \\xe — parallel to examples (same length when "
            "each example has a translation); empty list if monolingual examples only."
        ),
    )
    extra_fields: Dict[str, str] = Field(
        default={},
        description=(
            "Non-MDF-standard structurally marked fields only (etymology, gender, register, "
            "dialect, inflection tables). Use frozen allowlist keys when discovery mode is on; "
            "never duplicate phonetic, cross_references, target_glosses, or definition here; else {}."
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


class FlatTranscriptionResponse(BaseModel):
    """
    Structured output for flat Stage 1 transcription (eval-flat / PageTranscript).

    No column_id or line_number — body lines are in global reading order.
    """

    header: List[str] = Field(
        default=[],
        description=(
            "Page-level header lines above the dictionary body (running title, "
            "page number, letter band). One string per visible line. Empty if none."
        ),
    )
    lines: List[str] = Field(
        description=(
            "Every visible body line in reading order (top to bottom). For "
            "multi-column pages: complete the left column top-to-bottom, then "
            "the next column, etc. Wrap bold in <b>...</b> and italic in <i>...</i>. "
            "Preserve hyphenated line breaks as separate lines with trailing hyphen."
        )
    )
    footer: List[str] = Field(
        default=[],
        description=(
            "Page-level footer lines below the body. One string per visible line. "
            "Empty if none."
        ),
    )


class TranscriptionResponse(BaseModel):
    """
    Structured output schema for Stage 1 transcription.

    Page-level metadata (header, footer) is captured in dedicated fields,
    separate from the body content. Body content is split into columns,
    ordered left → right. Within each column, lines are ordered top → bottom.

    For single-column pages use one column with column_id='single'.
    """

    header: List[str] = Field(
        default=[],
        description=(
            "Page-level header text appearing ABOVE the body columns — e.g. "
            "running title, page number, chapter abbreviation, alphabetic letter "
            "band. One string per visible line (top to bottom). Headers may sit "
            "anywhere horizontally (centred, spanning columns); they are NEVER "
            "part of a column. Empty list if the page has no header. "
            "Do NOT include the first dictionary entry here."
        ),
    )
    columns: List[ColumnTranscription] = Field(
        description=(
            "Body columns detected on the page, ordered left to right. "
            "Transcribe each column fully (top to bottom) before moving to the next. "
            "Never mix lines from different columns in the same column entry. "
            "Do NOT include header or footer text inside any column."
        )
    )
    footer: List[str] = Field(
        default=[],
        description=(
            "Page-level footer text appearing BELOW the body columns — e.g. "
            "page number, footnote, decorative rule, copyright line. One string "
            "per visible line (top to bottom). Footers may sit anywhere "
            "horizontally; they are NEVER part of a column. Empty list if the "
            "page has no footer."
        ),
    )


class EntriesResponse(BaseModel):
    """
    Structured output schema for Stage 2 structuring.
    The LLM fills only fields that are actually present in each entry.
    """

    entries: List[DictionaryEntry]
