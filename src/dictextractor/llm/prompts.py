"""
Prompt templates for each extraction strategy.
Keeping prompts here separates linguistic engineering from pipeline logic.
"""

from dictextractor.schemas.entry import DictionaryEntry

# ---------------------------------------------------------------------------
# Manual extraction strategy prompts (current hand-tuned approach)
# ---------------------------------------------------------------------------

MANUAL_SYSTEM_PROMPT = """\
You are a linguistic expert digitizing a Chukchi-Russian dictionary.
Your task is to extract entries into a structured JSON format for high-accuracy TSV conversion.

### EXTRACTION LOGIC & MAPPING RULES:
1. **Headword/Phrase**:
   - **Words**: The bolded Chukchi term at the start. Preserve all diacritics and stress marks.
   - **Phrases**: Identified by the absence of an immediate POS tag in parentheses. Often separated from the Russian translation by a dash (—).
2. **POS (Part of Speech)**:
   - Identify abbreviations in parentheses: (сущ.) for noun, (гл.) for verb, (прил.) for adjective, (нар.) for adverb, etc.
3. **Literal_Meaning**:
   - Trigger: The abbreviation "букв." (буквально).
   - Rule: Extract the literal translation immediately following this marker.
4. **Translation (RU)**:
   - The primary Russian definition. If there are multiple meanings, separate them with a semicolon (;).
5. **Grammar/Notes**:
   - Capture cross-references marked by "см." (смотри).
   - Capture specific Chukchi grammatical inflections, tense markers (e.g., "прош. II") or plural forms.

### CRITICAL FORMATTING RULES:
- **Encoding**: Preserve ALL phonetic symbols (e.g., ŋ, æ, ʌ, ь, ə) exactly.
- **Cleanup**: Remove "букв." prefixes in the final JSON; provide content only.
- **Accuracy**: Prioritise the visual image over OCR text — standard OCR may miss or misinterpret Chukchi phonetic characters.
"""


def manual_user_prompt(ocr_text: str) -> str:
    """Build the user-turn prompt for the manual extraction strategy."""
    return f"""\
<extracted_text>{ocr_text}</extracted_text>

Using the OCR text as a reference for character shapes, extract all Chukchi entries from the image into a JSON array.
Map the data to these specific attributes:
- "headword_phrase": The Chukchi word or sentence.
- "pos": The grammatical category (e.g., 'сущ.', 'гл.').
- "translation_ru": The Russian translation/definition.
- "literal_meaning": The literal translation (found after 'букв.').
- "grammar_notes": Any grammatical forms, tense markers, or cross-references ('см.').

Return ONLY a valid JSON array.
[
{{
    "headword_phrase": "ac-úkwʌn",
    "pos": "сущ.",
    "translation_ru": "кремень",
    "literal_meaning": "жирный камень",
    "grammar_notes": "см. æc/ac"
}}]"""


# ---------------------------------------------------------------------------
# Join method prompts
# ---------------------------------------------------------------------------

JOIN_STRUCTURE_SYSTEM_PROMPT = """\
You are a linguistic expert analysing a Chukchi-Russian dictionary page.
Your task is to describe the layout and structural conventions of the entries you see.
Focus on: headword formatting, POS tag patterns, how translations are separated,
how literal meanings are marked, and any multi-column layout.
Return a concise prose description (2-4 paragraphs).
"""

JOIN_STRUCTURE_USER_PROMPT = """\
Please examine this dictionary page and describe its entry structure in detail.
Include: how headwords are formatted (bold, italic, etc.), POS tag abbreviations used,
separator characters between headword and translation, and any special markers.
"""


def join_extraction_system_prompt(structure_description: str) -> str:
    """Build the system prompt for the join extraction step."""
    return f"""\
You are a linguistic expert digitizing a Chukchi-Russian dictionary.
The following describes the structural conventions of this specific dictionary:

{structure_description}

Use this structure description to accurately extract all entries into a JSON array.
Preserve all diacritics and phonetic characters exactly.
Return ONLY a valid JSON array matching the schema below.
"""


def join_extraction_user_prompt(ocr_text: str, alphabet_hint: str = "") -> str:
    """Build the user-turn prompt for the join extraction step."""
    alphabet_section = f"\n<alphabet>{alphabet_hint}</alphabet>" if alphabet_hint else ""
    return f"""\
<extracted_text>{ocr_text}</extracted_text>{alphabet_section}

Extract all Chukchi dictionary entries from the image into a JSON array with these fields:
"headword_phrase", "pos", "translation_ru", "literal_meaning", "grammar_notes".

Return ONLY a valid JSON array.
"""


# ---------------------------------------------------------------------------
# Two-stage method prompts
# ---------------------------------------------------------------------------

# ── Stage 1: Transcription ────────────────────────────────────────────────
# System = fixed role + rules.  User = dynamic inputs (alphabet, OCR hint, image).

STAGE_1_SYSTEM = """\
You are a precise OCR transcription system specialising in historical and minority-language dictionaries.

Step 1 — Detect page-level header and footer (if present).
  Many dictionary pages have a running title, page number, chapter abbreviation,
  or alphabetic letter band at the very top, and/or page numbers, footnotes, or
  decorative rules at the very bottom. These are NOT dictionary entries.
    - Top metadata → emit one string per visible line into the `header` list.
    - Bottom metadata → emit one string per visible line into the `footer` list.
    - If a region has no metadata, leave its list empty.
  Headers and footers can sit ANYWHERE horizontally (centred, spanning columns,
  or aligned to one side). Do NOT force them into a column — always treat them
  as page-level. The first dictionary entry of the page belongs in a column,
  not in `header`.

Step 2 — Detect body columns.
  After excluding header/footer regions, examine the body layout. Most dictionary
  pages use one or two columns; some use three. Identify each column region
  (left, center, right) before transcribing.

Step 3 — Transcribe each column separately, left to right.
  For every column, list every visible line top to bottom, exactly as it appears.
  Never mix a line from one column into another column's list. Do NOT include
  header or footer text inside any column.

You may or may not given a ocr reference text wrap around by <ocr_reference>...</ocr_reference>, it comes from a standard OCR result of the page. This text is a reference for the character shapes and should be used to help you transcribe the page. However, prioritise the visual image over OCR text — standard OCR may miss or misinterpret certain phonetic characters.
Rules that apply to every line (header, footer, and column lines alike):
- Preserve ALL diacritics, stress marks, and special phonetic symbols exactly.
- Preserve visual formatting: wrap bold text in <b>...</b> and italic text in <i>...</i>.
  Only mark formatting you are confident about — when in doubt, leave text plain.
  These tags may appear anywhere within a line (e.g. "<b>headword</b> translation").
- Do NOT interpret, restructure, summarise, or reorder content.
- Do NOT skip any line, even if it looks like a sub-entry, continuation, or cross-reference.
- Do NOT correct apparent typos or inconsistencies.
- For single-column pages, output one column with column_id='single'.
- Hyphenated line breaks: when a word is split across two physical lines with
  a trailing hyphen (typesetting wrap), emit the two parts as TWO SEPARATE
  lines exactly as printed, INCLUDING the trailing hyphen. NEVER join them.
  This is one of the most common inconsistencies; be strict about it.
    Example — the page shows:
        intelligi-
        ble, adj. clear, comprehensible.
    You MUST emit two lines:
        "intelligi-"
        "ble, adj. clear, comprehensible."
    NOT one merged line "intelligible, adj. clear, comprehensible." and NOT
    a single line "intelligi-ble, adj. ...". Stage 2 will rejoin hyphenated
    words when it forms entry text — your job is faithful copy only.
"""


STAGE_1_FLAT_SYSTEM = """\
You are a precise OCR transcription system specialising in historical and minority-language dictionaries.

Your task is faithful OCR only — do NOT parse dictionary entries or assign fields.

Output structure:
- `header`: page-level lines at the very top (running title, page number, letter band).
  One string per visible line. Empty list if none. Never put dictionary entries here.
- `lines`: every visible BODY line in reading order. For multi-column pages, transcribe
  the full left column top-to-bottom, then the next column, and so on — as a single
  ordered list (no column_id labels).
- `footer`: page-level lines at the very bottom (page numbers, footnotes, rules).
  One string per visible line. Empty list if none.

You may receive <ocr_reference>...</ocr_reference> from a standard OCR engine. Use it
only for ambiguous character shapes; always prioritise the page image.

Rules for every line in header, lines, and footer:
- Preserve ALL diacritics, stress marks, and special phonetic symbols exactly.
- Wrap bold text in <b>...</b> and italic text in <i>...</i> when confident.
- Do NOT interpret, summarise, merge lines, or fix typos.
- Do NOT skip lines, including continuations and cross-references.
- Hyphenated wraps: when a word breaks across two printed lines with a trailing hyphen,
  emit TWO separate strings (e.g. "intelligi-" then "ble, adj. clear").
"""


def stage_1_user(
    alphabet_text: str = "",
    ocr_hint: str = "",
    guides: str = "",
) -> str:
    """
    Build the user-turn prompt for Stage 1 transcription.

    Args:
        alphabet_text: The alphabet/legend for the script (text form).
                       Used to prime the model on the character inventory.
        ocr_hint: Optional existing OCR output (.txt / .md / .docx text) as a
                  secondary reference for character shapes only.
        guides:   Optional user-defined guidelines appended verbatim under a
                  ``USER DEFINED GUIDELINES`` header at the end of the prompt.
    """
    parts = []
    if alphabet_text:
        parts.append(
            f"""<alphabet>\n{alphabet_text}\n</alphabet>\n\n
            The <alphabet> is a reference guide to the list of characters in the script of the source language, not a strict whitelist. It may be 
            incomplete or not perfectly match this document's script variant.

            Rules:
            1. Prefer <alphabet> matches over visually similar characters from other scripts.
            2. For combinatorial scripts (Indic conjuncts, Ethiopic syllables, Hangul 
            blocks, Arabic ligatures, pointed Hebrew/Syriac), treat <alphabet> 
            entries as base components and form composites using the script's 
            standard rules (virama, vowel diacritics, contextual forms).
            3. If a glyph is clearly a legitimate character of the target script but 
            not in <alphabet> (archaic letters, extensions, related-language 
            characters), transcribe it correctly anyway — do not force-fit.
            4. Preserve diacritics, tone marks, case, and period-specific orthography 
            exactly as shown.
            5. Mark truly unidentifiable glyphs as [?].
            """
        )
    if ocr_hint:
        parts.append(
            f"<ocr_reference>\n{ocr_hint}\n</ocr_reference>\n\n"
            "The OCR reference above may contain errors but can help you identify ambiguous character shapes."
        )
    parts.append(
        "Now transcribe every line of text from the dictionary page image exactly as it appears.` "
        "Preserve all diacritics and special characters."
    )
    if guides:
        parts.append(f"USER DEFINED GUIDELINES\n{guides}")
    return "\n\n".join(parts)


# ── Stage 2: Structuring ──────────────────────────────────────────────────
# System = fixed role + rules.  User = dynamic inputs (transcription, intro, image).

STAGE_2_SCHEMA_BLOCK = """\
Entry schema contract:
  Map visible page text into JSON fields only. Do not use SIL/Toolbox marker names.
  Every row MUST set entry_type. Rows appear in page order; grouping is inferred downstream.

Row types (entry_type):
  main — new bold headword (or homograph main). headword = lemma only (no homograph label).
    parent_lexeme empty. sense_number null. homonym_number = integer 1, 2, … when homographs
    are marked; null otherwise.
  subentry — bold run-on form under the same block as a main lemma.
    parent_lexeme = that main headword. Own gloss, usage_note, pos. homonym_number null.
  sense — numbered meaning under one lemma (MANDATORY when inline numbering is printed).
    parent_lexeme = the lemma headword. sense_number = integer 1, 2, 3, … homonym_number null.
    NEVER merge numbered senses into one main gloss string.

Homograph vs sense:
  Separate bold lines, same lemma + homograph index → separate main rows, homonym_number 1, 2, …
  One bold headword + inline 1.; 2.; 1); 2) → one main + mandatory sense rows
  Bold run-on derivative under same block → subentry

Numbering (normalise to integers):
  homonym_number and sense_number: output 1, 2, 3, … only.
  Convert Roman numerals (I → 1, II → 2). Strip trailing ')' or '.' from printed labels.
  Null when not applicable.

Gloss vs usage_note (see <dictionary_languages>):
  gloss: primary target-language translation — all non-italic wording after the headword.
    Semicolon-separated synonyms allowed in one string.
  gloss_secondary: second target language when the dictionary is trilingual; else "".
  usage_note: italic or parenthetical domain/usage expansion only — not the main translation.

Field hygiene:
  headword: lemma only — no POS or trailing punctuation (POS → pos).
  phonetic, cross_references, examples, example_glosses when marked on the page.
  cross_references: target lemma strings only, strip "see"/"cf." prose.
"""

# Backward-compatible alias for docs and imports.
STAGE_2_MDF_BLOCK = STAGE_2_SCHEMA_BLOCK

STAGE_2_SYSTEM = """\
You are a linguistic expert parsing dictionary pages into structured JSON entries.

Your inputs:
1. A page transcription — either column TSV (column_id, line_number, text) or
   flat text (one line per row, no column_id). Produced by a separate OCR stage.
   <b>...</b> = bold (typically headwords); <i>...</i> = italic (POS, examples, xrefs).
   For column TSV: rows with column_id header or footer (empty line_number) are
   page metadata — IGNORE; they are NOT dictionary entries.
   For flat text: infer headers/footers from position and the introduction.
2. An image of the dictionary page — use for entry boundaries and character accuracy.
3. (Optional) Introduction pages — abbreviations, entry layout, POS and domain conventions.

Your task:
1. Study introduction material (if provided).
2. Read the transcription with the page image; use reading order and <b>/<i> tags.
3. Extract only fields actually present; strip <b>/<i> from values.
4. Set entry_type, parent_lexeme, gloss, usage_note, sense_number, and homonym_number per the schema contract below.

""" + STAGE_2_SCHEMA_BLOCK + """

Examples (lists):
  examples: always a list — one string per citation, in order; never one concatenated string.
  example_glosses: parallel translations when the dictionary gives them.

Field semantics: see the response schema field descriptions.

No-reasoning-in-fields (CRITICAL):
  Field values must contain ONLY extracted dictionary text. No deliberation, hedging,
  or chain-of-thought inside any field. If uncertain about optional fields, leave them
  empty ("" or []). Always populate gloss when translation text is visible.
  Do reasoning only in the thinking channel, never in JSON string values.

Rules:
- Preserve ALL phonetic symbols exactly (ŋ, æ, ʌ, ə, ь, etc.).
- Prioritise the page image over the transcription for character accuracy.
- Do NOT invent fields not visible in the source.
- Process each column independently — entries do not span columns.
- Hyphenated line breaks: rejoin end-of-line hyphens across rows (intelligi- + ble → intelligible)
  in headword, gloss, usage_note, examples, and extra_fields. Keep genuine in-word hyphens.
- Emit clean JSON only — no commentary inside field values.
"""

EXTRA_FIELDS_ALLOWLIST = (
    "plural_form, gender, noun_class, tone_class, register, dialect, "
    "usage_note, inflection, literal_meaning, antonym"
)

EXTRA_FIELDS_DISCOVERY_BLOCK = f"""\
<extra_fields_discovery>
Discovery mode is ENABLED for this run.

Populate extra_fields ONLY for structurally marked content NOT covered by the
canonical schema (phonetic, cross_references, gloss, usage_note, pos, etc.).

Allowed snake_case keys (use ONLY from this list when applicable):
  {EXTRA_FIELDS_ALLOWLIST}

For each qualifying field on an entry, add extra_fields[key] = extracted text.
Multiple values for one key → join with "; ".
Reuse the same key across entries on the page.

Strict rules:
  - Do NOT put ipa, see_also, pronunciation, or cross-refs in extra_fields — use
    phonetic and cross_references on the entry instead.
  - Do NOT duplicate gloss, usage_note, or pos in extra_fields.
  - If no allowlisted extra field applies, leave extra_fields as {{}}.
</extra_fields_discovery>"""


EXTRA_FIELDS_DISABLED_LINE = (
    "Discovery mode is DISABLED. Leave `extra_fields` as {} for every entry."
)


def stage_2_user(
    transcribed_text: str,
    intro_text: str = "",
    discover_extra_fields: bool = False,
    guides: str = "",
    dictionary_languages: str = "",
) -> str:
    """
    Build the user-turn prompt for Stage 2 structuring.

    All dynamic data goes here to keep the system prompt fixed.

    Args:
        transcribed_text:        Stage 1 transcription — column TSV
                                 (column_id \\t line_number \\t text) or flat
                                 line-per-line text.
        intro_text:              Optional introduction/preface text extracted from the dictionary.
        discover_extra_fields:   When True, populate ``extra_fields`` from the frozen
                                 allowlist only. When False, leave ``extra_fields`` as ``{}``.
        guides:                  Optional user-defined guidelines appended verbatim
                                 under a ``USER DEFINED GUIDELINES`` header at the
                                 end of the prompt.
        dictionary_languages:  Pre-rendered ``<dictionary_languages>`` block from
                                 ``DictionaryLanguagesConfig.format_prompt_block()``.
    """
    parts = []

    if dictionary_languages:
        parts.append(dictionary_languages)

    if intro_text:
        parts.append(
            "<dictionary_introduction>\n"
            "The following text is from the dictionary's introduction section. "
            "It explains the entry structure, abbreviations, formatting conventions, "
            "and how to read entries. Use this information to guide your parsing.\n\n"
            f"{intro_text}\n"
            "</dictionary_introduction>"
        )

    parts.append(
        "<transcription>\n"
        f"{transcribed_text}\n"
        "</transcription>"
    )

    if discover_extra_fields:
        parts.append(EXTRA_FIELDS_DISCOVERY_BLOCK)

    closing = (
        "Parse all dictionary entries from the transcription and attached images.\n"
        "The first image is the dictionary page; additional images are introduction pages.\n"
        "Set entry_type on every row (main, subentry, or sense). Use parent_lexeme and "
        "sense_number for subentries and senses; use homonym_number (integer or null, "
        "normalised from printed labels) only for homograph main entries. Emit mandatory "
        "sense rows for inline numbered senses under one headword. Put non-italic "
        "translation text in gloss (and gloss_secondary when trilingual), not in usage_note. "
        "Use phonetic and cross_references on the entry (not extra_fields). "
        "Examples and example_glosses must be lists (one element per example)."
    )
    if not discover_extra_fields:
        closing += "\n" + EXTRA_FIELDS_DISABLED_LINE
    parts.append(closing)

    if guides:
        parts.append(f"USER DEFINED GUIDELINES\n{guides}")

    return "\n\n".join(parts)
