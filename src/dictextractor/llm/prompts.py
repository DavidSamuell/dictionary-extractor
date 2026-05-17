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
            The <alphabet> is a reference guide, not a strict whitelist. It may be 
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

STAGE_2_SYSTEM = """\
You are a linguistic expert parsing dictionary pages into structured data.

Your inputs:
1. A TSV transcription of the page (column_id, line_number, text) — use this for reading order and spatial position. It was produced by a separate OCR stage. The text column preserves visual formatting from the original page:
     <b>...</b> = bold text (typically headwords or entry starts)
     <i>...</i> = italic text (typically POS tags, examples, or cross-references)
   Use these tags as strong signals for identifying entry boundaries and field types.
   Rows whose `column_id` is `header` or `footer` (with empty `line_number`)
   are page-level metadata — running titles, page numbers, chapter abbreviations,
   decorative rules, etc. IGNORE these rows. They are NOT dictionary entries
   and must NOT produce any DictionaryEntry output.
2. An image of the actual dictionary page — use this for visual verification of
   entry boundaries and character accuracy.
3. (Optional) Introduction pages from the dictionary — these explain the dictionary's
   conventions: how entries are structured, what abbreviations mean, how to read
   POS tags, semantic-domain markers, etc. Use them to understand the entry format
   before parsing.

Your task:
1. Study the introduction pages (if provided) to understand the dictionary's
   entry structure, abbreviation key, and formatting conventions.
2. Read the transcription TSV together with the page image.
3. Use column_id and line_number for reading order. Use <b>/<i> tags in the
   transcription and the visual formatting in the image to determine entry
   boundaries — a new <b>...</b> headword typically signals a new entry.
   Multiple consecutive lines in the same column can belong to a single entry.
4. For each entry, extract ONLY the fields that are actually present. Leave
   string fields as "" and list fields as [] when not present.
5. Strip <b> and <i> tags from the extracted field values — they are structural
   hints, not part of the content.

Splitting subentries (IMPORTANT):
  A single visual entry block can contain MULTIPLE distinct headwords/subwords —
  for example, variant spellings, derived forms, run-on subentries, or numbered
  sub-senses with their own translations. When this happens you MUST emit one
  DictionaryEntry per subword, NOT a single combined entry. Indicators include but not limited to:
    - Multiple bolded forms inside one entry block.
    - Numbered sense markers (1., 2., I., II.) each introducing a different
      meaning of a related but distinct headform.
    - Run-on derivatives or compounds listed under a main lemma but bolded
      separately.
  If subentries inherit context (POS, semantic domain) from the parent lemma,
  copy that context into each split entry so each one is self-contained.

Examples handling (IMPORTANT):
  The `examples` field is ALWAYS a list of strings, never a single string.
    - 0 examples → []
    - 1 example  → ["one example string"]
    - N examples → one separate string per example, in the order they appear.
  Do NOT concatenate multiple examples into a single string with separators.
  Each illustrative phrase, sentence, or usage citation gets its own list element.

Field semantics: see the response schema (each field carries its own description).
The handling rules above (subentry splitting, examples-as-list) and the rules
below override anything ambiguous in the schema descriptions.

No-reasoning-in-fields (CRITICAL):
  Field values must contain ONLY the extracted dictionary text. Do NOT write
  deliberation, hedging, self-correction, or chain-of-thought INSIDE any field
  value. Forbidden patterns inside field strings include (non-exhaustive):
    "wait", "let me", "I will", "I'm writing", "actually", "hmm",
    "on second thought", "restart", "chain of thought", "thought block",
    "per rules", "applied here", "rule above".
  Do all reasoning in the dedicated thinking channel, never in the JSON.
  If you are uncertain whether a field applies, leave it empty ("") or [].
  Never narrate the uncertainty in the value itself.

Rules:
- Preserve ALL phonetic symbols exactly (ŋ, æ, ʌ, ə, ь, etc.).
- Prioritise what you see in the page image over the transcription for character accuracy.
- Do NOT invent or hallucinate fields not visible in the source.
- Process each column independently — entries do not span columns.
- Hyphenated line breaks: Stage 1 deliberately preserves typesetting hyphens —
  consecutive transcription rows like "intelligi-" / "ble, adj. clear" are ONE
  word "intelligible". When a field's text spans such a break you MUST rejoin
  it: drop the trailing hyphen and the line boundary so the field value reads
  naturally ("intelligible, adj. clear"), not "intelligi-ble" or
  "intelligi- ble". Apply this to headword, meaning_description, examples,
  and any extra_fields value. Genuine intra-word hyphens (compound words like
  "self-aware", or hyphens that do NOT sit at end-of-line) must be preserved.
- Emit clean JSON only. No commentary, no preamble, no postamble, no notes
  inside string values.
"""


EXTRA_FIELDS_DISCOVERY_BLOCK = """\
<extra_fields_discovery>
Discovery mode is ENABLED for this run.

In addition to the canonical fields (headword, pos, meaning_description,
semantic_domain, examples), scan the page and the introduction (if provided)
for any OTHER fields that the dictionary CONSISTENTLY and STRUCTURALLY marks
on its entries — for example:
  - etymology (origin, root, source language)
  - ipa or pronunciation guides
  - inflectional forms (plural, genitive, past tense, aspect pair, etc.)
  - gender / noun class / tone class
  - register / style markers (formal, slang, archaic, dialectal)
  - cross-references (synonyms, antonyms, "see also")
  - usage notes that the dictionary itself flags with a dedicated marker

For each such field you find on a given entry, add an item to that entry's
`extra_fields` map:
  - Key: a short snake_case English label (e.g. "etymology", "ipa",
    "plural_form", "gender", "register", "see_also").
  - Value: the extracted text as a string. If the dictionary lists multiple
    values (e.g. two cross-references), join them with "; ".
  - Reuse the SAME key across entries when the same field reappears, so
    downstream consumers can group consistently.

Strict rules for `extra_fields`:
  - ONLY include fields that the dictionary visibly marks with a dedicated
    convention (italic abbreviation, special symbol, fixed position, etc.).
    Do NOT invent fields, do NOT add freeform commentary, do NOT duplicate
    canonical fields here.
  - If an entry has no qualifying extra fields, leave its `extra_fields` as {}.
  - Field semantics must be consistent across the whole page — pick the key
    based on what the field IS, not where it appears.
</extra_fields_discovery>"""


EXTRA_FIELDS_DISABLED_LINE = (
    "Discovery mode is DISABLED. Leave `extra_fields` as {} for every entry."
)


def stage_2_user(
    transcribed_text: str,
    intro_text: str = "",
    discover_extra_fields: bool = False,
    guides: str = "",
) -> str:
    """
    Build the user-turn prompt for Stage 2 structuring.

    All dynamic data goes here to keep the system prompt fixed.

    Args:
        transcribed_text:        TSV output from Stage 1 (column_id \\t line_number \\t text).
        intro_text:              Optional introduction/preface text extracted from the dictionary.
        discover_extra_fields:   When True, instruct the LLM to populate
                                 ``DictionaryEntry.extra_fields`` with any
                                 dictionary-marked fields beyond the canonical
                                 schema. When False, instruct it to leave
                                 ``extra_fields`` as ``{}``.
        guides:                  Optional user-defined guidelines appended verbatim
                                 under a ``USER DEFINED GUIDELINES`` header at the
                                 end of the prompt.
    """
    parts = []

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
        "Parse all dictionary entries from the transcription and the attached images.\n"
        "The first image is the dictionary page. Any additional images are pages from "
        "the dictionary's introduction — study them to understand entry structure and conventions.\n"
        "Only populate a field if it is actually present in that entry. "
        "If a single entry block contains multiple subwords/subentries, emit each "
        "as its own entry. Examples must always be a list of strings (one per example)."
    )
    if not discover_extra_fields:
        closing += "\n" + EXTRA_FIELDS_DISABLED_LINE
    parts.append(closing)

    if guides:
        parts.append(f"USER DEFINED GUIDELINES\n{guides}")

    return "\n\n".join(parts)
