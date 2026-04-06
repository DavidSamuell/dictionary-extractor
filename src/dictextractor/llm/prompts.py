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

Step 1 — Detect columns.
  Examine the page layout. Most dictionary pages use one or two columns; some use three.
  Identify each column region (left, center, right) before transcribing.

Step 2 — Transcribe each column separately, left to right.
  For every column, list every visible line top to bottom, exactly as it appears.
  Never mix a line from one column into another column's list.

You may or may not given a ocr reference text wrap around by <ocr_reference>...</ocr_reference>, it comes from a standard OCR result of the page. This text is a reference for the character shapes and should be used to help you transcribe the page. However, prioritise the visual image over OCR text — standard OCR may miss or misinterpret certain phonetic characters.
Rules that apply to every line:
- Preserve ALL diacritics, stress marks, and special phonetic symbols exactly.
- Preserve visual formatting: wrap bold text in <b>...</b> and italic text in <i>...</i>.
  Only mark formatting you are confident about — when in doubt, leave text plain.
  These tags may appear anywhere within a line (e.g. "<b>headword</b> translation").
- Do NOT interpret, restructure, summarise, or reorder content.
- Do NOT skip any line, even if it looks like a sub-entry, continuation, or cross-reference.
- Do NOT correct apparent typos or inconsistencies.
- For single-column pages, output one column with column_id='single'.
"""


def stage_1_user(alphabet_text: str = "", ocr_hint: str = "") -> str:
    """
    Build the user-turn prompt for Stage 1 transcription.

    Args:
        alphabet_text: The alphabet/legend for the script (text form).
                       Used to prime the model on the character inventory.
        ocr_hint: Optional existing OCR output (.txt / .md / .docx text) as a
                  secondary reference for character shapes only.
    """
    parts = []
    if alphabet_text:
        parts.append(
            f"<alphabet>\n{alphabet_text}\n</alphabet>\n\n"
            "Use the alphabet above as the authoritative character set. "
            "Every character in the image should match one of these characters exactly."
        )
    if ocr_hint:
        parts.append(
            f"<ocr_reference>\n{ocr_hint}\n</ocr_reference>\n\n"
            "The OCR reference above may contain errors but can help you identify ambiguous character shapes."
        )
    parts.append(
        "Now transcribe every line of text from the dictionary page image exactly as it appears. "
        "Preserve all diacritics and special characters."
    )
    return "\n\n".join(parts)


# ── Stage 2: Structuring ──────────────────────────────────────────────────
# System = fixed role + rules.  User = dynamic inputs (transcription, intro, image).

STAGE_2_SYSTEM = """\
You are a linguistic expert parsing dictionary pages into structured data.

Your inputs:
1. A TSV transcription of the page (column_id, line_number, text) — use this for
   reading order and spatial position. It was produced by a separate OCR stage.
   The text column preserves visual formatting from the original page:
     <b>...</b> = bold text (typically headwords or entry starts)
     <i>...</i> = italic text (typically POS tags, examples, or cross-references)
   Use these tags as strong signals for identifying entry boundaries and field types.
2. An image of the actual dictionary page — use this for visual verification of
   entry boundaries and character accuracy.
3. (Optional) Introduction pages from the dictionary — these explain the dictionary's
   conventions: how entries are structured, what abbreviations mean, how to read
   POS tags, cross-references, etc. Use them to understand the entry format before
   parsing.

Your task:
1. Study the introduction pages (if provided) to understand the dictionary's
   entry structure, abbreviation key, and formatting conventions.
2. Read the transcription TSV together with the page image.
3. Use column_id and line_number for reading order. Use <b>/<i> tags in the
   transcription and the visual formatting in the image to determine entry
   boundaries — a new <b>...</b> headword typically signals a new entry.
   Multiple consecutive lines in the same column can belong to a single entry.
4. For each entry, extract ONLY the fields that are actually present. Leave
   fields as "" or [] if they do not appear in that specific entry.
5. Strip <b> and <i> tags from the extracted field values — they are structural
   hints, not part of the content.

Field reference:
  headword           — the headword word or phrase (all diacritics preserved)
  pos                — part-of-speech abbreviation if present, else ""
  target_translation — primary translation in the target language
  literal_definition — literal meaning if present, else ""
  grammar_notes      — grammatical inflections, tense markers, cross-refs, else ""
  examples           — example sentences/phrases if present, else []

Rules:
- Preserve ALL phonetic symbols exactly (ŋ, æ, ʌ, ə, ь, etc.).
- Prioritise what you see in the page image over the transcription for character accuracy.
- Do NOT invent or hallucinate fields not visible in the source.
- Process each column independently — entries do not span columns.
"""


def stage_2_user(transcribed_text: str, intro_text: str = "") -> str:
    """
    Build the user-turn prompt for Stage 2 structuring.

    All dynamic data goes here to keep the system prompt fixed.

    Args:
        transcribed_text: TSV output from Stage 1 (column_id \\t line_number \\t text).
        intro_text: Optional introduction/preface text extracted from the dictionary.
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

    parts.append(
        "Parse all dictionary entries from the transcription and the attached images.\n"
        "The first image is the dictionary page. Any additional images are pages from "
        "the dictionary's introduction — study them to understand entry structure and conventions.\n"
        "Only populate a field if it is actually present in that entry."
    )

    return "\n\n".join(parts)
