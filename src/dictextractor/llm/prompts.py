"""
Prompt templates for each extraction strategy.
Keeping prompts here separates linguistic engineering from pipeline logic.
"""

from dictextractor.schemas.entry import DictionaryEntry


# ---------------------------------------------------------------------------
# Shared example for few-shot prompting
# ---------------------------------------------------------------------------

EXAMPLE_ENTRY = {
    "headword_phrase": "ac-úkwʌn",
    "entry_type": "word",
    "pos": "сущ.",
    "translation_ru": "кремень",
    "literal_meaning": "жирный камень",
    "grammar_notes": "см. æc/ac",
}


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
2. **Entry_Type**:
   - Label as 'word' if it includes a POS tag or is a single vocabulary item.
   - Label as 'phrase' if it is a multi-word usage example or a full sentence.
3. **POS (Part of Speech)**:
   - Identify abbreviations in parentheses: (сущ.) for noun, (гл.) for verb, (прил.) for adjective, (нар.) for adverb, etc.
4. **Literal_Meaning**:
   - Trigger: The abbreviation "букв." (буквально).
   - Rule: Extract the literal translation immediately following this marker.
5. **Translation (RU)**:
   - The primary Russian definition. If there are multiple meanings, separate them with a semicolon (;).
6. **Grammar/Notes**:
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
- "entry_type": 'word' or 'phrase'.
- "pos": The grammatical category (e.g., 'сущ.', 'гл.').
- "translation_ru": The Russian translation/definition.
- "literal_meaning": The literal translation (found after 'букв.').
- "grammar_notes": Any grammatical forms, tense markers, or cross-references ('см.').

Return ONLY a valid JSON array.
[
{{
    "headword_phrase": "ac-úkwʌn",
    "entry_type": "word",
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
"headword_phrase", "entry_type", "pos", "translation_ru", "literal_meaning", "grammar_notes".

Return ONLY a valid JSON array.
"""


# ---------------------------------------------------------------------------
# Two-stage method prompts
# ---------------------------------------------------------------------------

TWO_STAGE_TRANSCRIBE_SYSTEM = """\
You are a precise OCR transcription system specialising in historical and minority-language dictionaries.

Step 1 — Detect columns.
  Examine the page layout. Most dictionary pages use one or two columns; some use three.
  Identify each column region (left, center, right) before transcribing.

Step 2 — Transcribe each column separately, left to right.
  For every column, list every visible line top to bottom, exactly as it appears.
  Never mix a line from one column into another column's list.

Rules that apply to every line:
- Preserve ALL diacritics, stress marks, and special phonetic symbols exactly.
- Do NOT interpret, restructure, summarise, or add markup.
- Do NOT skip any line, even if it looks like a sub-entry, continuation, or cross-reference.
- Do NOT correct apparent typos or inconsistencies.
- For single-column pages, output one column with column_id='single'.
"""


def two_stage_transcribe_user(alphabet_text: str = "", ocr_hint: str = "") -> str:
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
        "Preserve all diacritics and special characters. Output plain text only — no JSON, no markup."
    )
    return "\n\n".join(parts)


def two_stage_structure_system_prompt(
    transcribed_text: str,
    intro_text: str = "",
) -> str:
    """
    Build the system prompt for Stage 2 structuring.

    Args:
        transcribed_text: TSV output from Stage 1 (column_id \\t line_number \\t text).
        intro_text: Optional introduction/preface of the dictionary.
    """
    intro_section = (
        f"\n<dictionary_introduction>\n{intro_text}\n</dictionary_introduction>\n"
        if intro_text
        else ""
    )
    return f"""\
You are a linguistic expert parsing a dictionary page into structured data.
{intro_section}
The following TSV table is a faithful transcription of the page, produced line by line.
Each row contains the column it came from, its line number within that column, and the text.

<transcription>
{transcribed_text}
</transcription>

Your task:
1. Read the transcription TSV and the image together.
2. Use column_id and line_number to help understand reading order and spatial position on the page.
3. Determine entry boundaries from the visual formatting in the image (bold headwords,
   indentation, line spacing) — multiple consecutive lines in the same column can belong
   to a single entry.
4. For each entry, extract ONLY the fields that are present — leave fields as "" or []
   if they do not appear in that specific entry.

Field reference:
  headword           — the headword word or phrase (all diacritics preserved)
  entry_type         — "word" or "phrase"
  pos                — part-of-speech abbreviation if present, else ""
  target_translation — primary translation in the target language
  literal_definition — literal meaning (e.g. after "букв." marker) if present, else ""
  grammar_notes      — grammatical inflections, tense markers, cross-refs, else ""
  examples           — example sentences/phrases if present, else []

Rules:
- Preserve ALL phonetic symbols exactly (ŋ, æ, ʌ, ə, ь, etc.).
- Prioritise what you see in the image over the transcription for character accuracy.
- Do NOT invent or hallucinate fields not visible in the source.
- Process each column independently — entries do not span columns.
"""


TWO_STAGE_STRUCTURE_USER = """\
Parse all dictionary entries from the transcription TSV and image above.
Use column_id and line_number to understand reading order and spatial position within the page.
Determine entry boundaries from the visual formatting in the image (bold headwords, indentation, spacing).
Multiple consecutive lines in the same column may belong to a single entry.
Only populate a field if it is actually present in that entry.
"""
