"""
Chukchi-Russian Dictionary Extractor
Extracts dictionary entries from images using LLM (Gemini 2.5 Pro)
"""

import json
import os
import base64
import csv
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import litellm
from pydantic import BaseModel, Field
from docx import Document
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DictionaryEntry(BaseModel):
    """Schema for a dictionary entry"""

    headword_phrase: str = Field(
        ...,
        description="The Chukchi word or phrase with all diacritical marks preserved",
    )
    entry_type: str = Field(
        default="word", description="Type of entry: 'word' or 'phrase'"
    )
    pos: str = Field(
        default="", description="Part of speech (e.g., сущ., гл., нар., прил.)"
    )
    translation_ru: str = Field(..., description="The Russian translation/definition")
    literal_meaning: str = Field(
        default="", description="Literal translation (marked by 'букв.')"
    )
    grammar_notes: str = Field(
        default="", description="Grammatical forms, tense markers, or cross-references"
    )


class DictionaryPage(BaseModel):
    """Container for all entries on a page"""

    entries: List[DictionaryEntry]
    page_number: int
    source_file: str


def read_docx_text(docx_path: str) -> str:
    """
    Read text from a DOCX file
    """
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    return "\n".join(full_text)


def encode_image_base64(image_path: str) -> str:
    """
    Encode an image file to base64
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def extract_dictionary_entries(
    image_path: str,
    docx_path: Optional[str] = None,
    page_number: int = 1,
    model: str = "openrouter/qwen/qwen3-vl-30b-a3b-thinking",
) -> DictionaryPage:
    """
    Extract dictionary entries from an image using LLM

    Args:
        image_path: Path to the dictionary page image
        docx_path: Optional path to extracted text in DOCX format
        page_number: Page number for tracking
        model: LLM model to use

    Returns:
        DictionaryPage object containing all extracted entries
    """

    # Prepare the response schema
    response_schema = {"type": "array", "items": DictionaryEntry.model_json_schema()}

    # Read the extracted text if available (limit to avoid token issues)
    # DISABLED: OCR text increases token usage and can cause truncation issues
    # The model can read the image directly, so this is optional
    extracted_text = ""
    if docx_path and Path(docx_path).exists():
        extracted_text = read_docx_text(docx_path)

    # Encode the image
    encoded_image = encode_image_base64(image_path)

    # Prepare the prompt
    system_prompt = """ You are a linguistic expert digitizing a Chukchi-Russian dictionary. 
                        Your task is to extract entries into a structured JSON format for high-accuracy TSV conversion.

                        ### EXTRACTION LOGIC & MAPPING RULES:
                        1. **Headword/Phrase**: 
                        - **Words**: The bolded Chukchi term at the start. Preserve all diacritics and stress marks.
                        - **Phrases**: Identify these by the absence of an immediate POS tag in parentheses. Often separated from the Russian translation by a dash (—).
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
                        - Capture specific Chukchi grammatical inflections, such as tense markers (e.g., "прош. II") or plural forms, often found in parentheses.

                        ### CRITICAL FORMATTING RULES:
                        - **Encoding**: Preserve ALL phonetic symbols (e.g., ŋ, æ, ʌ, ь, ə) exactly. Chukchi uses specific characters that must not be simplified to standard Latin or Cyrillic.
                        - **Cleanup**: In the final JSON, provide the content only—remove the "букв." prefixes to keep the data clean for the TSV.
                        - **Accuracy**: Prioritize the visual image over OCR text, as standard OCR could sometimes misinterprets Chukchi phonetic characters. """

    user_prompt = f"""<extracted_text>{extracted_text}</extracted_text>. 
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

    # Prepare the LLM request
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                },
            ],
        },
    ]

    # Make the API call
    print(f"Calling LLM API with model: {model}...")

    # Determine API key based on model provider
    api_key = None
    if "openrouter" in model.lower():
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
    elif "gemini" in model.lower() or "google" in model.lower():
        api_key = os.getenv("GEMINI_API_KEY")
    elif "claude" in model.lower() or "anthropic" in model.lower():
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif "gpt" in model.lower() or "openai" in model.lower():
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print(
            f"Warning: No API key found for model {model}. Relying on environment variables."
        )

    # Configure API call
    api_params = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,  # Lower temperature for more consistent extraction
        "max_tokens": 64000,  # Reasonable limit for dictionary page extraction
    }

    # Only add api_key if we found one
    if api_key:
        api_params["api_key"] = api_key

    # For Qwen3 Next thinking models, the thinking is built into the model
    # For Gemini 2.5 models, try to disable extended thinking
    if "2.5" in model and "gemini" in model.lower():
        api_params["extra_body"] = {
            "generationConfig": {"thinking": {"thinkingConfig": {"mode": "DISABLED"}}}
        }

    response = litellm.completion(**api_params)

    # Debug: print response structure
    print(f"Response received. Model: {response.model}")
    print(f"Finish reason: {response.choices[0].finish_reason}")
    print(f"Usage: {response.usage}")

    content = response.choices[0].message.content

    if content is None:
        print("Error: Response content is None")
        print(f"Full response: {response}")
        raise ValueError(
            "API returned None content. This might be due to token limits or API configuration."
        )

    print(f"Content length: {len(content)} characters")
    print(f"First 200 characters: {content[:200]}")

    # Parse the response - handle potential markdown code blocks
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    elif content.startswith("```"):
        content = content[3:]  # Remove ```
    if content.endswith("```"):
        content = content[:-3]  # Remove trailing ```
    content = content.strip()

    try:
        entries_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Content that failed to parse: {content[:500]}")
        raise

    # Handle both array and object with entries key
    if isinstance(entries_data, list):
        raw_entries = entries_data
    elif isinstance(entries_data, dict) and "entries" in entries_data:
        raw_entries = entries_data["entries"]
    elif isinstance(entries_data, dict):
        # Might be a single entry
        raw_entries = [entries_data]
    else:
        raise ValueError(f"Unexpected response format: {type(entries_data)}")

    # Clean up entries - convert None to empty string and validate
    cleaned_entries = []
    for entry in raw_entries:
        # Replace None values with empty strings
        cleaned_entry = {
            "headword_phrase": entry.get("headword_phrase") or "",
            "entry_type": entry.get("entry_type") or "word",
            "pos": entry.get("pos") or "",
            "translation_ru": entry.get("translation_ru") or "",
            "literal_meaning": entry.get("literal_meaning") or "",
            "grammar_notes": entry.get("grammar_notes") or "",
        }
        # Skip entries without a headword_phrase
        if cleaned_entry["headword_phrase"]:
            cleaned_entries.append(DictionaryEntry(**cleaned_entry))

    entries = cleaned_entries
    print(
        f"Validated {len(entries)} entries (filtered out {len(raw_entries) - len(entries)} incomplete entries)"
    )

    return DictionaryPage(
        entries=entries, page_number=page_number, source_file=image_path
    )


def save_to_tsv(
    dictionary_page: DictionaryPage, output_path: str, append: bool = False
):
    """
    Save dictionary entries to TSV format

    Args:
        dictionary_page: DictionaryPage object containing entries
        output_path: Path to the output TSV file
        append: Whether to append to existing file or overwrite
    """
    mode = "a" if append else "w"
    file_exists = Path(output_path).exists() and append

    with open(output_path, mode, newline="", encoding="utf-8") as tsvfile:
        fieldnames = [
            "Headword_Phrase",
            "Entry_Type",
            "POS",
            "Translation_RU",
            "Literal_Meaning",
            "Grammar_Notes",
        ]
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")

        if not file_exists or not append:
            writer.writeheader()

        for entry in dictionary_page.entries:
            writer.writerow(
                {
                    "Headword_Phrase": entry.headword_phrase,
                    "Entry_Type": entry.entry_type,
                    "POS": entry.pos,
                    "Translation_RU": entry.translation_ru,
                    "Literal_Meaning": entry.literal_meaning,
                    "Grammar_Notes": entry.grammar_notes,
                }
            )

    print(f"Saved {len(dictionary_page.entries)} entries to {output_path}")


def main():
    """
    Main function to extract dictionary entries from the test image
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract dictionary entries from images using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with just image
  python extract_dictionary.py -i page1.png -o output.tsv
  
  # With OCR text reference
  python extract_dictionary.py -i page1.png -d page1.docx -o output.tsv
  
  # With specific model
  python extract_dictionary.py -i page1.png -o output.tsv -m gemini/gemini-2.5-pro
  
  # With page number for tracking
  python extract_dictionary.py -i page1.png -o output.tsv -p 1
  
  # Append to existing TSV
  python extract_dictionary.py -i page2.png -o output.tsv -a
        """,
    )

    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Path to the dictionary page image (PNG, JPG, etc.)",
    )

    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to the output TSV file"
    )

    parser.add_argument(
        "-d",
        "--docx",
        type=str,
        default=None,
        help="Optional path to extracted text in DOCX format (for OCR reference)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openrouter/qwen/qwen3-vl-30b-a3b-thinking",
        help="LLM model to use (default: openrouter/qwen/qwen3-vl-30b-a3b-thinking)",
    )

    parser.add_argument(
        "-p",
        "--page",
        type=int,
        default=1,
        help="Page number for tracking (default: 1)",
    )

    parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to existing TSV file instead of overwriting",
    )

    parser.add_argument(
        "--json", action="store_true", help="Also save output in JSON format"
    )

    args = parser.parse_args()

    # Validate paths
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    docx_path = args.docx
    if docx_path:
        docx_path = Path(docx_path)
        if not docx_path.exists():
            print(f"Warning: DOCX file not found at {docx_path}, proceeding without it")
            docx_path = None

    print(f"Extracting dictionary entries from: {image_path}")
    if docx_path:
        print(f"Using extracted text from: {docx_path}")
    print(f"Using model: {args.model}")
    print(f"Page number: {args.page}")
    print(f"Output: {output_path}")
    print(f"Mode: {'Append' if args.append else 'Overwrite'}")
    print("-" * 60)

    try:
        # Extract entries
        dictionary_page = extract_dictionary_entries(
            image_path=str(image_path),
            docx_path=str(docx_path) if docx_path else None,
            page_number=args.page,
            model=args.model,
        )

        print(f"Successfully extracted {len(dictionary_page.entries)} entries")

        # Save to TSV
        save_to_tsv(dictionary_page, str(output_path), append=args.append)

        # Print first few entries as preview
        print("\nFirst 5 entries extracted:")
        for i, entry in enumerate(dictionary_page.entries[:5]):
            translation_preview = (
                entry.translation_ru[:50]
                if len(entry.translation_ru) > 50
                else entry.translation_ru
            )
            literal = (
                f" [lit: {entry.literal_meaning}]" if entry.literal_meaning else ""
            )
            print(
                f"{i+1}. {entry.headword_phrase} ({entry.pos}) [{entry.entry_type}]: {translation_preview}{literal}"
            )

        # Save as JSON if requested
        if args.json:
            json_output = output_path.with_suffix(".json")
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(
                    [entry.model_dump() for entry in dictionary_page.entries],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"\nAlso saved JSON format to: {json_output}")

        return 0

    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
