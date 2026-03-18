"""
File I/O helpers: reading source documents and writing extraction outputs.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_docx_text(docx_path: str) -> str:
    """
    Read and return all non-empty paragraph text from a .docx file.

    Args:
        docx_path: Path to the DOCX file.

    Returns:
        Newline-joined paragraph text.
    """
    from docx import Document
    doc = Document(docx_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_text_file(text_path: str) -> str:
    """
    Read a plain text or Markdown file.

    Args:
        text_path: Path to the text file.

    Returns:
        File contents as a string.
    """
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()


def load_tsv(filepath: str) -> List[Dict[str, str]]:
    """
    Load a TSV file into a list of row dictionaries.

    Args:
        filepath: Path to the TSV file.

    Returns:
        List of dicts, one per row.
    """
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entries.append(dict(row))
    return entries


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

TSV_FIELDNAMES = [
    "Headword",
    "Entry_Type",
    "POS",
    "Target_Translation",
    "Literal_Definition",
    "Grammar_Notes",
    "Examples",
]


def save_to_tsv(dictionary_page, output_path: str, append: bool = False) -> None:
    """
    Write a DictionaryPage to a TSV file.

    Args:
        dictionary_page: DictionaryPage instance with .entries list.
        output_path: Destination file path.
        append: If True, append without writing a header (unless file is new).
    """
    mode = "a" if append else "w"
    file_exists = Path(output_path).exists() and append

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDNAMES, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        for entry in dictionary_page.entries:
            writer.writerow(
                {
                    "Headword": entry.headword,
                    "Entry_Type": entry.entry_type,
                    "POS": entry.pos,
                    "Target_Translation": entry.target_translation,
                    "Literal_Definition": entry.literal_definition,
                    "Grammar_Notes": entry.grammar_notes,
                    "Examples": " | ".join(entry.examples) if entry.examples else "",
                }
            )
    print(f"Saved {len(dictionary_page.entries)} entries to {output_path}")


def save_to_json(dictionary_page, output_path: str) -> None:
    """
    Write a DictionaryPage to a JSON file (array of entry dicts).

    Args:
        dictionary_page: DictionaryPage instance.
        output_path: Destination file path.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [entry.model_dump() for entry in dictionary_page.entries],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved JSON to {output_path}")


def json_to_tsv(json_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert a dictionary JSON file (array of entry dicts) to TSV.

    Args:
        json_path: Path to the input JSON file.
        output_path: Optional destination path (defaults to same stem + .tsv).

    Returns:
        Path to the written TSV file.
    """
    json_path = Path(json_path)
    if output_path is None:
        output_path = json_path.with_suffix(".tsv")
    else:
        output_path = Path(output_path)

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "Headword": entry.get("headword") or entry.get("headword_phrase", ""),
                    "Entry_Type": entry.get("entry_type", ""),
                    "POS": entry.get("pos", ""),
                    "Target_Translation": entry.get("target_translation") or entry.get("translation_ru", ""),
                    "Literal_Definition": entry.get("literal_definition") or entry.get("literal_meaning", ""),
                    "Grammar_Notes": entry.get("grammar_notes", ""),
                    "Examples": " | ".join(entry.get("examples", [])) if entry.get("examples") else "",
                }
            )

    print(f"Converted {len(entries)} entries from {json_path} to {output_path}")
    return str(output_path)
