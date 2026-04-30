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

CANONICAL_TSV_FIELDS = [
    "Headword",
    "POS",
    "Meaning_Description",
    "Semantic_Domain",
    "Examples",
]


def _key_to_column(key: str) -> str:
    """snake_case extra-field key → Title_Case_Underscore TSV header."""
    return "_".join(part.capitalize() for part in key.split("_") if part)


def _collect_extra_keys(entries: List[Dict]) -> List[str]:
    """
    Union of all extra_fields keys across entries, in first-appearance order.
    Skips keys whose Title_Case form would collide with a canonical column.
    """
    canonical = set(CANONICAL_TSV_FIELDS)
    seen: Dict[str, None] = {}
    for entry in entries:
        for k in (entry.get("extra_fields") or {}).keys():
            if not k or k in seen:
                continue
            if _key_to_column(k) in canonical:
                continue
            seen[k] = None
    return list(seen.keys())


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
    Render a TSV file from a saved entries-JSON file.

    The TSV always contains the canonical columns (Headword, POS,
    Meaning_Description, Semantic_Domain, Examples). When entries carry
    discovered ``extra_fields``, each unique key gets its own additional
    column (snake_case key → Title_Case header), in first-appearance order
    across the entries.

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

    extra_keys = _collect_extra_keys(entries)
    extra_columns = [_key_to_column(k) for k in extra_keys]
    fieldnames = CANONICAL_TSV_FIELDS + extra_columns

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for entry in entries:
            extras = entry.get("extra_fields") or {}
            row = {
                "Headword": entry.get("headword") or entry.get("headword_phrase", ""),
                "POS": entry.get("pos", ""),
                "Meaning_Description": entry.get("meaning_description", ""),
                "Semantic_Domain": entry.get("semantic_domain", ""),
                "Examples": " | ".join(entry.get("examples", [])) if entry.get("examples") else "",
            }
            for key, column in zip(extra_keys, extra_columns):
                row[column] = extras.get(key, "")
            writer.writerow(row)

    print(f"Rendered {len(entries)} entries → {output_path} ({len(extra_columns)} extra columns)")
    return str(output_path)
