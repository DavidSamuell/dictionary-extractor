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
    "Entry_Type",
    "Headword",
    "Parent_Lexeme",
    "Sense_Number",
    "Homonym_Number",
    "POS",
    "Definition",
    "Semantic_Domain",
    "Citation_Form",
    "Phonetic",
    "Cross_References",
    "Examples",
    "Example_Glosses",
]


def _key_to_column(key: str) -> str:
    """snake_case extra-field key → Title_Case_Underscore TSV header."""
    return "_".join(part.capitalize() for part in key.split("_") if part)


def _collect_target_gloss_keys(entries: List[Dict]) -> List[str]:
    """Union of target_glosses keys in first-appearance order."""
    seen: Dict[str, None] = {}
    for entry in entries:
        for k in (entry.get("target_glosses") or {}).keys():
            if k and k not in seen:
                seen[k] = None
    return list(seen.keys())


def _target_gloss_column(code: str) -> str:
    return f"Gloss_{code}"


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


def _resolve_definition(entry: Dict) -> str:
    """Definition column from ``definition`` only."""
    return entry.get("definition") or ""


def _join_list_field(value) -> str:
    """Join list values for TSV; pass through strings unchanged."""
    if isinstance(value, list):
        return " | ".join(str(v) for v in value if v)
    return str(value) if value else ""


def json_to_tsv(json_path: str, output_path: Optional[str] = None) -> str:
    """
    Render a TSV file from a saved entries-JSON file.

    The TSV contains MDF-oriented canonical columns plus any discovered
    ``extra_fields`` columns (snake_case key → Title_Case header).

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

    gloss_keys = _collect_target_gloss_keys(entries)
    gloss_columns = [_target_gloss_column(k) for k in gloss_keys]
    extra_keys = _collect_extra_keys(entries)
    extra_columns = [_key_to_column(k) for k in extra_keys]
    fieldnames = CANONICAL_TSV_FIELDS + gloss_columns + extra_columns

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for entry in entries:
            extras = entry.get("extra_fields") or {}
            tg = entry.get("target_glosses") or {}
            row = {
                "Entry_Type": entry.get("entry_type") or "main",
                "Headword": entry.get("headword") or entry.get("headword_phrase", ""),
                "Parent_Lexeme": entry.get("parent_lexeme", ""),
                "Sense_Number": entry.get("sense_number", ""),
                "Homonym_Number": entry.get("homonym_number", ""),
                "POS": entry.get("pos", ""),
                "Definition": _resolve_definition(entry),
                "Semantic_Domain": entry.get("semantic_domain", ""),
                "Citation_Form": entry.get("citation_form", ""),
                "Phonetic": entry.get("phonetic", ""),
                "Cross_References": _join_list_field(entry.get("cross_references")),
                "Examples": _join_list_field(entry.get("examples")),
                "Example_Glosses": _join_list_field(entry.get("example_glosses")),
            }
            if not tg and entry.get("gloss"):
                tg = {"legacy": entry.get("gloss", "")}
            for code, column in zip(gloss_keys, gloss_columns):
                row[column] = tg.get(code, "")
            for key, column in zip(extra_keys, extra_columns):
                row[column] = extras.get(key, "")
            writer.writerow(row)

    print(f"Rendered {len(entries)} entries → {output_path} ({len(extra_columns)} extra columns)")
    return str(output_path)
