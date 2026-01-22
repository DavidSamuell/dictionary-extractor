"""
Utility script to convert dictionary JSON to TSV format
"""

import json
import csv
import argparse
from pathlib import Path


def json_to_tsv(json_path: str, output_path: str = None) -> str:
    """
    Convert a dictionary JSON file to TSV format

    Args:
        json_path: Path to the input JSON file
        output_path: Path to the output TSV file (optional, defaults to same name with .tsv)

    Returns:
        Path to the output TSV file
    """
    json_path = Path(json_path)

    if output_path is None:
        output_path = json_path.with_suffix(".tsv")
    else:
        output_path = Path(output_path)

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    fieldnames = [
        "Headword_Phrase",
        "Entry_Type",
        "POS",
        "Translation_RU",
        "Literal_Meaning",
        "Grammar_Notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as tsvfile:
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for entry in entries:
            writer.writerow({
                "Headword_Phrase": entry.get("headword_phrase", ""),
                "Entry_Type": entry.get("entry_type", ""),
                "POS": entry.get("pos", ""),
                "Translation_RU": entry.get("translation_ru", ""),
                "Literal_Meaning": entry.get("literal_meaning", ""),
                "Grammar_Notes": entry.get("grammar_notes", ""),
            })

    print(f"Converted {len(entries)} entries from {json_path} to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert dictionary JSON to TSV")
    parser.add_argument("json_file", help="Path to the input JSON file")
    parser.add_argument("-o", "--output", help="Path to the output TSV file (optional)")

    args = parser.parse_args()
    json_to_tsv(args.json_file, args.output)


if __name__ == "__main__":
    main()
