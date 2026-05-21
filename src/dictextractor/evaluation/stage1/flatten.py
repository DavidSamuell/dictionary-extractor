"""Flatten stage-1 column TSV into per-page line text for eval-flat.

See ``PLAN.md`` §3 Layer 2 (spec v2): header rows (file order), body columns
left → center → right → single (``middle`` treated as center), footer rows
(file order).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Mapping, Sequence

logger = logging.getLogger(__name__)

FLAT_SPEC_VERSION = "v2"

_METADATA_COLUMN_IDS = frozenset({"header", "footer"})

# Sort key for body columns (lower sorts first).
_BODY_COLUMN_RANK: Mapping[str, int] = {
    "left": 0,
    "center": 1,
    "middle": 1,
    "right": 2,
    "single": 3,
}

_DEFAULT_BODY_RANK = 99


def _parse_line_number(raw: str) -> int:
    """Parse TSV line_number; missing or invalid values sort first within a column."""
    stripped = (raw or "").strip()
    if not stripped:
        return 0
    try:
        return int(stripped)
    except ValueError:
        logger.warning("Non-integer line_number %r; treating as 0", raw)
        return 0


def _metadata_lines(rows: Sequence[Mapping[str, str]], column_id: str) -> list[str]:
    """Collect header or footer lines in TSV file order."""
    return [
        row.get("text") or ""
        for row in rows
        if (row.get("column_id") or "").strip() == column_id
    ]


def flatten_stage1_body_rows(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """
    Convert body TSV rows to column-major lines (excludes header/footer).

    Time complexity: O(n log n) for n body rows.
    """
    by_column: dict[str, list[tuple[int, int, str]]] = {}
    for index, row in enumerate(rows):
        col = (row.get("column_id") or "").strip()
        if col in _METADATA_COLUMN_IDS:
            continue
        text = row.get("text") or ""
        line_no = _parse_line_number(row.get("line_number") or "")
        by_column.setdefault(col, []).append((line_no, index, text))

    ordered_columns = sorted(
        by_column.keys(),
        key=lambda c: (_BODY_COLUMN_RANK.get(c, _DEFAULT_BODY_RANK), c),
    )
    for col in ordered_columns:
        if col not in _BODY_COLUMN_RANK:
            logger.warning(
                "Unknown body column_id %r; appending after known columns", col
            )

    lines: list[str] = []
    for col in ordered_columns:
        entries = sorted(by_column[col], key=lambda item: (item[0], item[1]))
        lines.extend(text for _, _, text in entries)
    return lines


def flatten_stage1_rows(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """
    Convert stage-1 TSV rows to flat eval lines (spec v2).

    Order: header (file order) → body (column-major) → footer (file order).
    """
    headers = _metadata_lines(rows, "header")
    body = flatten_stage1_body_rows(rows)
    footers = _metadata_lines(rows, "footer")
    return headers + body + footers


def flat_transcription_to_text(
    header: Sequence[str],
    lines: Sequence[str],
    footer: Sequence[str],
) -> str:
    """Serialize flat transcription parts to newline-separated page text (v2)."""
    return "\n".join(list(header) + list(lines) + list(footer))


def flatten_stage1_tsv(tsv_path: str | Path) -> str:
    """
    Read a stage-1 TSV file and return flat page text (lines joined by ``\\n``).

    Args:
        tsv_path: Path to ``*_stage1_GOLD.tsv`` or prediction TSV with the same schema.

    Returns:
        Newline-separated lines per flat spec v2.
    """
    path = Path(tsv_path)
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "column_id" not in reader.fieldnames:
            raise ValueError(f"Invalid stage-1 TSV (missing column_id): {path}")
        for row in reader:
            rows.append(dict(row))
    return flat_transcription_to_text(
        _metadata_lines(rows, "header"),
        flatten_stage1_body_rows(rows),
        _metadata_lines(rows, "footer"),
    )


def load_flat_lines(flat_path: str | Path) -> list[str]:
    """Load a flat text file into lines (no trailing empty line normalization)."""
    text = Path(flat_path).read_text(encoding="utf-8")
    if not text:
        return []
    return text.splitlines()


def flat_output_path_for_gold(gold_tsv_path: Path) -> Path:
    """Return ``<stem>_stage1_GOLD_flat.txt`` beside the gold TSV."""
    stem = gold_tsv_path.name.replace("_stage1_GOLD.tsv", "")
    return gold_tsv_path.parent / f"{stem}_stage1_GOLD_flat.txt"


def flat_output_path_for_pred(page_dir: Path, stem: str) -> Path:
    """Return ``<stem>_stage1_flat.txt`` for a prediction page directory."""
    return page_dir / f"{stem}_stage1_flat.txt"


def write_flat_text(path: Path, lines: Sequence[str]) -> Path:
    """Write flat lines to *path* (always overwrites)."""
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    return path


def write_flat_gold(gold_tsv_path: Path) -> Path:
    """
    Write flattened gold next to the source TSV (always overwrites).

    Returns:
        Path to the written flat file.
    """
    out_path = flat_output_path_for_gold(gold_tsv_path)
    flat_text = flatten_stage1_tsv(gold_tsv_path)
    write_flat_text(out_path, flat_text.splitlines() if flat_text else [])
    return out_path
