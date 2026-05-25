"""Flatten stage-1 column TSV into per-page line text for eval-flat.

See ``PLAN.md`` §3 Layer 2 (spec v2): header rows (file order), body columns
left → center → right → single (``middle`` treated as center), footer rows
(file order).

``Circassian-English-Turkish`` uses row-major body flattening: each dictionary
row spans all three columns horizontally, so lines with the same offset from
each column header are emitted left → center → right.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Mapping, Sequence

logger = logging.getLogger(__name__)

FLAT_SPEC_VERSION = "v2"

ROW_MAJOR_FLAT_LANGUAGES = frozenset({"Circassian-English-Turkish"})

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


_COLUMN_HEADER_LABELS: Mapping[str, str] = {
    "left": "english.",
    "center": "circassian.",
    "middle": "circassian.",
    "right": "turkish.",
}


def _normalize_header_text(text: str) -> str:
    """Strip markup and lowercase for column header detection."""
    stripped = re.sub(r"<[^>]+>", "", text).strip().lower()
    return stripped


def _body_rows_by_column(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[int, str]]:
    """Group body rows as column_id → line_number → text."""
    by_column: dict[str, dict[int, str]] = {}
    for row in rows:
        col = (row.get("column_id") or "").strip()
        if col in _METADATA_COLUMN_IDS:
            continue
        line_no = _parse_line_number(row.get("line_number") or "")
        text = row.get("text") or ""
        by_column.setdefault(col, {})[line_no] = text
    return by_column


def _ordered_body_columns(by_column: Mapping[str, dict[int, str]]) -> list[str]:
    ordered_columns = sorted(
        by_column.keys(),
        key=lambda c: (_BODY_COLUMN_RANK.get(c, _DEFAULT_BODY_RANK), c),
    )
    for col in ordered_columns:
        if col not in _BODY_COLUMN_RANK:
            logger.warning(
                "Unknown body column_id %r; appending after known columns", col
            )
    return ordered_columns


def _column_header_line(column_lines: Mapping[int, str], column_id: str) -> int | None:
    """Return the line_number of a column header row, or the first line."""
    if not column_lines:
        return None
    expected = _COLUMN_HEADER_LABELS.get(column_id)
    if expected is not None:
        for line_no in sorted(column_lines):
            if _normalize_header_text(column_lines[line_no]) == expected:
                return line_no
    return min(column_lines)


def flatten_stage1_body_rows_column_major(rows: Sequence[Mapping[str, str]]) -> list[str]:
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

    ordered_columns = _ordered_body_columns(
        {col: {line_no: text for line_no, _, text in entries} for col, entries in by_column.items()}
    )

    lines: list[str] = []
    for col in ordered_columns:
        entries = sorted(by_column[col], key=lambda item: (item[0], item[1]))
        lines.extend(text for _, _, text in entries)
    return lines


def flatten_stage1_body_rows_row_major(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """
    Convert body TSV rows to row-major lines for horizontally aligned columns.

    Lines before the left column header (e.g. page markers) are emitted first.
    Then for each row offset from the column headers, emit left → center → right.

    Time complexity: O(n log n) for n body rows.
    """
    by_column = _body_rows_by_column(rows)
    ordered_columns = _ordered_body_columns(by_column)
    header_lines = {
        col: _column_header_line(by_column[col], col)
        for col in ordered_columns
        if col in by_column
    }

    lines: list[str] = []
    left_header = header_lines.get("left")
    if left_header is not None:
        for line_no in sorted(by_column["left"]):
            if line_no < left_header:
                lines.append(by_column["left"][line_no])

    max_offset = 0
    for col in ordered_columns:
        header_line = header_lines.get(col)
        if header_line is None:
            continue
        max_offset = max(max_offset, max(by_column[col]) - header_line)

    for offset in range(max_offset + 1):
        for col in ordered_columns:
            header_line = header_lines.get(col)
            if header_line is None:
                continue
            line_no = header_line + offset
            if line_no in by_column[col]:
                lines.append(by_column[col][line_no])
    return lines


def flatten_stage1_body_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    language: str | None = None,
) -> list[str]:
    """
    Convert body TSV rows to flat lines (excludes header/footer).

    Uses row-major order for ``Circassian-English-Turkish``; column-major otherwise.
    """
    if language in ROW_MAJOR_FLAT_LANGUAGES:
        return flatten_stage1_body_rows_row_major(rows)
    return flatten_stage1_body_rows_column_major(rows)


def language_from_sample_path(path: str | Path) -> str | None:
    """Extract language folder name from a path under ``assets/dictionaries/samples``."""
    parts = Path(path).parts
    try:
        samples_idx = parts.index("samples")
    except ValueError:
        return None
    if samples_idx + 1 >= len(parts):
        return None
    return parts[samples_idx + 1]


def flatten_stage1_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    language: str | None = None,
) -> list[str]:
    """
    Convert stage-1 TSV rows to flat eval lines (spec v2).

    Order: header (file order) → body → footer (file order).
    """
    headers = _metadata_lines(rows, "header")
    body = flatten_stage1_body_rows(rows, language=language)
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
    language = language_from_sample_path(path)
    return flat_transcription_to_text(
        _metadata_lines(rows, "header"),
        flatten_stage1_body_rows(rows, language=language),
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
