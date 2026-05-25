"""
Group Stage-2 JSON entries into Toolbox records and emit MDF text.

Post-processing assigns record grouping; the LLM schema does not include block_id.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Union

from dictextractor.schemas.dictionary_languages import DictionaryLanguagesConfig
from dictextractor.schemas.entry import DictionaryEntry

logger = logging.getLogger(__name__)

_MDF_MARKER_LINE = re.compile(r"^\\(\w+)\s+(.*)$", re.UNICODE)
_END_SENTENCE_PUNCT = frozenset(".!?")

IssueLevel = Literal["error", "warning"]

EntryLike = Union[DictionaryEntry, Dict]


@dataclass
class ValidationIssue:
    """One validation finding for a Stage-2 page."""

    level: IssueLevel
    message: str
    entry_index: Optional[int] = None
    block_id: str = ""


@dataclass
class ValidationReport:
    """Aggregated validation output."""

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class ToolboxRecord:
    """One blank-line-separated Toolbox lexicon record."""

    block_id: str
    main: DictionaryEntry
    children: List[DictionaryEntry] = field(default_factory=list)


def _as_entry(entry: EntryLike) -> DictionaryEntry:
    if isinstance(entry, DictionaryEntry):
        return entry
    return DictionaryEntry.model_validate(entry)


def strip_end_of_sentence_punctuation(text: str) -> str:
    """Remove trailing sentence punctuation (``.``, ``!``, ``?``) from a field value."""
    trimmed = text.rstrip()
    while trimmed and trimmed[-1] in _END_SENTENCE_PUNCT:
        trimmed = trimmed[:-1].rstrip()
    return trimmed


def normalize_mdf_text(mdf_text: str) -> str:
    """
    Post-process MDF text for Toolbox export.

    Strips end-of-sentence punctuation from marker field values; MDF glosses and
    definitions do not require trailing ``.``, ``!``, or ``?``.
    """
    out_lines: List[str] = []
    for line in mdf_text.splitlines():
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            continue
        match = _MDF_MARKER_LINE.match(stripped)
        if match:
            marker, value = match.groups()
            out_lines.append(f"\\{marker} {strip_end_of_sentence_punctuation(value)}")
        else:
            out_lines.append(stripped)
    return "\n".join(out_lines)


def _split_gloss_text(text: str) -> List[str]:
    """Split a gloss string on semicolons into MDF lines."""
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r";", text) if p.strip()]
    return parts if parts else [text]


def entry_gloss_map(
    entry: DictionaryEntry,
    config: Optional[DictionaryLanguagesConfig],
) -> Dict[str, List[str]]:
    """Map ``gloss`` / ``gloss_secondary`` to language-code → gloss lines."""
    codes = config.target_codes() if config else ["gloss"]
    texts = [entry.gloss, entry.gloss_secondary]
    out: Dict[str, List[str]] = {}
    for code, text in zip(codes, texts):
        lines = _split_gloss_text(text or "")
        if lines:
            out[code] = lines
    return out


def block_ids_for_entries(entries: Sequence[EntryLike]) -> List[str]:
    """Block id per row, inferred from main-row boundaries."""
    rows = normalize_stage2_entries(entries)
    return [_block_id_for_index(rows, idx) for idx in range(len(rows))]


def normalize_stage2_entries(
    entries: Sequence[EntryLike],
) -> List[DictionaryEntry]:
    """Validate and return Stage-2 rows (no block_id on the model)."""
    return [_as_entry(raw) for raw in entries]


def group_entries_to_toolbox_records(
    entries: Sequence[EntryLike],
) -> List[ToolboxRecord]:
    """
    Group flat Stage-2 rows into Toolbox records in page order.

    Each ``main`` starts a new record; following ``subentry`` / ``sense`` rows
    attach to the current record until the next ``main``.
    """
    rows = normalize_stage2_entries(entries)
    records: List[ToolboxRecord] = []
    current: Optional[ToolboxRecord] = None

    for row in rows:
        if row.entry_type == "main":
            hm = row.homonym_number or 0
            block_id = f"{row.headword}:{hm}"
            current = ToolboxRecord(block_id=block_id, main=row)
            records.append(current)
            continue

        if current is None:
            raise ValueError(
                f"First row must be entry_type=main (got {row.entry_type!r} "
                f"for {row.headword!r})."
            )
        if row.entry_type == "main":
            raise ValueError("Duplicate main in group walk.")
        current.children.append(row)

    return records


def _marker_map(config: Optional[DictionaryLanguagesConfig]) -> Dict[str, str]:
    if config is None:
        return {}
    from dictextractor.utils.dictionary_languages import markers_for_config

    return markers_for_config(config)


def _emit_gloss_lines(
    gloss_map: Dict[str, List[str]],
    marker_by_code: Dict[str, str],
    *,
    ordered_codes: Optional[Sequence[str]] = None,
) -> List[str]:
    codes = list(ordered_codes or marker_by_code.keys())
    for code in gloss_map.keys():
        if code not in codes:
            codes.append(code)

    lines: List[str] = []
    for code in codes:
        marker = marker_by_code.get(code, "ge")
        for text in gloss_map.get(code, []):
            if text.strip():
                lines.append(f"\\{marker} {text.strip()}")
    return lines


def _emit_entry_fields(
    entry: DictionaryEntry,
    marker_by_code: Dict[str, str],
    config: Optional[DictionaryLanguagesConfig],
    *,
    include_headword: bool = True,
) -> List[str]:
    lines: List[str] = []
    ordered_codes = config.target_codes() if config else None
    gloss_map = entry_gloss_map(entry, config)

    if include_headword:
        lines.append(f"\\lx {entry.headword}")
        if entry.entry_type == "main" and entry.homonym_number is not None:
            lines.append(f"\\hm {entry.homonym_number}")
    elif entry.entry_type == "subentry":
        lines.append(f"\\se {entry.headword}")
    elif entry.entry_type == "sense" and entry.sense_number is not None:
        lines.append(f"\\sn {entry.sense_number}")

    if entry.pos:
        lines.append(f"\\ps {entry.pos}")
    if entry.phonetic:
        lines.append(f"\\ph {entry.phonetic}")

    lines.extend(
        _emit_gloss_lines(gloss_map, marker_by_code, ordered_codes=ordered_codes)
    )

    if entry.usage_note.strip():
        lines.append(f"\\de {entry.usage_note.strip()}")

    for cf in entry.cross_references:
        if cf.strip():
            lines.append(f"\\cf {cf.strip()}")

    for idx, example in enumerate(entry.examples):
        if example.strip():
            lines.append(f"\\xv {example.strip()}")
            if idx < len(entry.example_glosses) and entry.example_glosses[idx].strip():
                lines.append(f"\\xe {entry.example_glosses[idx].strip()}")

    return lines


def toolbox_record_to_lines(
    record: ToolboxRecord,
    config: Optional[DictionaryLanguagesConfig] = None,
) -> List[str]:
    """Render one Toolbox record as MDF marker lines."""
    marker_by_code = _marker_map(config)
    lines = _emit_entry_fields(
        record.main,
        marker_by_code,
        config,
    )
    for child in record.children:
        lines.extend(
            _emit_entry_fields(
                child,
                marker_by_code,
                config,
                include_headword=False,
            )
        )
    return lines


def toolbox_to_mdf_text(
    records: Sequence[ToolboxRecord],
    config: Optional[DictionaryLanguagesConfig] = None,
) -> str:
    """Serialize Toolbox records to MDF text with blank lines between records."""
    blocks: List[str] = []
    for record in records:
        lines = toolbox_record_to_lines(record, config)
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def entries_to_mdf_text(
    entries: Sequence[EntryLike],
    config: Optional[DictionaryLanguagesConfig] = None,
) -> str:
    """Group entries then emit MDF text."""
    records = group_entries_to_toolbox_records(entries)
    return normalize_mdf_text(toolbox_to_mdf_text(records, config))


def _block_id_for_index(entries: List[DictionaryEntry], index: int) -> str:
    """Block id of the record containing ``entries[index]``."""
    current = ""
    for row in entries[: index + 1]:
        if row.entry_type == "main":
            hm = row.homonym_number or 0
            current = f"{row.headword}:{hm}"
    return current


def validate_stage2_entries(
    entries: Sequence[EntryLike],
    config: Optional[DictionaryLanguagesConfig] = None,
) -> ValidationReport:
    """Validate Stage-2 entries."""
    report = ValidationReport()
    rows = normalize_stage2_entries(entries)
    has_second_target = bool(config and len(config.targets) > 1)

    current_main: Optional[DictionaryEntry] = None
    for idx, entry in enumerate(rows):
        block_id = _block_id_for_index(rows, idx)

        if entry.entry_type == "main":
            current_main = entry
        elif current_main is None:
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"{entry.entry_type} row appears before any main row.",
                    entry_index=idx,
                )
            )

        if entry.entry_type in ("subentry", "sense") and not entry.parent_lexeme.strip():
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"{entry.entry_type} row missing parent_lexeme.",
                    entry_index=idx,
                    block_id=block_id,
                )
            )

        if entry.entry_type == "sense" and entry.sense_number is None:
            report.issues.append(
                ValidationIssue(
                    "error",
                    "sense row missing sense_number.",
                    entry_index=idx,
                    block_id=block_id,
                )
            )

        if entry.entry_type != "main" and entry.homonym_number is not None:
            report.issues.append(
                ValidationIssue(
                    "error",
                    "homonym_number must be null unless entry_type=main.",
                    entry_index=idx,
                    block_id=block_id,
                )
            )

        has_gloss = bool(entry.gloss.strip()) or bool(entry.gloss_secondary.strip())
        if not has_gloss and entry.usage_note.strip():
            report.issues.append(
                ValidationIssue(
                    "warning",
                    "usage_note is populated but gloss is empty — "
                    "move primary translation to gloss.",
                    entry_index=idx,
                    block_id=block_id,
                )
            )

        if not has_gloss and entry.entry_type in ("main", "subentry", "sense"):
            report.issues.append(
                ValidationIssue(
                    "warning",
                    "gloss is empty for a row with expected translation text.",
                    entry_index=idx,
                    block_id=block_id,
                )
            )

        if has_second_target and entry.gloss.strip() and not entry.gloss_secondary.strip():
            if config and config.layout != "bilingual":
                report.issues.append(
                    ValidationIssue(
                        "warning",
                        "gloss_secondary empty on a multi-target dictionary row.",
                        entry_index=idx,
                        block_id=block_id,
                    )
                )

        if (
            current_main
            and entry.entry_type in ("subentry", "sense")
            and entry.parent_lexeme.strip() != current_main.headword.strip()
        ):
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"parent_lexeme {entry.parent_lexeme!r} must match "
                    f"main headword {current_main.headword!r}.",
                    entry_index=idx,
                    block_id=block_id,
                )
            )

    return report


def log_validation_report(report: ValidationReport, *, page_label: str = "") -> None:
    """Log validation issues at warning/error level."""
    prefix = f"{page_label}: " if page_label else ""
    for issue in report.issues:
        msg = f"{prefix}{issue.message}"
        if issue.entry_index is not None:
            msg += f" (entry[{issue.entry_index}])"
        if issue.level == "error":
            logger.error(msg)
        else:
            logger.warning(msg)
