"""Wire and validate per-language sample folder inputs into CLI args."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dictextractor.ocr.vlm.page_inputs import list_snippet_pages
from dictextractor.ocr.vlm.prompts import (
    find_ocr_hint_file,
    load_alphabet_text,
    load_ocr_hint_text,
)

_IMAGE_ALPHABET_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def stage1_context_inputs_apply(args: Any) -> bool:
    """Return True when alphabet/OCR hint settings apply to this run."""
    strategy = getattr(args, "strategy", None)
    if strategy in ("vlm_ocr", "mathpix_ocr"):
        return True
    if strategy == "two_stage":
        return getattr(args, "stage", "both") in ("1", "both")
    return False


def configure_sample_entry_args(args: Any, entry_dir: Path) -> tuple[Path, Path]:
    """Discover ``snippets/``, ``alphabet.txt``, ``mathpix/``, etc. for one entry.

    Mutates ``args`` in place (same fields as ``--samples-dir`` batch mode for
    ``two_stage``). Respects ``--no-alphabet`` and ``--no-ocr-hint``.

    Args:
        args: Parsed CLI namespace.
        entry_dir: Language folder under ``--samples-dir``.

    Returns:
        ``(snippets_dir, output_dir)`` for the entry.
    """
    snippets_dir = entry_dir / "snippets"
    intro_dir = entry_dir / "introduction"
    mathpix_dir = entry_dir / "mathpix"
    alphabet_file = entry_dir / "alphabet.txt"
    output_dir = entry_dir / "outputs"

    args.input_image = str(snippets_dir)
    args.ocr_text = (
        str(mathpix_dir)
        if mathpix_dir.is_dir() and not getattr(args, "no_ocr_hint", False)
        else None
    )
    args.intro = (
        str(intro_dir)
        if intro_dir.is_dir() and not getattr(args, "no_intro", False)
        else None
    )
    args.alphabet = (
        str(alphabet_file)
        if alphabet_file.exists() and not getattr(args, "no_alphabet", False)
        else None
    )
    args.output = str(output_dir)
    args.entry_dir = str(entry_dir)
    return snippets_dir, output_dir


def validate_alphabet_file(path: Path) -> list[str]:
    """Return validation errors for an alphabet file (empty list if OK)."""
    if not path.is_file():
        return [f"alphabet file not found: {path}"]
    suffix = path.suffix.lower()
    if suffix in _IMAGE_ALPHABET_EXTS:
        if path.stat().st_size == 0:
            return [f"alphabet image is empty: {path}"]
        return []
    if not load_alphabet_text(str(path)).strip():
        return [f"alphabet file is empty: {path}"]
    return []


def validate_mathpix_sources_for_snippets(
    mathpix_dir: Path,
    snippets_dir: Path,
) -> list[str]:
    """Return validation errors for Mathpix markdown, DOCX, or lines.json sources."""
    if not mathpix_dir.is_dir():
        return [f"Mathpix source directory not found: {mathpix_dir}"]

    try:
        snippets = list_snippet_pages(snippets_dir)
    except FileNotFoundError as exc:
        return [str(exc)]

    errors: list[str] = []
    for snippet in snippets:
        hint = find_ocr_hint_file(mathpix_dir, snippet.stem)
        lines_json = mathpix_dir / f"{snippet.stem}.lines.json"
        if hint is None and not lines_json.is_file():
            errors.append(
                f"Mathpix source missing for {snippet.stem} "
                f"(expected {mathpix_dir}/{snippet.stem}.{{md,docx,lines.json}})"
            )
            continue
        if lines_json.is_file():
            continue
        if hint is not None:
            try:
                hint_text = load_ocr_hint_text(hint).strip()
            except (OSError, ValueError) as exc:
                errors.append(f"Mathpix source unreadable: {hint} ({exc})")
                continue
            if not hint_text:
                errors.append(f"Mathpix source is empty: {hint}")
    return errors


def validate_ocr_hints_for_snippets(
    ocr_dir: Path,
    snippets_dir: Path,
) -> list[str]:
    """Return validation errors for per-page OCR hints (empty list if OK)."""
    if not ocr_dir.is_dir():
        return [f"OCR hint directory not found: {ocr_dir}"]

    try:
        snippets = list_snippet_pages(snippets_dir)
    except FileNotFoundError as exc:
        return [str(exc)]

    errors: list[str] = []
    for snippet in snippets:
        hint_file = find_ocr_hint_file(ocr_dir, snippet.stem)
        if hint_file is None:
            errors.append(
                f"OCR hint file missing for {snippet.stem} "
                f"(expected {ocr_dir}/{snippet.stem}.{{txt,md,docx}})"
            )
            continue
        if not load_ocr_hint_text(hint_file).strip():
            errors.append(f"OCR hint file is empty: {hint_file}")
    return errors


def validate_configured_sample_entry(
    args: Any,
    entry_dir: Path,
    snippets_dir: Path,
) -> list[str]:
    """Validate alphabet/OCR inputs for one entry given configured ``args``."""
    if not stage1_context_inputs_apply(args):
        return []

    if getattr(args, "strategy", None) == "mathpix_ocr":
        mathpix_dir = entry_dir / "mathpix"
        if not mathpix_dir.is_dir():
            return [f"Mathpix source directory not found: {mathpix_dir}"]
        return validate_mathpix_sources_for_snippets(mathpix_dir, snippets_dir)

    errors: list[str] = []
    if not getattr(args, "no_alphabet", False):
        if not args.alphabet:
            errors.append(
                f"alphabet required but missing: {entry_dir / 'alphabet.txt'}"
            )
        else:
            errors.extend(validate_alphabet_file(Path(args.alphabet)))
    if not getattr(args, "no_ocr_hint", False):
        if not args.ocr_text:
            errors.append(
                f"OCR hint required but missing: {entry_dir / 'mathpix'}/"
            )
        else:
            errors.extend(
                validate_ocr_hints_for_snippets(
                    Path(args.ocr_text),
                    snippets_dir,
                )
            )
    return errors


def preflight_validate_sample_entries(
    args: Any,
    entries: list[Path],
) -> list[tuple[str, list[str]]]:
    """Validate all sample entries before processing any pages."""
    if not stage1_context_inputs_apply(args):
        return []

    failures: list[tuple[str, list[str]]] = []
    for entry_dir in entries:
        snippets_dir = entry_dir / "snippets"
        if not snippets_dir.is_dir():
            continue
        configure_sample_entry_args(args, entry_dir)
        errors = validate_configured_sample_entry(args, entry_dir, snippets_dir)
        if errors:
            failures.append((entry_dir.name, errors))
    return failures


def report_entry_input_failures(
    entry_name: str,
    errors: list[str],
    *,
    experiment_name: str = "",
) -> None:
    """Print a summary and indicate the entry will be skipped."""
    exp_suffix = f" (experiment {experiment_name!r})" if experiment_name else ""
    print(
        f"\nWARNING: skipping entry {entry_name!r}{exp_suffix} — "
        "required alphabet and/or OCR hint inputs missing or empty:"
    )
    for err in errors:
        print(f"  - {err}")
    print("  Continuing to the next language.\n")
