#!/usr/bin/env python3
"""Validate stage-1 flat experiment outputs and OCR-hint usage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dictextractor.ocr.vlm.page_inputs import list_snippet_pages
from dictextractor.ocr.vlm.prompts import find_ocr_hint_file

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"}


def _snippet_stems(snippets_dir: Path) -> list[str]:
    return sorted(p.stem for p in snippets_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def validate_language(
    samples_dir: Path,
    language: str,
    experiment_name: str,
    *,
    stage1_subdir: str = "stage-1",
) -> list[str]:
    """Return validation errors for one language (empty if OK)."""
    entry = samples_dir / language
    snippets_dir = entry / "snippets"
    mathpix_dir = entry / "mathpix"
    exp_dir = entry / "outputs" / stage1_subdir / experiment_name
    errors: list[str] = []

    if not snippets_dir.is_dir():
        return [f"{language}: snippets dir missing"]

    try:
        stems = [p.stem for p in list_snippet_pages(snippets_dir)]
    except FileNotFoundError as exc:
        return [f"{language}: {exc}"]

    run_config_path = exp_dir / "run_config.json"
    if not run_config_path.is_file():
        errors.append(f"{language}: missing run_config.json")
    else:
        cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
        ocr_hint = cfg.get("ocr_hint", {})
        if not ocr_hint.get("used"):
            errors.append(f"{language}: run_config ocr_hint.used is false")
        per_page = {row["stem"]: row.get("ocr_hint_file") for row in cfg.get("per_page", [])}
        for stem in stems:
            hint_path = per_page.get(stem)
            if not hint_path:
                errors.append(f"{language}/{stem}: no ocr_hint_file in run_config")
                continue
            hint = Path(hint_path)
            if not hint.is_file():
                errors.append(f"{language}/{stem}: ocr_hint_file missing: {hint_path}")
            elif hint.suffix.lower() != ".md":
                errors.append(
                    f"{language}/{stem}: ocr_hint_file is {hint.suffix}, expected .md: {hint_path}"
                )
            expected = find_ocr_hint_file(mathpix_dir, stem)
            if expected and hint.resolve() != expected.resolve():
                errors.append(
                    f"{language}/{stem}: ocr_hint_file {hint.name} != expected {expected.name}"
                )

    for stem in stems:
        page_dir = exp_dir / stem
        flat_path = page_dir / f"{stem}_stage1_flat.txt"
        input_path = page_dir / f"{stem}_stage1_input.json"
        if not flat_path.is_file() or flat_path.stat().st_size == 0:
            errors.append(f"{language}/{stem}: missing or empty flat output")
        if not input_path.is_file():
            errors.append(f"{language}/{stem}: missing stage1_input.json")
            continue
        messages = json.loads(input_path.read_text(encoding="utf-8"))
        user_content = next(
            (m for m in messages if m.get("role") == "user"),
            None,
        )
        if not user_content:
            errors.append(f"{language}/{stem}: no user message in stage1_input.json")
            continue
        text_parts: list[str] = []
        content = user_content.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
        joined = "\n".join(text_parts)
        if "<ocr_reference>" not in joined:
            errors.append(f"{language}/{stem}: stage1_input missing <ocr_reference>")
        if ".md" not in joined and "<ocr_reference>" in joined:
            # OCR text is injected; md path may not appear in prompt — OK.
            pass

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--stage1-output-subdir",
        default="stage-1",
        help="Parent folder under outputs/ for Stage-1 artifacts (default: stage-1).",
    )
    parser.add_argument("--languages", nargs="+", required=True)
    args = parser.parse_args()

    all_errors: list[str] = []
    for language in args.languages:
        all_errors.extend(
            validate_language(
                args.samples_dir,
                language,
                args.experiment_name,
                stage1_subdir=args.stage1_output_subdir,
            )
        )

    if all_errors:
        print(f"VALIDATION FAILED ({len(all_errors)} issue(s)):", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    n_pages = 0
    for language in args.languages:
        snippets_dir = args.samples_dir / language / "snippets"
        if snippets_dir.is_dir():
            n_pages += len(_snippet_stems(snippets_dir))
    print(
        f"VALIDATION OK: {args.experiment_name} — "
        f"{len(args.languages)} languages, {n_pages} pages, all .md OCR hints"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
