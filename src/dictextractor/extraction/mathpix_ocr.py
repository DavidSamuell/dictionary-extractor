"""Stage-1 batch extraction from pre-generated Mathpix Convert markdown files."""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dictextractor.extraction.sample_entry import (
    configure_sample_entry_args,
    report_entry_input_failures,
    validate_mathpix_sources_for_snippets,
)
from dictextractor.ocr.adapters.flat_export import write_stage1_flat_for_page
from dictextractor.ocr.adapters.markdown_to_flat import find_markdown_source
from dictextractor.ocr.adapters.mathpix_flat import (
    mathpix_docx_path,
    mathpix_lines_path,
    mathpix_page_is_complete,
)
from dictextractor.ocr.vlm.page_inputs import list_snippet_pages
from dictextractor.ocr.vlm.prompts import find_ocr_hint_file

logger = logging.getLogger(__name__)


def _git_short_sha() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _write_run_config(target_dir: Path, manifest: dict[str, Any], *, force: bool) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "run_config.json"
    if not force and path.exists():
        print(
            f"  Keeping existing {path} (resume; pass --overwrite to refresh it)."
        )
        return
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_mathpix_manifest(
    args: Any,
    snippets_dir: Path,
    snippets: list[Path],
    mathpix_dir: Path,
) -> dict[str, Any]:
    from dictextractor.evaluation.stage1.flatten import FLAT_SPEC_VERSION

    return {
        "stage": "1",
        "experiment_name": args.experiment_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": "mathpix_ocr",
        "git_sha": _git_short_sha(),
        "ocr_backend": "mathpix-convert",
        "product_label": "Mathpix-OCR",
        "flat_spec_version": FLAT_SPEC_VERSION,
        "adapter": "markdown_to_flat_v1",
        "adapter_preference": "markdown when present, else lines.json/docx geometry",
        "alphabet": {
            "used": False,
            "path": None,
        },
        "ocr_hint": {
            "used": False,
            "dir": None,
        },
        "mathpix_source_dir": str(mathpix_dir),
        "inputs": {
            "snippets_dir": str(snippets_dir),
            "page_count": len(snippets),
        },
        "per_page": [
            {
                "stem": s.stem,
                "snippet_path": str(s),
                "mathpix_source": str(find_ocr_hint_file(mathpix_dir, s.stem) or ""),
                "mathpix_lines_json": str(
                    mathpix_dir / f"{s.stem}.lines.json"
                    if (mathpix_dir / f"{s.stem}.lines.json").is_file()
                    else ""
                ),
            }
            for s in snippets
        ],
    }


def run_mathpix_ocr_entry(
    args: Any,
    input_dir: Path,
    output_dir: Path,
    *,
    mathpix_dir: Path,
) -> int:
    """Materialize Mathpix markdown artifacts and flat preds for one language entry."""
    snippets_dir = input_dir
    snippets = list_snippet_pages(snippets_dir)
    if args.limit:
        snippets = snippets[: args.limit]

    stage1_dir = output_dir / "stage-1" / args.experiment_name
    _write_run_config(
        stage1_dir,
        _build_mathpix_manifest(args, snippets_dir, snippets, mathpix_dir),
        force=args.overwrite,
    )

    total = len(snippets)
    skipped = processed = failed = 0

    print(f"\nFound {total} snippet(s) in {snippets_dir}")
    print(
        f"Mathpix OCR | Output: {stage1_dir} | Source: {mathpix_dir} | "
        f"Alphabet: off | OCR hint: off (source OCR, not prompt hint)"
    )

    for idx, snippet in enumerate(snippets):
        stem = snippet.stem
        page_dir = stage1_dir / stem
        source_hint = find_ocr_hint_file(mathpix_dir, stem)
        source_lines = mathpix_dir / f"{stem}.lines.json"

        if not args.overwrite and mathpix_page_is_complete(page_dir, stem=stem):
            print(f"[{idx + 1}/{total}] SKIP {snippet.name} (already complete)")
            skipped += 1
            continue

        if source_hint is None and not source_lines.is_file():
            print(
                f"[{idx + 1}/{total}] ERROR {snippet.name}: "
                f"no Mathpix md/docx or lines.json in {mathpix_dir}"
            )
            failed += 1
            continue

        print(f"\n[{idx + 1}/{total}] Processing: {snippet.name}")
        page_dir.mkdir(parents=True, exist_ok=True)

        try:
            started = time.perf_counter()
            if source_hint is not None:
                if source_hint.suffix.lower() in {".md", ".mmd"}:
                    for name in (f"{stem}.md", "output.md"):
                        target_md = page_dir / name
                        if source_hint.resolve() != target_md.resolve():
                            shutil.copy2(source_hint, target_md)
                elif source_hint.suffix.lower() == ".docx":
                    if find_markdown_source(mathpix_dir, stem=stem) is None:
                        target_docx = mathpix_docx_path(page_dir)
                        if source_hint.resolve() != target_docx.resolve():
                            shutil.copy2(source_hint, target_docx)
            if source_lines.is_file():
                target_lines = mathpix_lines_path(page_dir)
                if source_lines.resolve() != target_lines.resolve():
                    shutil.copy2(source_lines, target_lines)

            flat_path = write_stage1_flat_for_page(page_dir, stem=stem)
            flat_lines = flat_path.read_text(encoding="utf-8", errors="replace").splitlines()
            adapter = "markdown" if find_markdown_source(page_dir, stem=stem) else "geometry"
            preview = "\n".join(flat_lines[:3])
            result_path = page_dir / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "adapter": adapter,
                        "source_hint": str(source_hint) if source_hint else None,
                        "source_lines_json": (
                            str(source_lines) if source_lines.is_file() else None
                        ),
                        "line_count": len(flat_lines),
                        "preview": preview,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            elapsed = time.perf_counter() - started
            artifact_names = [flat_path.name, result_path.name]
            md_source = find_markdown_source(page_dir, stem=stem)
            if md_source is not None:
                artifact_names.insert(0, md_source.name)
            if mathpix_lines_path(page_dir).is_file():
                artifact_names.insert(0, mathpix_lines_path(page_dir).name)
            if mathpix_docx_path(page_dir).is_file():
                artifact_names.insert(0, mathpix_docx_path(page_dir).name)
            print(
                f"  Done in {elapsed:.1f}s -> {page_dir} "
                f"({', '.join(artifact_names)}; adapter={adapter}; "
                f"{len(flat_lines)} lines)"
            )
            processed += 1
        except OSError as exc:
            logger.exception("Mathpix OCR failed for %s", snippet.name)
            print(f"  ERROR: {exc}")
            failed += 1

    print(
        f"\nMathpix OCR summary: {processed} processed, {skipped} skipped, "
        f"{failed} failed."
    )
    return 0 if failed == 0 else 1


def run_mathpix_ocr_batch(args: Any, entries: list[Path]) -> int:
    """Process multiple language entries from existing ``mathpix/`` folders."""
    any_failure = False
    for entry_dir in entries:
        snippets_dir = entry_dir / "snippets"
        if not snippets_dir.is_dir():
            print(f"[skip] {entry_dir.name}: no snippets/ folder")
            continue

        configure_sample_entry_args(args, entry_dir)
        mathpix_dir = entry_dir / "mathpix"
        input_errors = validate_mathpix_sources_for_snippets(
            mathpix_dir,
            snippets_dir,
        )
        if input_errors:
            report_entry_input_failures(
                entry_dir.name,
                input_errors,
                experiment_name=getattr(args, "experiment_name", ""),
            )
            any_failure = True
            continue

        print("\n" + "#" * 60)
        print(f"# Entry: {entry_dir.name}")
        print("#" * 60)
        rc = run_mathpix_ocr_entry(
            args,
            snippets_dir,
            entry_dir / "outputs",
            mathpix_dir=mathpix_dir,
        )
        if rc != 0:
            any_failure = True
    return 1 if any_failure else 0
