"""Stage-1 batch extraction using specialized OCR/VLM models."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dictextractor.ocr.adapters.flat_export import write_stage1_flat_for_page
from dictextractor.ocr.vlm.page_inputs import list_snippet_pages, materialize_page_image
from dictextractor.ocr.vlm.runner import VlmOcrRunner, create_vlm_runner, page_is_complete

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


def _build_vlm_manifest(
    args: Any,
    snippets_dir: Path,
    snippets: list[Path],
    spec: Any,
) -> dict[str, Any]:
    return {
        "stage": "1",
        "experiment_name": args.experiment_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": "vlm_ocr",
        "git_sha": _git_short_sha(),
        "vlm_model": spec.key,
        "model_id": spec.model_id,
        "product_label": spec.product_label,
        "vlm_dpi": getattr(args, "vlm_dpi", 200),
        "glm_ocr_prompt": getattr(args, "glm_ocr_prompt", None),
        "inputs": {
            "snippets_dir": str(snippets_dir),
            "page_count": len(snippets),
        },
        "per_page": [
            {"stem": s.stem, "snippet_path": str(s)} for s in snippets
        ],
    }


def run_vlm_ocr_entry(
    args: Any,
    input_dir: Path,
    output_dir: Path,
    runner: VlmOcrRunner,
) -> int:
    """Run VLM OCR on all snippets in ``input_dir`` for one dictionary entry.

    Args:
        args: Parsed CLI namespace.
        input_dir: ``snippets/`` directory.
        output_dir: Entry ``outputs/`` directory.
        runner: Loaded VLM runner.

    Returns:
        Exit code 0 on success, 1 if any page failed.
    """
    snippets_dir = input_dir
    snippets = list_snippet_pages(snippets_dir)
    if args.limit:
        snippets = snippets[: args.limit]

    stage1_dir = output_dir / "stage-1" / args.experiment_name
    render_cache = output_dir / ".rendered_snippets"
    dpi = getattr(args, "vlm_dpi", 200)

    _write_run_config(
        stage1_dir,
        _build_vlm_manifest(args, snippets_dir, snippets, runner.spec),
        force=args.overwrite,
    )

    total = len(snippets)
    skipped = processed = failed = 0

    print(f"\nFound {total} snippet(s) in {snippets_dir}")
    print(f"VLM: {runner.spec.product_label} | Output: {stage1_dir}")

    for idx, snippet in enumerate(snippets):
        stem = snippet.stem
        page_dir = stage1_dir / stem

        if not args.overwrite and page_is_complete(runner, page_dir, stem=stem):
            flat_path = page_dir / f"{stem}_stage1_flat.txt"
            if not flat_path.is_file():
                flat_path = write_stage1_flat_for_page(page_dir, stem=stem)
                print(
                    f"[{idx + 1}/{total}] SKIP {snippet.name} "
                    f"(already complete; backfilled {flat_path.name})"
                )
            else:
                print(f"[{idx + 1}/{total}] SKIP {snippet.name} (already complete)")
            skipped += 1
            continue

        print(f"\n[{idx + 1}/{total}] Processing: {snippet.name}")
        page_dir.mkdir(parents=True, exist_ok=True)

        try:
            started = time.perf_counter()
            image_path = materialize_page_image(
                snippet, render_cache / runner.spec.key, dpi=dpi
            )
            input_copy = page_dir / "input.png"
            if image_path.resolve() != input_copy.resolve():
                import shutil

                shutil.copy2(image_path, input_copy)

            artifacts = runner.run_page(image_path, page_dir, stem=stem)
            flat_path = write_stage1_flat_for_page(page_dir, stem=stem)
            elapsed = time.perf_counter() - started
            print(
                f"  Done in {elapsed:.1f}s -> {page_dir} "
                f"({', '.join(Path(v).name for v in artifacts.values())}, "
                f"{flat_path.name})"
            )
            processed += 1
        except Exception as exc:
            logger.exception("VLM OCR failed for %s", snippet.name)
            print(f"  ERROR: {exc}")
            failed += 1

    print(
        f"\nVLM OCR summary: {processed} processed, {skipped} skipped, {failed} failed."
    )
    return 0 if failed == 0 else 1


def run_vlm_ocr_batch(args: Any, entries: list[Path]) -> int:
    """Process multiple language entries with one loaded VLM."""
    runner = create_vlm_runner(
        args.vlm_model,
        glm_prompt=getattr(args, "glm_ocr_prompt", None),
        glm_max_new_tokens=getattr(args, "glm_max_new_tokens", None),
    )
    runner.load()
    any_failure = False
    try:
        for entry_dir in entries:
            snippets_dir = entry_dir / "snippets"
            if not snippets_dir.is_dir():
                print(f"[skip] {entry_dir.name}: no snippets/ folder")
                continue
            output_dir = entry_dir / "outputs"
            print("\n" + "#" * 60)
            print(f"# Entry: {entry_dir.name}")
            print("#" * 60)
            rc = run_vlm_ocr_entry(
                args,
                snippets_dir,
                output_dir,
                runner,
            )
            if rc != 0:
                any_failure = True
    finally:
        runner.unload()
    return 1 if any_failure else 0
