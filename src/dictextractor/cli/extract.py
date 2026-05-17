"""
CLI: extract dictionary entries from a directory of page images.
Usage: python -m dictextractor.cli.extract [options]
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

from dictextractor.ocr.mathpix import MathpixBackend
from dictextractor.schemas.ocr_result import OCRPageResult
from dictextractor.extraction.llm_manual import ManualLLMExtraction
from dictextractor.extraction.llm_join import JoinLLMExtraction
from dictextractor.extraction.llm_two_stage import TwoStageLLMExtraction
from dictextractor.utils.io import save_to_json, json_to_tsv


_STRATEGIES = {
    "manual": ManualLLMExtraction,
    "join": JoinLLMExtraction,
    "two_stage": TwoStageLLMExtraction,
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_PDF_EXTS = {".pdf"}
_TEXT_EXTS = {".txt", ".md", ".docx"}
_PDF_RENDER_DPI = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_pdf_pages(
    pdf_path: Path, cache_dir: Path, dpi: int = _PDF_RENDER_DPI
) -> List[Path]:
    """Render each page of ``pdf_path`` to a PNG under ``cache_dir``.

    Returns the list of rendered image paths (one per page, in order).
    Cached outputs are reused when newer than the source PDF.
    """
    import pymupdf  # PyMuPDF; lazy-imported so image-only runs don't pay for it

    cache_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    results: List[Path] = []
    try:
        pdf_mtime = pdf_path.stat().st_mtime
        for page_index in range(doc.page_count):
            suffix = "" if doc.page_count == 1 else f"_p{page_index + 1}"
            out_path = cache_dir / f"{pdf_path.stem}{suffix}.png"
            if out_path.exists() and out_path.stat().st_mtime >= pdf_mtime:
                results.append(out_path)
                continue
            pix = doc.load_page(page_index).get_pixmap(dpi=dpi)
            pix.save(str(out_path))
            results.append(out_path)
    finally:
        doc.close()
    return results


def _materialize_page_inputs(
    input_dir: Path, cache_dir: Path, *, render_pdfs: bool
) -> List[Path]:
    """Return sorted page input paths from ``input_dir``.

    When ``render_pdfs`` is True, PDFs are rasterized to PNG in ``cache_dir``
    (required for cv2 preprocessing). When False, PDFs are passed through
    as-is to the LLM (Gemini accepts ``application/pdf`` inline data).
    """
    collected: List[Path] = []
    for f in sorted(input_dir.iterdir()):
        if f.name.startswith((".", "~")):
            continue
        suffix = f.suffix.lower()
        if suffix in _IMAGE_EXTS:
            collected.append(f)
        elif suffix in _PDF_EXTS:
            if render_pdfs:
                collected.extend(_render_pdf_pages(f, cache_dir))
            else:
                collected.append(f)
    return sorted(collected)


def _collect_intro(
    intro_path: Path, pdf_cache_dir: Path, *, render_pdfs: bool
) -> Tuple[str, List[str]]:
    """
    Load intro context from a file or directory. Supports images, PDFs,
    and text files.

    When ``render_pdfs`` is True, PDF intro pages are rendered to PNG via
    ``pdf_cache_dir``; otherwise they are passed through as PDF paths for
    the LLM to ingest directly.
    """

    def _as_vision_inputs(f: Path) -> List[str]:
        if render_pdfs:
            return [str(p) for p in _render_pdf_pages(f, pdf_cache_dir)]
        return [str(f)]

    if intro_path.is_file():
        suffix = intro_path.suffix.lower()
        if suffix in _IMAGE_EXTS:
            return "", [str(intro_path)]
        if suffix in _PDF_EXTS:
            return "", _as_vision_inputs(intro_path)
        return _read_text_file(intro_path), []

    text_parts, image_paths = [], []
    for f in sorted(intro_path.iterdir()):
        if f.name.startswith((".", "~")):
            continue
        suffix = f.suffix.lower()
        if suffix in _IMAGE_EXTS:
            image_paths.append(str(f))
        elif suffix in _PDF_EXTS:
            image_paths.extend(_as_vision_inputs(f))
        elif suffix in _TEXT_EXTS:
            text_parts.append(_read_text_file(f))

    return "\n\n".join(text_parts), image_paths


def _read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        from dictextractor.utils.io import read_docx_text

        return read_docx_text(str(path))
    return path.read_text(encoding="utf-8")


def _find_ocr_file(ocr_dir: Path, stem: str) -> Optional[Path]:
    """Find an OCR hint file in ocr_dir whose stem matches the image stem."""
    for ext in (".docx", ".txt", ".md"):
        candidate = ocr_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def _build_ocr_result(image_path: str, ocr_file: Optional[Path]) -> OCRPageResult:
    """Create an OCRPageResult, optionally populated from an OCR hint file."""
    if ocr_file:
        return MathpixBackend().run(image_path, ocr_file=str(ocr_file))
    return OCRPageResult(source_image=image_path, backend="none", blocks=[])


def _apply_preprocessing(image_path: str, enabled: bool, preprocess_dir: Path) -> str:
    """Run the full cv2 preprocessing pipeline when enabled; otherwise no-op."""
    if not enabled:
        return image_path
    from dictextractor.preprocessing.preprocess import DictionaryPreprocessor

    preprocessor = DictionaryPreprocessor(image_path, output_dir=str(preprocess_dir))
    preprocessor.step1_convert_to_grayscale()
    preprocessor.step2_deskew()
    preprocessor.step3_denoise()
    preprocessor.step4_contrast_normalization()
    preprocessor.step5_sharpen()
    out = preprocess_dir / f"preprocessed_{Path(image_path).name}"
    return preprocessor.save_result(str(out))


def _build_strategy(args, intro_text: str, intro_image_paths: List[str]):
    """Instantiate the correct extraction strategy."""
    if args.strategy == "manual":
        return ManualLLMExtraction(model=args.model)
    if args.strategy == "join":
        return JoinLLMExtraction(model=args.model)
    if args.strategy == "two_stage":
        return TwoStageLLMExtraction(
            transcribe_model=args.model,
            structure_model=args.structure_model or args.model,
            alphabet_path=args.alphabet or None,
            intro_text=intro_text,
            intro_image_paths=intro_image_paths,
            discover_extra_fields=getattr(args, "discover_extra_fields", False),
            stage2_reasoning_effort=getattr(args, "stage2_reasoning_effort", "medium"),
            stage1_guides=getattr(args, "stage1_guides_text", ""),
            stage2_guides=getattr(args, "stage2_guides_text", ""),
        )
    raise ValueError(f"Unknown strategy: {args.strategy}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch-extract dictionary entries from a directory of page images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two-stage on a directory of pages
  python -m dictextractor.cli.extract \\
    --strategy two_stage \\
    --model gemini/gemini-3-flash-preview \\
    --input-image assets/pages/ \\
    --ocr-text assets/mathpix/ \\
    --alphabet assets/alphabet.txt \\
    --intro assets/introduction/ \\
    --output outputs/run1/ \\
    --limit 5

  # Resume: already-processed pages are skipped automatically.
  # Re-run the same command and only missing pages will be processed.
        """,
    )

    # Input — single-entry mode
    parser.add_argument(
        "--input-image",
        dest="input_image",
        help="Directory of page images (.png/.jpg/.jpeg). "
        "Required unless --samples-dir is used.",
    )
    parser.add_argument(
        "--ocr-text",
        dest="ocr_text",
        help="Directory of OCR hint files (.docx/.txt/.md). "
        "Each file must share the same stem as its matching image "
        "(e.g. page_1.png → page_1.docx). Optional.",
    )

    # Batch mode — process every language subfolder under a samples root
    parser.add_argument(
        "--samples-dir",
        dest="samples_dir",
        help="Parent directory containing one subfolder per dictionary "
        "(e.g. assets/dictionaries/samples-2). When set, every "
        "subfolder is processed using its default layout "
        "(snippets/, introduction/, mathpix/, alphabet.txt) and "
        "outputs are written to {entry}/outputs/stage-1.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Optional list of language subfolder names to process when "
        "--samples-dir is used (e.g. --languages Armenian-English "
        "Yiddish-English). Defaults to every subfolder.",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory. Per page: <stem>.json (canonical) and "
        "<stem>.tsv (rendered from the JSON, with one column per discovered "
        "extra field). For two_stage runs also: <stem>_stage1.tsv. "
        "Required unless --samples-dir is used.",
    )
    # Model / strategy
    parser.add_argument(
        "-m",
        "--model",
        default="gemini/gemini-3-flash-preview",
        help="Primary LLM model (Stage 1 for two_stage). Default: gemini/gemini-3-flash-preview",
    )
    parser.add_argument(
        "--structure-model",
        default=None,
        help="Stage 2 model for two_stage (defaults to --model).",
    )
    parser.add_argument(
        "--strategy",
        choices=list(_STRATEGIES.keys()),
        default="two_stage",
        help="Extraction strategy (default: two_stage).",
    )

    # Two-stage specific
    parser.add_argument(
        "--alphabet",
        help="Alphabet/legend file (.txt/.md) or image (.png/.jpg). "
        "Sent to Stage 1 to prime the character inventory.",
    )
    parser.add_argument(
        "--intro",
        help="Dictionary introduction/preface — a file (.txt/.md/.docx) or a directory. "
        "Text files are embedded in the Stage 2 system prompt; "
        "images are sent as vision context. Loaded once, shared across all pages.",
    )
    parser.add_argument(
        "--discover-extra-fields",
        action="store_true",
        dest="discover_extra_fields",
        help="Stage 2 only. Instruct the LLM to also extract any "
        "structurally-marked fields beyond the canonical schema "
        "(etymology, IPA, plural form, gender, register, cross-refs, etc.) "
        "into each entry's `extra_fields` map. Off by default.",
    )
    parser.add_argument(
        "--reasoning",
        choices=["low", "medium", "high"],
        default="medium",
        dest="stage2_reasoning_effort",
        help="Reasoning effort for the Stage 2 LLM call (default: medium). "
        "High reasoning has been observed to leak chain-of-thought into "
        "JSON string fields on dense pages — drop to low for problematic "
        "inputs, bump to high only when needed.",
    )
    parser.add_argument(
        "--stage-1-guides",
        dest="stage1_guides_path",
        help="Path to a .txt/.md/.docx file of extra rules appended verbatim to "
        "the Stage 1 user prompt under a 'USER DEFINED GUIDELINES' header. "
        "Optional — leave unset to use the default prompt.",
    )
    parser.add_argument(
        "--stage-2-guides",
        dest="stage2_guides_path",
        help="Path to a .txt/.md/.docx file of extra rules appended verbatim to "
        "the Stage 2 user prompt under a 'USER DEFINED GUIDELINES' header. "
        "Optional — leave unset to use the default prompt.",
    )

    # Preprocessing — off by default. When on, PDFs are rendered to PNG first
    # (cv2 can't read PDFs); when off, PDFs flow straight to the LLM as
    # application/pdf inline data.
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable the full cv2 preprocessing pipeline "
        "(grayscale → deskew → denoise → contrast → sharpen). Off by default.",
    )

    # Batch control
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images (useful for quick tests).",
    )
    parser.add_argument(
        "-p",
        "--page-offset",
        type=int,
        default=1,
        help="Page number assigned to the first image (increments per image).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process pages even if output already exists (disables resume).",
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "both"],
        default="both",
        help="Run only stage 1, only stage 2, or both (default: both). "
        "Stage-2-only requires existing Stage 1 TSV in the output directory.",
    )

    args = parser.parse_args()

    # ── Load user-defined guides (if any) once, shared across all pages ──────
    args.stage1_guides_text = ""
    if getattr(args, "stage1_guides_path", None):
        p = Path(args.stage1_guides_path)
        if not p.exists():
            parser.error(f"--stage-1-guides path not found: {p}")
        args.stage1_guides_text = _read_text_file(p)
    args.stage2_guides_text = ""
    if getattr(args, "stage2_guides_path", None):
        p = Path(args.stage2_guides_path)
        if not p.exists():
            parser.error(f"--stage-2-guides path not found: {p}")
        args.stage2_guides_text = _read_text_file(p)

    # ── Dispatch: samples-dir batch mode vs. single-entry mode ────────────────
    if args.samples_dir:
        return _run_samples_dir(args, parser)

    if not args.input_image or not args.output:
        parser.error(
            "--input-image and --output are required unless --samples-dir is used."
        )

    return _run_single_entry(args, parser)


def _run_samples_dir(args, parser) -> int:
    """Iterate over every language subfolder under ``args.samples_dir``."""
    samples_root = Path(args.samples_dir)
    if not samples_root.is_dir():
        parser.error(f"--samples-dir must be a directory: {samples_root}")

    all_entries = sorted(p for p in samples_root.iterdir() if p.is_dir())
    if args.languages:
        requested = set(args.languages)
        available = {p.name for p in all_entries}
        missing = requested - available
        if missing:
            parser.error(
                f"--languages references unknown subfolders: {sorted(missing)}. "
                f"Available: {sorted(available)}"
            )
        entries = [p for p in all_entries if p.name in requested]
    else:
        entries = all_entries

    if not entries:
        print(f"No entry subfolders found under {samples_root}")
        return 1

    print(
        f"Batch mode: processing {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} under {samples_root}"
    )

    any_failure = False
    for entry_dir in entries:
        snippets_dir = entry_dir / "snippets"
        if not snippets_dir.is_dir():
            print(f"[skip] {entry_dir.name}: no snippets/ folder")
            continue

        intro_dir = entry_dir / "introduction"
        mathpix_dir = entry_dir / "mathpix"
        alphabet_file = entry_dir / "alphabet.txt"
        output_dir = entry_dir / "outputs"

        args.input_image = str(snippets_dir)
        args.ocr_text = str(mathpix_dir) if mathpix_dir.is_dir() else None
        args.intro = str(intro_dir) if intro_dir.is_dir() else None
        args.alphabet = str(alphabet_file) if alphabet_file.exists() else None
        args.output = str(output_dir)

        print("\n" + "#" * 60)
        print(f"# Entry: {entry_dir.name}")
        print("#" * 60)
        rc = _run_single_entry(args, parser)
        if rc != 0:
            any_failure = True

    return 1 if any_failure else 0


def _run_single_entry(args, parser) -> int:
    """Run extraction for a single entry (one --input-image directory)."""
    # ── Validate input dir ─────────────────────────────────────────────────────
    input_dir = Path(args.input_image)
    if not input_dir.is_dir():
        parser.error(f"--input-image must be a directory: {input_dir}")

    # ── Output dir (set up early so we can cache rendered PDF pages) ──────────
    # Stage 1 artifacts (transcription TSV + raw/input JSONs) live under
    # <output>/stage-1/<page>/.  Stage 2 artifacts (final TSV + JSON +
    # raw/input/usage JSONs) live under <output>/stage-2/<page>/.
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = output_dir / "stage-1"
    stage2_dir = output_dir / "stage-2"
    preprocess_dir = output_dir / ".preprocessed"
    snippets_cache_dir = output_dir / ".rendered_snippets"
    intro_cache_dir = output_dir / ".rendered_intro"

    # ── Collect snippet pages (images + PDFs; render only when preprocessing) ─
    images = _materialize_page_inputs(
        input_dir, snippets_cache_dir, render_pdfs=args.preprocess
    )
    if not images:
        print(f"No images or PDFs found in {input_dir}")
        return 1

    if args.limit:
        images = images[: args.limit]

    # ── OCR text dir ───────────────────────────────────────────────────────────
    ocr_dir: Optional[Path] = None
    if args.ocr_text:
        ocr_dir = Path(args.ocr_text)
        if not ocr_dir.is_dir():
            parser.error(f"--ocr-text must be a directory: {ocr_dir}")

    # ── Intro (loaded once for all pages) ─────────────────────────────────────
    intro_text, intro_image_paths = "", []
    if args.intro:
        intro_path = Path(args.intro)
        if not intro_path.exists():
            print(f"Warning: --intro path not found: {args.intro}")
        else:
            intro_text, intro_image_paths = _collect_intro(
                intro_path, intro_cache_dir, render_pdfs=args.preprocess
            )
            print(
                f"Intro: {len(intro_text)} chars of text, {len(intro_image_paths)} images loaded."
            )

    # ── Strategy (instantiated once, shared across pages) ─────────────────────
    strategy = _build_strategy(args, intro_text, intro_image_paths)

    # ── Batch loop ─────────────────────────────────────────────────────────────
    total = len(images)
    skipped = 0
    processed = 0
    failed = 0

    print(f"\nFound {total} image(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(
        f"Strategy: {args.strategy} | Model: {args.model} | Stage: {args.stage} | Overwrite: {args.overwrite}"
    )
    print("=" * 60)

    for idx, image_file in enumerate(images):
        page_number = args.page_offset + idx
        stem = image_file.stem
        stage1_page_dir = stage1_dir / stem
        stage2_page_dir = stage2_dir / stem
        stage1_tsv = stage1_page_dir / (stem + "_stage1.tsv")
        out_tsv = stage2_page_dir / (stem + ".tsv")

        # ── Resume: skip already-processed pages ──────────────────────────────
        if not args.overwrite:
            if args.stage == "both" and out_tsv.exists():
                print(
                    f"[{idx+1}/{total}] SKIP {image_file.name} → stage-2/{stem}/ already exists"
                )
                skipped += 1
                continue
            if args.stage == "1" and stage1_tsv.exists():
                print(
                    f"[{idx+1}/{total}] SKIP {image_file.name} → stage1 already exists"
                )
                skipped += 1
                continue
            if args.stage == "2" and out_tsv.exists():
                print(
                    f"[{idx+1}/{total}] SKIP {image_file.name} → stage2 already exists"
                )
                skipped += 1
                continue

        # ── Stage-2-only: verify stage 1 output exists ───────────────────────
        if args.stage == "2" and not stage1_tsv.exists():
            print(
                f"[{idx+1}/{total}] SKIP {image_file.name} → no stage1 TSV at {stage1_tsv}"
            )
            skipped += 1
            continue

        print(
            f"\n[{idx+1}/{total}] Processing: {image_file.name}  (page {page_number})"
        )
        if args.stage in ("1", "both"):
            stage1_page_dir.mkdir(parents=True, exist_ok=True)
        if args.stage in ("2", "both"):
            stage2_page_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Preprocessing
            preprocess_dir.mkdir(parents=True, exist_ok=True)
            image_path = _apply_preprocessing(
                str(image_file), args.preprocess, preprocess_dir
            )

            # OCR hint
            ocr_file = _find_ocr_file(ocr_dir, image_file.stem) if ocr_dir else None
            if ocr_dir and not ocr_file:
                print(f"  Note: no OCR hint found for {image_file.stem} in {ocr_dir}")
            ocr_result = _build_ocr_result(image_path, ocr_file)

            # Extract
            extract_kwargs = {}
            if args.strategy == "two_stage":
                extract_kwargs["stage1_output_path"] = str(stage1_tsv)
                extract_kwargs["stage2_output_path"] = str(out_tsv)
                extract_kwargs["run_stage"] = args.stage

            page = strategy.extract(
                ocr_result,
                image_path,
                page_number=page_number,
                **extract_kwargs,
            )

            # Save outputs (skip for stage-1-only since there are no entries)
            if args.stage != "1":
                out_json = out_tsv.with_suffix(".json")
                save_to_json(page, str(out_json))
                json_to_tsv(str(out_json), str(out_tsv))
                print(
                    f"  → {len(page.entries)} entries saved to stage-2/{stem}/{out_tsv.name}"
                )

            processed += 1

        except Exception as exc:
            print(f"  ERROR processing {image_file.name}: {exc}")
            import traceback

            traceback.print_exc()
            failed += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Done. {processed} processed, {skipped} skipped (resume), {failed} failed.")

    # Aggregate usage across all pages for this run
    if args.strategy == "two_stage" and processed > 0:
        _write_run_usage(output_dir)

    return 0 if failed == 0 else 1


def _write_run_usage(output_dir: Path) -> None:
    """Collect all per-page usage.json files and write a run-level summary."""
    import json

    pages = []
    total_cost = 0.0
    cost_available = False

    for usage_file in sorted(output_dir.rglob("*_usage.json")):
        data = json.loads(usage_file.read_text(encoding="utf-8"))
        page_cost = data.get("total_cost_usd")
        if page_cost is not None:
            total_cost += page_cost
            cost_available = True
        pages.append({"page": usage_file.parent.name, **data})

    run_summary = {
        "pages": pages,
        "run_total_cost_usd": round(total_cost, 8) if cost_available else None,
    }

    out = output_dir / "run_usage.json"
    out.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cost_str = f"  Total estimated cost: ${total_cost:.4f}" if cost_available else ""
    print(f"Run usage saved → {out}{cost_str}")


if __name__ == "__main__":
    exit(main())
