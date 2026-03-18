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
from dictextractor.utils.io import save_to_tsv, save_to_json


_STRATEGIES = {
    "manual": ManualLLMExtraction,
    "join": JoinLLMExtraction,
    "two_stage": TwoStageLLMExtraction,
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_TEXT_EXTS  = {".txt", ".md", ".docx"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_intro(intro_path: Path) -> Tuple[str, List[str]]:
    """
    Load intro context from a file or directory.

    Returns:
        (intro_text, intro_image_paths)
        - intro_text: concatenated text from any .txt/.md/.docx files found
        - intro_image_paths: sorted list of image paths found
    """
    if intro_path.is_file():
        if intro_path.suffix.lower() in _IMAGE_EXTS:
            return "", [str(intro_path)]
        return _read_text_file(intro_path), []

    # Directory: collect all text and image files
    text_parts, image_paths = [], []
    for f in sorted(intro_path.iterdir()):
        if f.name.startswith((".", "~")):
            continue
        if f.suffix.lower() in _IMAGE_EXTS:
            image_paths.append(str(f))
        elif f.suffix.lower() in _TEXT_EXTS:
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


def _apply_preprocessing(image_path: str, mode: str, preprocess_dir: Path) -> str:
    if mode == "none":
        return image_path
    from dictextractor.preprocessing.pipeline import DictionaryPreprocessor
    preprocessor = DictionaryPreprocessor(image_path, output_dir=str(preprocess_dir))
    preprocessor.step1_convert_to_grayscale()
    if mode in ("deskew", "all"):
        preprocessor.step2_deskew()
    if mode in ("denoise", "all"):
        preprocessor.step3_denoise()
    if mode in ("contrast", "all"):
        preprocessor.step4_contrast_normalization()
    if mode in ("sharpen", "all"):
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

    # Input
    parser.add_argument(
        "--input-image", dest="input_image", required=True,
        help="Directory of page images (.png/.jpg/.jpeg)",
    )
    parser.add_argument(
        "--ocr-text", dest="ocr_text",
        help="Directory of OCR hint files (.docx/.txt/.md). "
             "Each file must share the same stem as its matching image "
             "(e.g. page_1.png → page_1.docx). Optional.",
    )

    # Output
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory. One <stem>.tsv (+ <stem>_stage1.txt for two_stage) per page.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="save_json",
        help="Also save a <stem>.json alongside each TSV.",
    )

    # Model / strategy
    parser.add_argument(
        "-m", "--model", default="gemini/gemini-3-flash-preview",
        help="Primary LLM model (Stage 1 for two_stage). Default: gemini/gemini-3-flash-preview",
    )
    parser.add_argument(
        "--structure-model", default=None,
        help="Stage 2 model for two_stage (defaults to --model).",
    )
    parser.add_argument(
        "--strategy", choices=list(_STRATEGIES.keys()), default="two_stage",
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

    # Preprocessing
    parser.add_argument(
        "--preprocess",
        choices=["none", "deskew", "denoise", "contrast", "sharpen", "all"],
        default="none",
        help="Preprocessing pipeline to apply to each image before extraction.",
    )

    # Batch control
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N images (useful for quick tests).",
    )
    parser.add_argument(
        "-p", "--page-offset", type=int, default=1,
        help="Page number assigned to the first image (increments per image).",
    )

    args = parser.parse_args()

    # ── Validate input dir ─────────────────────────────────────────────────────
    input_dir = Path(args.input_image)
    if not input_dir.is_dir():
        parser.error(f"--input-image must be a directory: {input_dir}")

    # ── Collect and sort images ────────────────────────────────────────────────
    images = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in _IMAGE_EXTS and not f.name.startswith(".")
    )
    if not images:
        print(f"No images found in {input_dir}")
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
            intro_text, intro_image_paths = _collect_intro(intro_path)
            print(f"Intro: {len(intro_text)} chars of text, {len(intro_image_paths)} images loaded.")

    # ── Output dir ─────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir = output_dir / ".preprocessed"

    # ── Strategy (instantiated once, shared across pages) ─────────────────────
    strategy = _build_strategy(args, intro_text, intro_image_paths)

    # ── Batch loop ─────────────────────────────────────────────────────────────
    total = len(images)
    skipped = 0
    processed = 0
    failed = 0

    print(f"\nFound {total} image(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Strategy: {args.strategy} | Model: {args.model}")
    print("=" * 60)

    for idx, image_file in enumerate(images):
        page_number = args.page_offset + idx
        page_dir = output_dir / image_file.stem
        out_tsv = page_dir / (image_file.stem + ".tsv")

        # ── Resume: skip already-processed pages ──────────────────────────────
        if out_tsv.exists():
            print(f"[{idx+1}/{total}] SKIP {image_file.name} → {page_dir.name}/ already exists")
            skipped += 1
            continue

        print(f"\n[{idx+1}/{total}] Processing: {image_file.name}  (page {page_number})")
        page_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Preprocessing
            preprocess_dir.mkdir(parents=True, exist_ok=True)
            image_path = _apply_preprocessing(str(image_file), args.preprocess, preprocess_dir)

            # OCR hint
            ocr_file = _find_ocr_file(ocr_dir, image_file.stem) if ocr_dir else None
            if ocr_dir and not ocr_file:
                print(f"  Note: no OCR hint found for {image_file.stem} in {ocr_dir}")
            ocr_result = _build_ocr_result(image_path, ocr_file)

            # Extract
            extract_kwargs = {}
            if args.strategy == "two_stage":
                extract_kwargs["stage1_output_path"] = str(
                    page_dir / (image_file.stem + "_stage1.tsv")
                )

            page = strategy.extract(
                ocr_result, image_path,
                page_number=page_number,
                **extract_kwargs,
            )

            # Save outputs
            save_to_tsv(page, str(out_tsv))
            if args.save_json:
                save_to_json(page, str(out_tsv.with_suffix(".json")))

            print(f"  → {len(page.entries)} entries saved to {page_dir.name}/{out_tsv.name}")
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
    out.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    cost_str = f"  Total estimated cost: ${total_cost:.4f}" if cost_available else ""
    print(f"Run usage saved → {out}{cost_str}")


if __name__ == "__main__":
    exit(main())
