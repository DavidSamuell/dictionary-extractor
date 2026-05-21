"""
CLI: extract dictionary entries from a directory of page images.
Usage: python -m dictextractor.cli.extract [options]
"""

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dictextractor.ocr.mathpix import MathpixBackend
from dictextractor.schemas.ocr_result import OCRPageResult
from dictextractor.extraction.llm_manual import ManualLLMExtraction
from dictextractor.extraction.llm_join import JoinLLMExtraction
from dictextractor.extraction.llm_two_stage import TwoStageLLMExtraction
from dictextractor.extraction.vlm_ocr import run_vlm_ocr_batch, run_vlm_ocr_entry
from dictextractor.ocr.vlm.registry import get_vlm_spec, list_vlm_keys
from dictextractor.ocr.vlm.runner import create_vlm_runner
from dictextractor.utils.io import save_to_json, json_to_tsv


_STRATEGIES = {
    "manual": ManualLLMExtraction,
    "join": JoinLLMExtraction,
    "two_stage": TwoStageLLMExtraction,
}

_STRATEGY_CHOICES = list(_STRATEGIES.keys()) + ["vlm_ocr"]

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


_IMAGE_ALPHABET_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _git_short_sha() -> Optional[str]:
    """Best-effort short git SHA of the working tree; None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        sha = out.stdout.strip()
        return sha or None
    except (OSError, subprocess.SubprocessError):
        return None


def _alphabet_manifest_entry(alphabet_path: Optional[str]) -> Dict[str, Any]:
    """Describe the alphabet input for a run manifest.

    Text-form alphabets (.txt/.md/.docx/etc.) are embedded inline so the
    exact content used by the run is preserved even if the source file is
    later edited or deleted. Image-form alphabets are referenced by path
    only (you swap them, you don't edit them).
    """
    if not alphabet_path:
        return {"used": False, "path": None, "kind": None, "text": None}
    p = Path(alphabet_path)
    if p.suffix.lower() in _IMAGE_ALPHABET_EXTS:
        return {"used": True, "path": str(p), "kind": "image", "text": None}
    try:
        text = _read_text_file(p)
    except OSError as exc:
        return {
            "used": True, "path": str(p), "kind": "text",
            "text": None, "read_error": str(exc),
        }
    return {"used": True, "path": str(p), "kind": "text", "text": text}


def _guides_manifest_entry(
    path: Optional[str], loaded_text: str
) -> Dict[str, Any]:
    """Describe an inline guides file (stage-1 or stage-2 guides)."""
    if not path:
        return {"used": False, "path": None, "text": None}
    return {"used": True, "path": path, "text": loaded_text or ""}


def _per_page_inputs_stage1(
    images: List[Path], ocr_dir: Optional[Path]
) -> List[Dict[str, Any]]:
    """Resolve the per-page input bundle for stage 1 (snippet + ocr-hint)."""
    rows: List[Dict[str, Any]] = []
    for image_file in images:
        stem = image_file.stem
        ocr_file = _find_ocr_file(ocr_dir, stem) if ocr_dir else None
        rows.append({
            "stem": stem,
            "snippet_path": str(image_file),
            "ocr_hint_file": str(ocr_file) if ocr_file else None,
        })
    return rows


def _per_page_inputs_stage2(
    images: List[Path], stage1_dir: Path
) -> List[Dict[str, Any]]:
    """Resolve the per-page stage-1 TSV that stage 2 consumes."""
    rows: List[Dict[str, Any]] = []
    for image_file in images:
        stem = image_file.stem
        tsv = stage1_dir / stem / f"{stem}_stage1.tsv"
        rows.append({"stem": stem, "stage1_tsv_path": str(tsv)})
    return rows


def _write_run_config(
    target_dir: Path, manifest: Dict[str, Any], *, force: bool
) -> None:
    """Write a run_config.json into ``target_dir`` honoring the resume guard.

    On resume (file exists, ``force`` False) the existing manifest wins so
    the on-disk config never drifts from what produced the predictions
    sitting in the slot. With ``force=True`` (i.e. ``--overwrite``) the
    manifest is rewritten to match the fresh invocation.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "run_config.json"
    if not force and path.exists():
        print(
            f"  Keeping existing {path} (resume; pass --overwrite to refresh it)."
        )
        return
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _build_stage1_manifest(
    args,
    snippets_dir: Path,
    images: List[Path],
    ocr_dir: Optional[Path],
) -> Dict[str, Any]:
    """Assemble the stage-1 manifest dict (no I/O)."""
    from dictextractor.evaluation.stage1.flatten import FLAT_SPEC_VERSION

    return {
        "stage": "1",
        "experiment_name": args.experiment_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": args.strategy,
        "stage1_mode": getattr(args, "stage1_mode", "column"),
        "flat_spec_version": FLAT_SPEC_VERSION,
        "git_sha": _git_short_sha(),
        "model": args.model,
        "reasoning_effort": args.stage1_reasoning_effort,
        "preprocess": bool(args.preprocess),
        "alphabet": _alphabet_manifest_entry(args.alphabet),
        "ocr_hint": {
            "used": bool(ocr_dir),
            "dir": str(ocr_dir) if ocr_dir else None,
        },
        "stage1_guides": _guides_manifest_entry(
            getattr(args, "stage1_guides_path", None),
            getattr(args, "stage1_guides_text", ""),
        ),
        "inputs": {
            "snippets_dir": str(snippets_dir),
            "page_count": len(images),
        },
        "per_page": _per_page_inputs_stage1(images, ocr_dir),
    }


def _build_stage2_manifest(
    args,
    snippets_dir: Path,
    images: List[Path],
    stage1_dir: Path,
    intro_image_paths: List[str],
) -> Dict[str, Any]:
    """Assemble the stage-2 manifest dict (no I/O)."""
    return {
        "stage": "2",
        "experiment_name": args.stage2_experiment_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": args.strategy,
        "git_sha": _git_short_sha(),
        "model": args.structure_model or args.model,
        "reasoning_effort": args.stage2_reasoning_effort,
        "discover_extra_fields": bool(
            getattr(args, "discover_extra_fields", False)
        ),
        "stage2_output_format": "mdf",
        "stage1_source": {
            "experiment_name": args.experiment_name,
            "stage1_dir": str(stage1_dir),
            "samples_dir": getattr(args, "samples_dir", None),
        },
        # Intro is path-only regardless of format (image / pdf / txt / md /
        # docx) per the user's explicit choice — embedding intro PDFs/images
        # would balloon the manifest, and intro text rarely changes mid-sweep.
        "intro": {
            "used": bool(args.intro),
            "source_path": args.intro,
            "resolved_image_or_pdf_paths": list(intro_image_paths),
        },
        "stage2_guides": _guides_manifest_entry(
            getattr(args, "stage2_guides_path", None),
            getattr(args, "stage2_guides_text", ""),
        ),
        "inputs": {
            "snippets_dir": str(snippets_dir),
            "page_count": len(images),
        },
        "per_page": _per_page_inputs_stage2(images, stage1_dir),
    }


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
            stage1_reasoning_effort=getattr(args, "stage1_reasoning_effort", "low"),
            stage2_reasoning_effort=getattr(args, "stage2_reasoning_effort", "low"),
            stage1_guides=getattr(args, "stage1_guides_text", ""),
            stage2_guides=getattr(args, "stage2_guides_text", ""),
            stage1_mode=getattr(args, "stage1_mode", "column"),
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
        choices=_STRATEGY_CHOICES,
        default="two_stage",
        help="Extraction strategy (default: two_stage). Use vlm_ocr for "
        "specialized OCR/VLM models (MinerU, PaddleOCR-VL, GLM-OCR).",
    )
    parser.add_argument(
        "--vlm-model",
        dest="vlm_model",
        choices=list_vlm_keys(),
        default=None,
        help="Specialized OCR/VLM backend when --strategy vlm_ocr: "
        "mineru2.5-pro, paddleocr-vl-1.5, glm-ocr.",
    )
    parser.add_argument(
        "--vlm-dpi",
        dest="vlm_dpi",
        type=int,
        default=200,
        help="DPI for rasterizing snippet PDFs in vlm_ocr mode (default: 200).",
    )
    parser.add_argument(
        "--glm-ocr-prompt",
        dest="glm_ocr_prompt",
        default=None,
        help='GLM-OCR prompt when --vlm-model glm-ocr (default: "Text Recognition:").',
    )
    parser.add_argument(
        "--glm-max-new-tokens",
        dest="glm_max_new_tokens",
        type=int,
        default=None,
        help="GLM-OCR max_new_tokens (default: 8192).",
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
        "--stage1-reasoning",
        choices=["low", "medium", "high"],
        default="low",
        dest="stage1_reasoning_effort",
        help="Reasoning effort for the Stage 1 transcription LLM call "
        "(default: low). Stage 1 is a faithful-copy task — higher reasoning "
        "tends to over-interpret the page (silent 'corrections', diacritic "
        "normalization, dropped chars). On Gemini 3, 'low' is the floor "
        "(thinking cannot be fully disabled).",
    )
    parser.add_argument(
        "--stage2-reasoning",
        choices=["low", "medium", "high"],
        default="low",
        dest="stage2_reasoning_effort",
        help="Reasoning effort for the Stage 2 structuring LLM call "
        "(default: low). High reasoning has been observed to leak chain-of-"
        "thought into JSON string fields on dense pages — bump only when "
        "you've confirmed the leak doesn't happen for your model + pages.",
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

    # Per-stage experiment namespacing + ablation toggles
    parser.add_argument(
        "--experiment-name",
        dest="experiment_name",
        default="default",
        help="Stage-1 experiment slot under outputs/stage-1/<name>/. Lets you "
        "keep multiple ablation runs side-by-side (alphabet on/off, OCR hint "
        "on/off, different models) without overwriting each other. Default: "
        "'default'. Must match ^[A-Za-z0-9_.-]+$. Also used as the stage-2 "
        "slot unless --stage2-experiment-name is set, and selects which "
        "stage-1 TSV stage 2 consumes.",
    )
    parser.add_argument(
        "--stage2-experiment-name",
        dest="stage2_experiment_name",
        default=None,
        help="Stage-2 experiment slot under outputs/stage-2/<name>/. Defaults "
        "to --experiment-name. Use a different value to sweep stage-2 "
        "configurations (intro, structure-model, reasoning, --discover-extra-"
        "fields, stage-2 guides) against a fixed stage-1 baseline; the "
        "stage-2 manifest records --experiment-name as its stage1_source.",
    )
    parser.add_argument(
        "--no-alphabet",
        action="store_true",
        dest="no_alphabet",
        help="Suppress alphabet/legend input for Stage 1. In --samples-dir mode "
        "this skips auto-discovery of <lang>/alphabet.txt; in single-entry "
        "mode it ignores --alphabet. Use for alphabet-ablation experiments.",
    )
    parser.add_argument(
        "--no-ocr-hint",
        action="store_true",
        dest="no_ocr_hint",
        help="Suppress OCR hint input for Stage 1. In --samples-dir mode this "
        "skips auto-discovery of <lang>/mathpix/; in single-entry mode it "
        "ignores --ocr-text. Use for OCR-hint ablation experiments.",
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
    parser.add_argument(
        "--stage1-mode",
        choices=["column", "flat"],
        default="column",
        help="Stage-1 output schema for two_stage: column TSV (default) or flat "
        "transcription (eval-flat). Flat mode supports --stage 1 only.",
    )

    args = parser.parse_args()

    # ── VLM OCR strategy validation ───────────────────────────────────────────
    if args.strategy == "vlm_ocr":
        if not args.vlm_model:
            parser.error("--vlm-model is required when --strategy vlm_ocr")
        if args.stage != "1":
            parser.error("--strategy vlm_ocr only supports --stage 1")
        spec = get_vlm_spec(args.vlm_model)
        if args.experiment_name == "default":
            args.experiment_name = spec.experiment_name
            print(f"Using experiment slot: {args.experiment_name}")
    elif args.vlm_model:
        parser.error("--vlm-model is only valid with --strategy vlm_ocr")

    if args.stage1_mode == "flat":
        if args.strategy != "two_stage":
            parser.error("--stage1-mode flat requires --strategy two_stage")
        if args.stage != "1":
            parser.error("--stage1-mode flat supports --stage 1 only (not stage 2)")

    # ── Validate experiment slot names (must be safe directory names) ────────
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.experiment_name):
        parser.error(
            f"--experiment-name must match ^[A-Za-z0-9_.-]+$ (got: "
            f"{args.experiment_name!r})"
        )
    if args.stage2_experiment_name is None:
        args.stage2_experiment_name = args.experiment_name
    elif not re.fullmatch(r"[A-Za-z0-9_.-]+", args.stage2_experiment_name):
        parser.error(
            f"--stage2-experiment-name must match ^[A-Za-z0-9_.-]+$ (got: "
            f"{args.stage2_experiment_name!r})"
        )

    # ── Apply ablation toggles to single-entry inputs too ────────────────────
    # (batch mode applies these inside _run_samples_dir before calling
    # _run_single_entry, so this only matters when --samples-dir is unset.)
    if args.no_alphabet:
        args.alphabet = None
    if args.no_ocr_hint:
        args.ocr_text = None

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
    if args.strategy == "vlm_ocr":
        if args.samples_dir:
            return _run_samples_dir_vlm(args, parser)
        if not args.input_image or not args.output:
            parser.error(
                "--input-image and --output are required for vlm_ocr "
                "unless --samples-dir is used."
            )
        return _run_single_entry_vlm(args, parser)

    if args.samples_dir:
        return _run_samples_dir(args, parser)

    if not args.input_image or not args.output:
        parser.error(
            "--input-image and --output are required unless --samples-dir is used."
        )

    return _run_single_entry(args, parser)


def _discover_sample_entries(samples_root: Path, languages: Optional[List[str]]) -> List[Path]:
    """Return entry subfolders to process under ``samples_root``."""
    all_entries = sorted(p for p in samples_root.iterdir() if p.is_dir())
    if languages:
        requested = set(languages)
        available = {p.name for p in all_entries}
        missing = requested - available
        if missing:
            raise ValueError(
                f"--languages references unknown subfolders: {sorted(missing)}. "
                f"Available: {sorted(available)}"
            )
        return [p for p in all_entries if p.name in requested]
    return all_entries


def _run_samples_dir_vlm(args, parser) -> int:
    """Batch VLM OCR over sample entries (one model load for all languages)."""
    samples_root = Path(args.samples_dir)
    if not samples_root.is_dir():
        parser.error(f"--samples-dir must be a directory: {samples_root}")

    try:
        entries = _discover_sample_entries(samples_root, args.languages)
    except ValueError as exc:
        parser.error(str(exc))

    if not entries:
        print(f"No entry subfolders found under {samples_root}")
        return 1

    print(
        f"VLM OCR batch: {args.vlm_model} on {len(entries)} "
        f"entr{'y' if len(entries) == 1 else 'ies'} under {samples_root}"
    )
    return run_vlm_ocr_batch(args, entries)


def _run_single_entry_vlm(args, parser) -> int:
    """Run VLM OCR on a single entry's snippets directory."""
    input_dir = Path(args.input_image)
    if not input_dir.is_dir():
        parser.error(f"--input-image must be a directory: {input_dir}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = create_vlm_runner(
        args.vlm_model,
        glm_prompt=getattr(args, "glm_ocr_prompt", None),
        glm_max_new_tokens=getattr(args, "glm_max_new_tokens", None),
    )
    runner.load()
    try:
        return run_vlm_ocr_entry(args, input_dir, output_dir, runner)
    finally:
        runner.unload()


def _run_samples_dir(args, parser) -> int:
    """Iterate over every language subfolder under ``args.samples_dir``."""
    samples_root = Path(args.samples_dir)
    if not samples_root.is_dir():
        parser.error(f"--samples-dir must be a directory: {samples_root}")

    try:
        entries = _discover_sample_entries(samples_root, args.languages)
    except ValueError as exc:
        parser.error(str(exc))

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
        args.ocr_text = (
            str(mathpix_dir)
            if mathpix_dir.is_dir() and not args.no_ocr_hint
            else None
        )
        args.intro = str(intro_dir) if intro_dir.is_dir() else None
        args.alphabet = (
            str(alphabet_file)
            if alphabet_file.exists() and not args.no_alphabet
            else None
        )
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
    # Both stages are namespaced by experiment name so multiple ablation runs
    # coexist without overwriting each other. Stage 2 defaults to the same
    # slot as stage 1 but can be overridden via --stage2-experiment-name to
    # sweep stage-2 configurations against a fixed stage-1 baseline.
    stage1_dir = output_dir / "stage-1" / args.experiment_name
    stage2_dir = output_dir / "stage-2" / args.stage2_experiment_name
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
    if args.strategy == "two_stage":
        print(
            f"Stage-1 slot: {args.experiment_name} | Alphabet: "
            f"{'on' if args.alphabet else 'off'} | OCR hint: "
            f"{'on' if args.ocr_text else 'off'} | "
            f"Reasoning: {args.stage1_reasoning_effort}"
        )
        if args.stage in ("2", "both"):
            print(
                f"Stage-2 slot: {args.stage2_experiment_name} "
                f"(stage1_source={args.experiment_name}) | "
                f"Reasoning: {args.stage2_reasoning_effort}"
            )
        # Manifests describe what's on disk in each experiment slot. On
        # resume (slot already populated) the existing manifest wins so it
        # never drifts from the predictions it documents.
        if args.stage in ("1", "both"):
            _write_run_config(
                stage1_dir,
                _build_stage1_manifest(args, input_dir, images, ocr_dir),
                force=args.overwrite,
            )
        if args.stage in ("2", "both"):
            _write_run_config(
                stage2_dir,
                _build_stage2_manifest(
                    args, input_dir, images, stage1_dir, intro_image_paths
                ),
                force=args.overwrite,
            )
    print("=" * 60)

    for idx, image_file in enumerate(images):
        page_number = args.page_offset + idx
        stem = image_file.stem
        stage1_page_dir = stage1_dir / stem
        stage2_page_dir = stage2_dir / stem
        stage1_tsv = stage1_page_dir / (stem + "_stage1.tsv")
        stage1_flat = stage1_page_dir / (stem + "_stage1_flat.txt")
        stage1_done = (
            stage1_flat if getattr(args, "stage1_mode", "column") == "flat" else stage1_tsv
        )
        out_tsv = stage2_page_dir / (stem + ".tsv")

        # ── Resume: skip already-processed pages ──────────────────────────────
        if not args.overwrite:
            if args.stage == "both" and out_tsv.exists():
                print(
                    f"[{idx+1}/{total}] SKIP {image_file.name} → stage-2/{stem}/ already exists"
                )
                skipped += 1
                continue
            if args.stage == "1" and stage1_done.exists():
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
                extract_kwargs["stage1_output_path"] = str(stage1_done)
                if args.stage in ("2", "both"):
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
