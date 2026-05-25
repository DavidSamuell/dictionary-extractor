#!/usr/bin/env python3
"""
Smoke-test document OCR / VLM models on assets under ``models/assets``.

Run from the dictionary-extractor project root:

    python models/test_model_inference.py --models mineru

See models/README.md for per-model installation instructions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
DEFAULT_ASSETS_DIR = MODELS_DIR / "assets"
DEFAULT_OUTPUT_ROOT = MODELS_DIR / "outputs"
PROJECT_ROOT = MODELS_DIR.parent
sys.path.insert(0, str(MODELS_DIR))

from common import (  # noqa: E402
    AssetInput,
    InferenceResult,
    discover_assets,
    get_pdf_page_count,
    parse_page_spec,
    prepare_page_images,
    write_json,
)
from runners.glmocr import run_glmocr  # noqa: E402
from runners.mineru import run_mineru  # noqa: E402
from runners.openrouter_vlm import (  # noqa: E402
    DEFAULT_MODEL as QWEN3_VL_DEFAULT_MODEL,
    run_openrouter_vlm,
)
from runners.paddleocr import run_paddleocr_vl  # noqa: E402

ALL_MODELS = ("mineru", "paddleocr", "glm-ocr", "qwen3-vl-235b")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_model_inference")


def build_output_dir(base_dir: Path | None, label: str) -> Path:
    """Return the run output directory under ``models/outputs``."""
    if base_dir is not None:
        return base_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUTPUT_ROOT / f"{label}_{timestamp}").resolve()


def run_model_on_asset(
    model_name: str,
    asset: AssetInput,
    asset_output_dir: Path,
    *,
    page_images: list[Path],
    pdf_path: Path | None,
    page_indices: list[int],
    dpi: int,
    mineru_backend: str,
    mineru_use_cli: bool,
    use_full_pdf: bool,
    glm_prompt: str,
    glm_max_new_tokens: int,
    qwen3_vl_model: str,
    qwen3_vl_prompt: str,
    qwen3_vl_max_tokens: int,
    qwen3_vl_temperature: float,
    qwen3_vl_top_p: float | None,
) -> InferenceResult:
    """Run a single model on one asset."""
    model_output = asset_output_dir / model_name
    logger.info("=== %s on %s ===", model_name, asset.stem)

    source_path = pdf_path or asset.path
    full_pdf = use_full_pdf and pdf_path is not None

    if model_name == "mineru":
        result = run_mineru(
            source_path,
            model_output,
            page_images=page_images,
            backend=mineru_backend,
            use_cli=mineru_use_cli or full_pdf,
        )
    elif model_name == "paddleocr":
        result = run_paddleocr_vl(
            source_path,
            model_output,
            page_images=page_images,
            use_full_pdf=full_pdf,
        )
    elif model_name == "glm-ocr":
        result = run_glmocr(
            source_path,
            model_output,
            page_images=page_images,
            prompt=glm_prompt,
            max_new_tokens=glm_max_new_tokens,
            use_full_pdf=full_pdf,
        )
    elif model_name == "qwen3-vl-235b":
        result = run_openrouter_vlm(
            model_output,
            page_images=page_images,
            model=qwen3_vl_model,
            prompt=qwen3_vl_prompt,
            max_tokens=qwen3_vl_max_tokens,
            temperature=qwen3_vl_temperature,
            top_p=qwen3_vl_top_p,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    result.page_indices = page_indices
    status = "OK" if result.success else "FAILED"
    logger.info(
        "%s / %s finished in %.2fs (%s)",
        model_name,
        asset.stem,
        result.elapsed_seconds,
        status,
    )
    return result


def run_all(
    *,
    assets: list[AssetInput],
    output_dir: Path,
    models: list[str],
    page_spec: str,
    dpi: int,
    mineru_backend: str,
    mineru_use_cli: bool,
    use_full_pdf: bool,
    glm_prompt: str,
    glm_max_new_tokens: int,
    qwen3_vl_model: str,
    qwen3_vl_prompt: str,
    qwen3_vl_max_tokens: int,
    qwen3_vl_temperature: float,
    qwen3_vl_top_p: float | None,
) -> list[dict]:
    """Run each model on every asset; return per-asset result records."""
    run_records: list[dict] = []

    for asset in assets:
        asset_dir = output_dir / asset.stem
        asset_dir.mkdir(parents=True, exist_ok=True)

        if asset.kind == "pdf":
            total_pages = get_pdf_page_count(asset.path)
            page_indices = parse_page_spec(page_spec, total_pages)
        else:
            total_pages = 1
            page_indices = [0]

        render_indices = list(range(total_pages)) if use_full_pdf and asset.kind == "pdf" else page_indices
        page_images, pdf_path = prepare_page_images(
            asset,
            asset_dir,
            dpi=dpi,
            page_indices=render_indices if asset.kind == "pdf" else None,
        )

        asset_record: dict = {
            "asset": asset.stem,
            "source": str(asset.path),
            "kind": asset.kind,
            "page_indices": page_indices,
            "results": [],
        }

        for model_name in models:
            result = run_model_on_asset(
                model_name,
                asset,
                asset_dir,
                page_images=page_images,
                pdf_path=pdf_path,
                page_indices=page_indices,
                dpi=dpi,
                mineru_backend=mineru_backend,
                mineru_use_cli=mineru_use_cli,
                use_full_pdf=use_full_pdf,
                glm_prompt=glm_prompt,
                glm_max_new_tokens=glm_max_new_tokens,
                qwen3_vl_model=qwen3_vl_model,
                qwen3_vl_prompt=qwen3_vl_prompt,
                qwen3_vl_max_tokens=qwen3_vl_max_tokens,
                qwen3_vl_temperature=qwen3_vl_temperature,
                qwen3_vl_top_p=qwen3_vl_top_p,
            )
            asset_record["results"].append(result.to_dict())

        run_records.append(asset_record)

    return run_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OCR/VLM model inference smoke tests on models/assets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help="Directory with test PDFs/images (default: models/assets)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Optional single PDF/image; if set, overrides --assets-dir",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: models/outputs/assets_<timestamp>)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(ALL_MODELS),
        help="Models to run (sequentially on each asset)",
    )
    parser.add_argument(
        "--pages",
        default="0",
        help="PDF page indices: '0', '0,1', or 'all' (ignored for images)",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Use native full-PDF pipelines where supported (PDF assets only)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI when rendering PDF pages to PNG")

    parser.add_argument(
        "--mineru-backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="MinerU backend when not using the MinerU CLI",
    )
    parser.add_argument(
        "--mineru-use-cli",
        action="store_true",
        help="Use the full MinerU CLI pipeline (mineru -p ... -o ...)",
    )

    parser.add_argument(
        "--glm-ocr-prompt",
        default="Text Recognition:",
        help="GLM-OCR task prompt (see https://huggingface.co/zai-org/GLM-OCR)",
    )
    parser.add_argument(
        "--glm-ocr-max-new-tokens",
        type=int,
        default=8192,
        help="Max tokens for GLM-OCR generation",
    )

    parser.add_argument(
        "--qwen3-vl-model",
        default=QWEN3_VL_DEFAULT_MODEL,
        help="OpenRouter litellm model id for Qwen3-VL-235B",
    )
    parser.add_argument(
        "--qwen3-vl-prompt",
        default=(
            "Transcribe all visible text from this dictionary page image. "
            "Preserve line order and diacritics exactly. "
            "Return plain text only, no commentary."
        ),
        help="Prompt for Qwen3-VL via OpenRouter",
    )
    parser.add_argument("--qwen3-vl-max-tokens", type=int, default=8192)
    parser.add_argument("--qwen3-vl-temperature", type=float, default=0.1)
    parser.add_argument(
        "--qwen3-vl-top-p",
        type=float,
        default=None,
        help="Optional top_p forwarded to OpenRouter (omit to use provider default)",
    )

    args = parser.parse_args()

    if args.input is not None:
        input_path = args.input.resolve()
        if not input_path.exists():
            logger.error("Input not found: %s", input_path)
            return 1
        suffix = input_path.suffix.lower()
        if suffix == ".pdf":
            assets = [AssetInput(path=input_path, stem=input_path.stem, kind="pdf")]
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
            assets = [AssetInput(path=input_path, stem=input_path.stem, kind="image")]
        else:
            logger.error("Unsupported input type: %s", input_path)
            return 1
        label = input_path.stem
    else:
        assets_dir = args.assets_dir.resolve()
        assets = discover_assets(assets_dir)
        label = "assets"

    output_dir = build_output_dir(args.output_dir, label)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Assets: %s", ", ".join(a.stem for a in assets))
    logger.info("Output: %s", output_dir)
    logger.info("Models: %s", ", ".join(args.models))

    run_records = run_all(
        assets=assets,
        output_dir=output_dir,
        models=args.models,
        page_spec=args.pages,
        dpi=args.dpi,
        mineru_backend=args.mineru_backend,
        mineru_use_cli=args.mineru_use_cli,
        use_full_pdf=args.all_pages,
        glm_prompt=args.glm_ocr_prompt,
        glm_max_new_tokens=args.glm_ocr_max_new_tokens,
        qwen3_vl_model=args.qwen3_vl_model,
        qwen3_vl_prompt=args.qwen3_vl_prompt,
        qwen3_vl_max_tokens=args.qwen3_vl_max_tokens,
        qwen3_vl_temperature=args.qwen3_vl_temperature,
        qwen3_vl_top_p=args.qwen3_vl_top_p,
    )

    failures: list[str] = []
    for record in run_records:
        for result in record["results"]:
            if not result["success"]:
                failures.append(f"{record['asset']}/{result['model']}")

    summary = {
        "assets_dir": str(args.assets_dir.resolve()) if args.input is None else None,
        "inputs": [str(a.path) for a in assets],
        "output_dir": str(output_dir),
        "pages": args.pages,
        "all_pages": args.all_pages,
        "assets": run_records,
    }
    write_json(output_dir / "run_summary.json", summary)

    if failures:
        logger.error("Failed runs: %s", ", ".join(failures))
        return 1

    logger.info("All selected models completed successfully on all assets.")
    logger.info("Summary written to %s", output_dir / "run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
