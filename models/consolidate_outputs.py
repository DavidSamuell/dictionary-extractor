#!/usr/bin/env python3
"""Merge per-model run dirs into one tree: outputs/<name>/<asset>/<model>/."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = MODELS_DIR / "outputs"


def consolidate(name: str, run_dirs: dict[str, Path]) -> Path:
    """Copy model outputs into ``outputs/<name>/<asset>/<model>/``."""
    dest_root = OUTPUT_ROOT / name
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)

    combined_assets: list[dict] = []

    for model_key, run_dir in run_dirs.items():
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for asset_record in summary.get("assets", []):
            asset = asset_record["asset"]
            asset_dest = dest_root / asset
            asset_dest.mkdir(parents=True, exist_ok=True)

            src_asset = run_dir / asset
            if (src_asset / model_key).exists():
                shutil.copytree(src_asset / model_key, asset_dest / model_key)
            if (src_asset / "pages").exists() and not (asset_dest / "pages").exists():
                shutil.copytree(src_asset / "pages", asset_dest / "pages")

            existing = next((a for a in combined_assets if a["asset"] == asset), None)
            if existing is None:
                existing = {
                    "asset": asset,
                    "source": asset_record.get("source"),
                    "kind": asset_record.get("kind"),
                    "results": [],
                }
                combined_assets.append(existing)
            for result in asset_record.get("results", []):
                existing["results"].append(result)

    manifest = {
        "output_dir": str(dest_root),
        "sources": {k: str(v) for k, v in run_dirs.items()},
        "assets": combined_assets,
    }
    (dest_root / "run_summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dest_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="latest", help="Output folder name under models/outputs/")
    args = parser.parse_args()

    run_dirs = {
        "mineru": OUTPUT_ROOT / "run_mineru",
        "paddleocr": OUTPUT_ROOT / "run_paddleocr",
        "glm-ocr": OUTPUT_ROOT / "run_glmocr",
    }
    dest = consolidate(args.name, run_dirs)
    print(f"Consolidated -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
