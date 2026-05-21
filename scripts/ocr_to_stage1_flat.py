#!/usr/bin/env python3
"""Backfill ``*_stage1_flat.txt`` from OCR artifacts under stage-1 experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SAMPLES = _REPO_ROOT / "assets" / "dictionaries" / "samples"

if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dictextractor.ocr.adapters.flat_export import (  # noqa: E402
    detect_backend,
    write_stage1_flat_for_page,
)

logger = logging.getLogger(__name__)


def discover_page_dirs(
    root: Path,
    experiments: list[str] | None,
) -> list[tuple[Path, str]]:
    """Yield (page_dir, stem) under ``outputs/stage-1/<experiment>/<stem>/``."""
    pairs: list[tuple[Path, str]] = []
    for lang_dir in sorted(root.iterdir()):
        if not lang_dir.is_dir():
            continue
        stage1 = lang_dir / "outputs" / "stage-1"
        if not stage1.is_dir():
            continue
        exp_dirs = sorted(stage1.iterdir())
        for exp_dir in exp_dirs:
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            if experiments and exp_dir.name not in experiments:
                continue
            for page_dir in sorted(exp_dir.iterdir()):
                if page_dir.is_dir():
                    pairs.append((page_dir, page_dir.name))
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write stage1 flat preds from OCR raw.")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=_DEFAULT_SAMPLES,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Alternate root (e.g. models/outputs/run_mineru) instead of samples-dir layout",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.root:
        root = args.root.resolve()
        pairs = []
        for page_dir in sorted(root.glob("page_*")):
            if page_dir.is_dir():
                pairs.append((page_dir, page_dir.name))
    else:
        root = args.samples_dir.resolve()
        pairs = discover_page_dirs(root, args.experiments)

    written = 0
    for index, (page_dir, stem) in enumerate(pairs, start=1):
        backend = detect_backend(page_dir, stem=stem)
        if backend is None:
            logger.debug("[%d] skip %s (no artifacts)", index, page_dir)
            continue
        if args.dry_run:
            logger.info("[%d] %s backend=%s", index, page_dir, backend.value)
            continue
        try:
            out = write_stage1_flat_for_page(page_dir, stem=stem)
            logger.info("[%d] %s → %s", index, page_dir, out.name)
            written += 1
        except (OSError, ValueError, FileNotFoundError) as exc:
            logger.error("[%d] %s failed: %s", index, page_dir, exc)

    logger.info("Done. Wrote %d flat file(s).", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
