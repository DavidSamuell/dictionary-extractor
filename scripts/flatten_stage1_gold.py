#!/usr/bin/env python3
"""Generate eval-flat gold files from stage-1 column TSV under the samples tree.

Scans ``<samples-dir>/<lang>/outputs/stage-1-gold/<page>/<page>_stage1_GOLD.tsv``
and writes ``<page>_stage1_GOLD_flat.txt`` beside each file (overwrite if present).

Flatten rule (PLAN.md §3, spec v2): header rows (file order), body columns
left → center → right → single (``middle`` aliases center), footer rows
(file order); one output line per TSV row.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SAMPLES = _REPO_ROOT / "assets" / "dictionaries" / "samples"

# Allow ``python scripts/flatten_stage1_gold.py`` without installing the package.
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dictextractor.evaluation.stage1.flatten import (  # noqa: E402
    FLAT_SPEC_VERSION,
    flatten_stage1_tsv,
    flat_output_path_for_gold,
    write_flat_gold,
)

logger = logging.getLogger(__name__)


def discover_gold_tsvs(samples_dir: Path, languages: list[str] | None) -> list[Path]:
    """List gold TSV paths under ``outputs/stage-1-gold`` only."""
    pattern = "*/outputs/stage-1-gold/*/*_stage1_GOLD.tsv"
    paths = sorted(samples_dir.glob(pattern))
    if languages is None:
        return paths
    allowed = set(languages)
    return [p for p in paths if p.parts[-5] in allowed]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive *_stage1_GOLD_flat.txt from column gold TSV files.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=_DEFAULT_SAMPLES,
        help=f"Root of dictionary samples (default: {_DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional language folder names to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log paths that would be written without writing files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log each file processed",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    samples_dir = args.samples_dir.resolve()
    if not samples_dir.is_dir():
        logger.error("Samples directory not found: %s", samples_dir)
        return 1

    gold_paths = discover_gold_tsvs(samples_dir, args.languages)
    if not gold_paths:
        logger.warning("No *_stage1_GOLD.tsv files under %s", samples_dir)
        return 0

    logger.info(
        "Flatten spec %s | %d gold TSV(s) under %s",
        FLAT_SPEC_VERSION,
        len(gold_paths),
        samples_dir,
    )

    written = 0
    for index, gold_path in enumerate(gold_paths, start=1):
        out_path = flat_output_path_for_gold(gold_path)
        lang = gold_path.parts[-5]
        page = gold_path.parent.name
        if args.dry_run:
            line_count = len(flatten_stage1_tsv(gold_path).splitlines())
            logger.info(
                "[%d/%d] %s / %s → %s (%d lines)",
                index,
                len(gold_paths),
                lang,
                page,
                out_path.name,
                line_count,
            )
            continue
        try:
            write_flat_gold(gold_path)
        except (OSError, ValueError) as exc:
            logger.error("[%d/%d] failed %s: %s", index, len(gold_paths), gold_path, exc)
            return 1
        if args.verbose:
            line_count = len(out_path.read_text(encoding="utf-8").splitlines())
            logger.info(
                "[%d/%d] %s / %s → %s (%d lines)",
                index,
                len(gold_paths),
                lang,
                page,
                out_path.name,
                line_count,
            )
        written += 1

    if not args.dry_run:
        logger.info("Done. Wrote %d flat gold file(s).", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
