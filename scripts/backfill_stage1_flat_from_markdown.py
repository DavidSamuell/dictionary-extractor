#!/usr/bin/env python3
"""Regenerate ``*_stage1_flat.txt`` from normalized markdown OCR artifacts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SAMPLES = _REPO_ROOT / "assets" / "dictionaries" / "samples"

if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dictextractor.evaluation.stage1.flatten import (  # noqa: E402
    flat_output_path_for_pred,
    write_flat_text,
)
from dictextractor.ocr.adapters.markdown_to_flat import (  # noqa: E402
    find_markdown_source,
    markdown_transcript_from_page_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_FLAT_EXPERIMENTS = (
    "MinerU2.5-Pro",
    "PaddleOCR-VL-1.5",
    "GLM-OCR-flat_alpha",
    "GLM-OCR-flat_noalpha",
    "Mathpix-OCR",
)


def discover_markdown_page_dirs(
    root: Path,
    experiments: list[str],
    languages: list[str] | None = None,
) -> list[tuple[Path, str, str]]:
    """Return ``(page_dir, stem, experiment)`` for pages with markdown."""
    allowed = set(languages) if languages else None
    pairs: list[tuple[Path, str, str]] = []
    for lang_dir in sorted(root.iterdir()):
        if not lang_dir.is_dir():
            continue
        if allowed is not None and lang_dir.name not in allowed:
            continue
        stage1 = lang_dir / "outputs" / "stage-1"
        if not stage1.is_dir():
            continue
        for exp_name in experiments:
            exp_dir = stage1 / exp_name
            if not exp_dir.is_dir():
                continue
            for page_dir in sorted(exp_dir.iterdir()):
                if not page_dir.is_dir():
                    continue
                stem = page_dir.name
                if find_markdown_source(page_dir, stem=stem) is None:
                    logger.debug("skip %s (no markdown)", page_dir)
                    continue
                pairs.append((page_dir, stem, exp_name))
    return pairs


def write_flat_from_markdown(page_dir: Path, *, stem: str) -> Path:
    """Write flat text using the markdown adapter only."""
    parts = markdown_transcript_from_page_dir(page_dir, stem=stem)
    out = flat_output_path_for_pred(page_dir, stem)
    write_flat_text(out, parts.all_lines())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill stage-1 flat preds from markdown OCR artifacts.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=_DEFAULT_SAMPLES,
        help=f"Root of language sample folders (default: {_DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional language folder names to process (default: all)",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        default=None,
        help="Stage-1 experiment folder name (repeatable). "
        f"Default: {', '.join(DEFAULT_FLAT_EXPERIMENTS)}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing *_stage1_flat.txt files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pages that would be written without writing files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    experiments = args.experiments or list(DEFAULT_FLAT_EXPERIMENTS)
    root = args.samples_dir.resolve()
    if not root.is_dir():
        logger.error("samples dir not found: %s", root)
        return 1

    pairs = discover_markdown_page_dirs(root, experiments, args.languages)
    total = len(pairs)
    logger.info(
        "Found %d page dir(s) with markdown under %d experiment(s) in %s",
        total,
        len(experiments),
        root,
    )

    written = skipped = 0
    for index, (page_dir, stem, experiment) in enumerate(pairs, start=1):
        out = flat_output_path_for_pred(page_dir, stem)
        if out.is_file() and not args.overwrite:
            logger.debug(
                "[%d/%d] skip %s (exists, use --overwrite)",
                index,
                total,
                out,
            )
            skipped += 1
            continue

        md_source = find_markdown_source(page_dir, stem=stem)
        if args.dry_run:
            logger.info(
                "[%d/%d] %s experiment=%s md=%s → %s",
                index,
                total,
                page_dir,
                experiment,
                md_source.name if md_source else "?",
                out.name,
            )
            continue

        try:
            out_path = write_flat_from_markdown(page_dir, stem=stem)
            logger.info(
                "[%d/%d] %s experiment=%s → %s",
                index,
                total,
                page_dir,
                experiment,
                out_path.name,
            )
            written += 1
        except (OSError, ValueError, FileNotFoundError) as exc:
            logger.error("[%d/%d] %s failed: %s", index, total, page_dir, exc)

    if args.dry_run:
        would_write = total - skipped
        logger.info(
            "Dry run complete. Would write %d flat file(s); %d skipped (exist).",
            would_write,
            skipped,
        )
    else:
        logger.info(
            "Done. Wrote %d flat file(s); %d skipped (exist).",
            written,
            skipped,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
