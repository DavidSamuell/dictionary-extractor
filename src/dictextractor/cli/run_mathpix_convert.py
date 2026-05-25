"""CLI: OCR every dictionary's snippet pages via the Mathpix Convert API.

For each ``{source}-{target...}`` entry folder under the samples root, if no
``mathpix/`` subfolder exists, create it and convert every PDF/PNG found in
``snippets/`` into markdown using the Mathpix Convert API.

Usage:
    python -m dictextractor.cli.run_mathpix_convert \\
        --samples-dir assets/dictionaries/samples-2

Environment variables ``MATHPIX_APP_ID`` and ``MATHPIX_APP_KEY`` must be set
(``.env`` is loaded automatically if present).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from dictextractor.ocr.mathpix_convert import (
    MathpixConvertClient,
    MathpixConvertError,
)
from dictextractor.ocr.vlm.page_inputs import list_snippet_pages

logger = logging.getLogger(__name__)

DEFAULT_SAMPLES_DIR = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "dictionaries"
    / "samples-2"
)


def iter_entry_folders(samples_dir: Path) -> list[Path]:
    """Return sorted ``{source}-{target...}`` subfolders under ``samples_dir``."""
    return sorted(p for p in samples_dir.iterdir() if p.is_dir())


def resolve_entry_folders(
    samples_dir: Path,
    languages: list[str] | None,
) -> list[Path]:
    """Return entry folders to process, optionally filtered by language names."""
    all_entries = iter_entry_folders(samples_dir)
    if not languages:
        return all_entries

    available = {p.name for p in all_entries}
    missing = sorted(set(languages) - available)
    if missing:
        raise ValueError(
            f"--languages references unknown subfolders: {missing}. "
            f"Available: {sorted(available)}"
        )
    return [p for p in all_entries if p.name in set(languages)]


def process_entry(
    entry_dir: Path,
    client: MathpixConvertClient,
    *,
    force: bool,
    overwrite_files: bool,
) -> None:
    """Convert every snippet page in ``entry_dir`` into markdown in ``mathpix/``."""
    snippets_dir = entry_dir / "snippets"
    mathpix_dir = entry_dir / "mathpix"

    if not snippets_dir.is_dir():
        logger.warning("Skipping %s: no snippets/ folder", entry_dir.name)
        return

    try:
        snippet_files = list_snippet_pages(snippets_dir)
    except FileNotFoundError:
        logger.warning("Skipping %s: no snippet pages in snippets/", entry_dir.name)
        return

    if mathpix_dir.exists() and not force:
        complete = all(
            (mathpix_dir / f"{snippet.stem}.md").is_file()
            and (mathpix_dir / f"{snippet.stem}.lines.json").is_file()
            for snippet in snippet_files
        )
        if complete:
            logger.info(
                "Skipping %s: mathpix/ complete (md + lines.json for all pages)",
                entry_dir.name,
            )
            return

    mathpix_dir.mkdir(parents=True, exist_ok=True)
    upload_cache = mathpix_dir / ".upload_cache"
    logger.info("Processing %s (%d snippet(s))", entry_dir.name, len(snippet_files))

    for snippet_path in snippet_files:
        output_path = mathpix_dir / f"{snippet_path.stem}.md"
        lines_path = mathpix_dir / f"{snippet_path.stem}.lines.json"
        md_ok = output_path.is_file()
        lines_ok = lines_path.is_file()
        legacy_docx = mathpix_dir / f"{snippet_path.stem}.docx"
        if md_ok and lines_ok and not overwrite_files:
            logger.info(
                "  %s + %s already exist, skipping",
                output_path.name,
                lines_path.name,
            )
            continue
        if not md_ok and legacy_docx.is_file() and lines_ok and not overwrite_files:
            logger.info(
                "  %s missing (legacy docx present); converting to fetch markdown",
                output_path.name,
            )
        if md_ok and not lines_ok and not overwrite_files:
            logger.info(
                "  %s exists but %s missing; re-converting to fetch lines.json",
                output_path.name,
                lines_path.name,
            )
        try:
            client.convert_pdf_page(
                snippet_path,
                md_path=output_path,
                lines_json_path=lines_path,
                upload_cache_dir=upload_cache,
            )
            logger.info(
                "  %s -> %s (+ %s)",
                snippet_path.name,
                output_path.name,
                lines_path.name,
            )
        except MathpixConvertError as e:
            logger.error("  Failed %s: %s", snippet_path.name, e)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Root directory containing {source}-{target} entry folders",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Optional language subfolder names to process (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process entries even if a mathpix/ folder already exists",
    )
    parser.add_argument(
        "--overwrite-files",
        action="store_true",
        help="Re-convert individual snippets even if the .md is already present",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=3.0,
        help="Seconds between Mathpix status polls",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=600.0,
        help="Maximum seconds to wait per snippet before giving up",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()

    if not args.samples_dir.is_dir():
        logger.error("Samples directory not found: %s", args.samples_dir)
        return 1

    try:
        entries = resolve_entry_folders(args.samples_dir, args.languages)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    try:
        client = MathpixConvertClient(
            poll_interval_seconds=args.poll_interval,
            max_wait_seconds=args.max_wait,
        )
    except MathpixConvertError as e:
        logger.error(str(e))
        return 1

    for entry_dir in entries:
        process_entry(
            entry_dir,
            client,
            force=args.force,
            overwrite_files=args.overwrite_files,
        )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
