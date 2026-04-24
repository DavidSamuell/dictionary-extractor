"""CLI: OCR every dictionary's snippet PDFs via the Mathpix Convert API.

For each ``{source}-{target...}`` entry folder under the samples root, if no
``mathpix/`` subfolder exists, create it and convert every PDF found in
``snippets/`` into a .docx using the Mathpix Convert API.

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


def process_entry(
    entry_dir: Path,
    client: MathpixConvertClient,
    *,
    force: bool,
    overwrite_files: bool,
) -> None:
    """Convert every snippet PDF in ``entry_dir`` into a .docx in ``mathpix/``."""
    snippets_dir = entry_dir / "snippets"
    mathpix_dir = entry_dir / "mathpix"

    if not snippets_dir.is_dir():
        logger.warning("Skipping %s: no snippets/ folder", entry_dir.name)
        return

    if mathpix_dir.exists() and not force:
        logger.info("Skipping %s: mathpix/ already exists (use --force to re-run)", entry_dir.name)
        return

    pdf_files = sorted(snippets_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("Skipping %s: no PDFs in snippets/", entry_dir.name)
        return

    mathpix_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Processing %s (%d PDFs)", entry_dir.name, len(pdf_files))

    for pdf_path in pdf_files:
        output_path = mathpix_dir / f"{pdf_path.stem}.docx"
        if output_path.exists() and not overwrite_files:
            logger.info("  %s already exists, skipping", output_path.name)
            continue
        try:
            client.convert_pdf_to_docx(pdf_path, output_path)
            logger.info("  %s -> %s", pdf_path.name, output_path.name)
        except MathpixConvertError as e:
            logger.error("  Failed %s: %s", pdf_path.name, e)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Root directory containing {source}-{target} entry folders",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process entries even if a mathpix/ folder already exists",
    )
    parser.add_argument(
        "--overwrite-files",
        action="store_true",
        help="Re-convert individual PDFs even if the .docx is already present",
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
        help="Maximum seconds to wait per PDF before giving up",
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
        client = MathpixConvertClient(
            poll_interval_seconds=args.poll_interval,
            max_wait_seconds=args.max_wait,
        )
    except MathpixConvertError as e:
        logger.error(str(e))
        return 1

    for entry_dir in iter_entry_folders(args.samples_dir):
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
