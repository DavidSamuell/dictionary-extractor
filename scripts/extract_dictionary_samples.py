"""Extract introduction and dictionary snippet pages from full-dictionary PDFs.

Reads the dictionary metadata CSV and, for every row, uses `pdftk` to burst out
each introduction page and each sampled dictionary-entry page into its own PDF.
Output is organised per-dictionary under a `{source}-{target1}-{target2}...`
folder with `introduction/` and `snippets/` subfolders.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "assets" / "dictionaries" / "full dictionaries" / "dictionary_metadata.csv"
DEFAULT_PDF_DIR = REPO_ROOT / "assets" / "dictionaries" / "full dictionaries"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets" / "dictionaries" / "samples-2"


def sanitize_language_token(token: str) -> str:
    """Make a language token safe for use inside a folder name.

    Slashes (e.g. "Kurdish/Turkish") become underscores so the token survives
    as a single path component.
    """
    return token.strip().replace("/", "_")


def row_pdf_name(row: dict[str, str]) -> str:
    """PDF stem from metadata row.

    The CSV's first column holds the pdf stem but often has no header (leading
    comma in the file), so DictReader keys it as ``""`` rather than
    ``pdf_name``.
    """
    return (row.get("pdf_name") or row.get("") or "").strip()


def build_folder_name(source_language: str, target_language: str) -> str:
    """Build the per-dictionary folder name from source/target languages.

    A target cell like ``"English, Hindi"`` contributes two tokens; a token
    like ``"Kurdish/Turkish"`` is kept as a single ``Kurdish_Turkish`` chunk.
    """
    source = source_language.strip()
    targets = [sanitize_language_token(t) for t in target_language.split(",") if t.strip()]
    parts = [source, *targets]
    return "-".join(parts)


def parse_page_spec(spec: str) -> list[int]:
    """Expand a page specification string into an ordered list of page numbers.

    Supports comma-separated singletons and hyphen-separated inclusive ranges,
    e.g. ``"97 - 123, 179 - 182"`` or ``"19, 83, 162"``. Returns ``[]`` for
    empty/whitespace input.
    """
    if not spec or not spec.strip():
        return []

    pages: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", chunk)
            if not match:
                raise ValueError(f"Unrecognised page range: {chunk!r}")
            start, end = int(match.group(1)), int(match.group(2))
            if end < start:
                raise ValueError(f"Descending range not supported: {chunk!r}")
            pages.extend(range(start, end + 1))
        else:
            if not chunk.isdigit():
                raise ValueError(f"Unrecognised page token: {chunk!r}")
            pages.append(int(chunk))
    return pages


def extract_single_page(source_pdf: Path, page: int, output_pdf: Path) -> None:
    """Extract a single page from ``source_pdf`` to ``output_pdf`` via pdftk."""
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "pdftk",
        str(source_pdf),
        "cat",
        str(page),
        "output",
        str(output_pdf),
    ]
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"pdftk failed for {source_pdf.name} page {page}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def process_row(
    row: dict[str, str],
    pdf_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Extract introduction and snippet pages for a single metadata row."""
    pdf_name = row_pdf_name(row)
    source_language = (row.get("source_language") or "").strip()
    target_language = (row.get("target_language") or "").strip()
    if not pdf_name:
        logger.warning("Skipping row with empty pdf name (source=%s)", source_language)
        return
    intro_spec = (row.get("introduction") or "").strip()
    snippet_spec = (row.get("pages") or "").strip()

    source_pdf = pdf_dir / f"{pdf_name}.pdf"
    if not source_pdf.exists():
        logger.warning("Skipping %s: source PDF not found at %s", pdf_name, source_pdf)
        return

    folder_name = build_folder_name(source_language, target_language)
    dict_dir = output_dir / folder_name
    intro_dir = dict_dir / "introduction"
    snippets_dir = dict_dir / "snippets"

    try:
        intro_pages = parse_page_spec(intro_spec)
        snippet_pages = parse_page_spec(snippet_spec)
    except ValueError as e:
        logger.error("Skipping %s: %s", pdf_name, e)
        return

    logger.info(
        "Processing %s -> %s (intro: %d pages, snippets: %d pages)",
        pdf_name,
        folder_name,
        len(intro_pages),
        len(snippet_pages),
    )

    intro_dir.mkdir(parents=True, exist_ok=True)
    snippets_dir.mkdir(parents=True, exist_ok=True)

    for page in intro_pages:
        out_path = intro_dir / f"page_{page}.pdf"
        if out_path.exists() and not overwrite:
            logger.debug("Intro page %d already exists, skipping", page)
            continue
        extract_single_page(source_pdf, page, out_path)

    for page in snippet_pages:
        out_path = snippets_dir / f"page_{page}.pdf"
        if out_path.exists() and not overwrite:
            logger.debug("Snippet page %d already exists, skipping", page)
            continue
        extract_single_page(source_pdf, page, out_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Metadata CSV path")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="Directory of source PDFs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output samples directory")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract pages even if an output PDF already exists")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if shutil.which("pdftk") is None:
        logger.error("pdftk is not available on PATH; install it (e.g. `brew install pdftk-java`)")
        return 1

    if not args.csv.exists():
        logger.error("CSV not found: %s", args.csv)
        return 1
    if not args.pdf_dir.is_dir():
        logger.error("PDF directory not found: %s", args.pdf_dir)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = skipped_empty = 0
    with args.csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row_pdf_name(row):
                skipped_empty += 1
                continue
            process_row(row, args.pdf_dir, args.output_dir, overwrite=args.overwrite)
            processed += 1

    if processed == 0:
        logger.warning(
            "No rows processed (%d skipped with empty pdf name). "
            "Check that the CSV first column contains pdf stems.",
            skipped_empty,
        )
    logger.info(
        "Done. %d dictionaries processed, output at %s",
        processed,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
