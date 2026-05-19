"""Download a Google Spreadsheet as a local CSV file.

Uses the public ``/export?format=csv`` endpoint, which works when the sheet is
shared with link access (no Google API credentials required).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1UeG5ZNLuf-rJYb_5sR2PBuHaR4Ks586SpsnVX6WiNgI/edit?usp=sharing"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "assets" / "dictionaries" / "full dictionaries" / "dictionary_metadata.csv"
)

_SPREADSHEET_ID_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")


class GoogleSheetExportError(RuntimeError):
    """Raised when the spreadsheet cannot be downloaded."""


def extract_spreadsheet_id(sheet_url: str) -> str:
    """Return the spreadsheet ID embedded in a Google Sheets URL."""
    match = _SPREADSHEET_ID_RE.search(sheet_url)
    if not match:
        raise GoogleSheetExportError(
            f"Could not parse spreadsheet ID from URL: {sheet_url!r}"
        )
    return match.group(1)


def build_export_url(spreadsheet_id: str, gid: int | None = None) -> str:
    """Build the CSV export URL for a spreadsheet (optionally one tab)."""
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"
    if gid is not None:
        url = f"{url}&gid={gid}"
    return url


def download_csv(export_url: str, timeout_seconds: float = 60.0) -> bytes:
    """Fetch CSV bytes from a Google Sheets export URL."""
    request = Request(
        export_url,
        headers={"User-Agent": "dictextractor-export-google-sheet/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except HTTPError as error:
        if error.code in {401, 403}:
            raise GoogleSheetExportError(
                "Access denied. Ensure the spreadsheet is shared with "
                "'Anyone with the link' (viewer access is enough)."
            ) from error
        raise GoogleSheetExportError(
            f"HTTP {error.code} while downloading sheet: {error.reason}"
        ) from error
    except URLError as error:
        raise GoogleSheetExportError(f"Network error: {error.reason}") from error

    if not payload.strip():
        raise GoogleSheetExportError("Downloaded CSV is empty.")
    return payload


def export_sheet(
    output_file: Path,
    sheet_url: str = DEFAULT_SHEET_URL,
    gid: int | None = None,
) -> Path:
    """Download a Google Sheet and write it to ``output_file``.

    Args:
        output_file: Destination path (parent directories are created).
        sheet_url: Google Sheets edit or share URL.
        gid: Optional worksheet tab ID (``gid`` query param from the browser URL).

    Returns:
        Resolved path of the written file.
    """
    spreadsheet_id = extract_spreadsheet_id(sheet_url)
    export_url = build_export_url(spreadsheet_id, gid=gid)
    logger.info("Downloading %s", export_url)
    payload = download_csv(export_url)

    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(payload)
    logger.info("Wrote %s (%d bytes)", output_file, len(payload))
    return output_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=(
            "Destination CSV path (filename and directory). "
            f"Default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument(
        "--sheet-url",
        default=DEFAULT_SHEET_URL,
        help="Google Sheets URL (edit or share link).",
    )
    parser.add_argument(
        "--gid",
        type=int,
        default=None,
        help="Worksheet tab ID (from #gid=... in the browser URL).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Google Sheets CSV export."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        export_sheet(args.output_file, sheet_url=args.sheet_url, gid=args.gid)
    except GoogleSheetExportError as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
