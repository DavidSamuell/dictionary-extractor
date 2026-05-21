"""Snippet discovery and PDF rasterization for VLM OCR stage-1."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
PDF_SUFFIX = ".pdf"
SNIPPET_SUFFIXES = IMAGE_SUFFIXES | {PDF_SUFFIX}


def list_snippet_pages(snippets_dir: Path) -> list[Path]:
    """Return sorted page files under ``snippets_dir``.

    Time complexity: O(n log n) for n files.
    """
    if not snippets_dir.is_dir():
        raise FileNotFoundError(f"Snippets directory not found: {snippets_dir}")

    pages = [
        path
        for path in sorted(snippets_dir.iterdir())
        if path.is_file()
        and not path.name.startswith((".", "~"))
        and path.suffix.lower() in SNIPPET_SUFFIXES
    ]
    if not pages:
        raise FileNotFoundError(
            f"No snippet pages in {snippets_dir} "
            f"(expected {', '.join(sorted(SNIPPET_SUFFIXES))})"
        )
    return pages


def materialize_page_image(
    snippet: Path,
    cache_dir: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Return a PNG path suitable for VLM inference.

    Images are copied into ``cache_dir`` when needed. PDFs are rasterized
    (first page only — sample snippets are one page per file).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = snippet.suffix.lower()
    dest = cache_dir / f"{snippet.stem}.png"

    if suffix in IMAGE_SUFFIXES:
        if snippet.resolve() != dest.resolve():
            shutil.copy2(snippet, dest)
        logger.debug("Using image snippet -> %s", dest)
        return dest

    if suffix == PDF_SUFFIX:
        import fitz

        doc = fitz.open(snippet)
        try:
            if doc.page_count == 0:
                raise ValueError(f"PDF has no pages: {snippet}")
            page_index = 0
            pdf_mtime = snippet.stat().st_mtime
            if dest.exists() and dest.stat().st_mtime >= pdf_mtime:
                logger.debug("Reusing rendered PDF cache -> %s", dest)
                return dest
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = doc.load_page(page_index).get_pixmap(matrix=matrix, alpha=False)
            pix.save(dest)
            logger.info("Rendered %s -> %s", snippet.name, dest)
            return dest
        finally:
            doc.close()

    raise ValueError(f"Unsupported snippet type: {snippet}")
