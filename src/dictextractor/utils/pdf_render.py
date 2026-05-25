"""Render PDF pages to PNG for VLMs that do not accept inline PDF data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

PDF_RENDER_DPI = 200


def needs_pdf_rasterization(model: str, *, preprocess: bool = False) -> bool:
    """
    Return True when PDF inputs must be rendered to PNG before LLM calls.

    Gemini accepts ``application/pdf`` inline; most hosted VLMs (OpenRouter GPT,
    Claude via image_url, etc.) require raster images.
    """
    if preprocess:
        return True
    model_lower = model.lower()
    if "gemini" in model_lower or model_lower.startswith("google/"):
        return False
    return True


def render_pdf_pages(
    pdf_path: Path,
    cache_dir: Path,
    dpi: int = PDF_RENDER_DPI,
) -> List[Path]:
    """
    Render each page of ``pdf_path`` to a PNG under ``cache_dir``.

    Cached outputs are reused when newer than the source PDF.

    Returns:
        Rendered image paths in page order.
    """
    import pymupdf  # lazy import — PyMuPDF is optional for image-only runs

    cache_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    results: List[Path] = []
    page_count = 0
    try:
        pdf_mtime = pdf_path.stat().st_mtime
        page_count = doc.page_count
        for page_index in range(page_count):
            suffix = "" if page_count == 1 else f"_p{page_index + 1}"
            out_path = cache_dir / f"{pdf_path.stem}{suffix}.png"
            if out_path.exists() and out_path.stat().st_mtime >= pdf_mtime:
                results.append(out_path)
                continue
            pix = doc.load_page(page_index).get_pixmap(dpi=dpi)
            pix.save(str(out_path))
            results.append(out_path)
            logger.debug("Rendered %s page %d -> %s", pdf_path.name, page_index + 1, out_path)
    finally:
        doc.close()
    if page_count > 1:
        logger.info(
            "Rendered %d pages from %s to %s",
            page_count,
            pdf_path.name,
            cache_dir,
        )
    return results
