"""Shared helpers for model inference smoke tests."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
PDF_SUFFIX = ".pdf"
SUPPORTED_ASSET_SUFFIXES = IMAGE_SUFFIXES | {PDF_SUFFIX}


@dataclass(frozen=True)
class AssetInput:
    """A single test asset from ``models/assets``."""

    path: Path
    stem: str
    kind: str  # "image" or "pdf"
    page_indices: list[int] | None = None  # PDF only; None means all pages


def discover_assets(assets_dir: Path) -> list[AssetInput]:
    """List supported image/PDF files in ``assets_dir``, sorted by name.

    Time complexity: O(n log n) for n files in the directory.
    """
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")

    assets: list[AssetInput] = []
    for path in sorted(assets_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            assets.append(AssetInput(path=path.resolve(), stem=path.stem, kind="image"))
        elif suffix == PDF_SUFFIX:
            assets.append(AssetInput(path=path.resolve(), stem=path.stem, kind="pdf"))

    if not assets:
        raise FileNotFoundError(
            f"No supported assets in {assets_dir} "
            f"(expected {', '.join(sorted(SUPPORTED_ASSET_SUFFIXES))})"
        )
    return assets


def prepare_page_images(
    asset: AssetInput,
    output_dir: Path,
    *,
    dpi: int = 200,
    page_indices: list[int] | None = None,
) -> tuple[list[Path], Path | None]:
    """Materialize page PNGs for an asset under ``output_dir/pages``.

    Returns:
        Tuple of (page image paths, PDF path if asset is a PDF else None).
    """
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    if asset.kind == "image":
        dest = pages_dir / "page_0000.png"
        if asset.path.resolve() != dest.resolve():
            shutil.copy2(asset.path, dest)
        logger.info("Using image asset -> %s", dest)
        return [dest], None

    rendered = render_pdf_pages(
        asset.path,
        output_dir,
        dpi=dpi,
        page_indices=page_indices,
    )
    return rendered, asset.path


@dataclass
class InferenceResult:
    """Outcome of a single model inference run."""

    model: str
    success: bool
    elapsed_seconds: float
    output_dir: Path
    page_indices: list[int] = field(default_factory=list)
    error: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "success": self.success,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "output_dir": str(self.output_dir),
            "page_indices": self.page_indices,
            "error": self.error,
            "artifacts": self.artifacts,
        }


def parse_page_spec(spec: str, total_pages: int) -> list[int]:
    """Parse ``0``, ``all``, or comma-separated page indices."""
    if spec.strip().lower() == "all":
        return list(range(total_pages))

    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        index = int(part)
        if index < 0 or index >= total_pages:
            raise ValueError(f"Page index {index} out of range (PDF has {total_pages} page(s))")
        indices.append(index)
    if not indices:
        raise ValueError("No page indices provided")
    return indices


def get_pdf_page_count(pdf_path: Path) -> int:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 200,
    page_indices: list[int] | None = None,
) -> list[Path]:
    """Render PDF pages to PNG files.

    Args:
        pdf_path: Input PDF path.
        output_dir: Directory to write ``page_XXXX.png`` files.
        dpi: Render resolution.
        page_indices: Zero-based page indices to render. ``None`` renders all pages.

    Returns:
        List of rendered image paths in page order.

    Time complexity: O(n) in number of rendered pages.
    """
    import fitz

    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        total_pages = doc.page_count
        indices = list(range(total_pages)) if page_indices is None else page_indices
        for index in indices:
            if index < 0 or index >= total_pages:
                raise ValueError(
                    f"Page index {index} out of range for PDF with {total_pages} page(s)"
                )

        rendered: list[Path] = []
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for index in indices:
            page = doc.load_page(index)
            image_path = pages_dir / f"page_{index:04d}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(image_path)
            rendered.append(image_path)
            logger.info("Rendered page %d -> %s", index, image_path)

        return rendered
    finally:
        doc.close()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def timed_run(model_name: str, output_dir: Path, fn) -> InferenceResult:
    """Execute ``fn`` and capture timing plus errors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        artifacts = fn()
        elapsed = time.perf_counter() - started
        return InferenceResult(
            model=model_name,
            success=True,
            elapsed_seconds=elapsed,
            output_dir=output_dir,
            artifacts=artifacts,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.exception("%s inference failed", model_name)
        return InferenceResult(
            model=model_name,
            success=False,
            elapsed_seconds=elapsed,
            output_dir=output_dir,
            error=str(exc),
        )
