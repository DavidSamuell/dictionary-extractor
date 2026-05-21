"""PaddleOCR-VL-1.5 inference runner."""

from __future__ import annotations

import logging
from pathlib import Path

from common import InferenceResult, timed_run

logger = logging.getLogger(__name__)


def run_paddleocr_vl(
    pdf_path: Path,
    output_dir: Path,
    *,
    page_images: list[Path],
    use_full_pdf: bool = False,
) -> InferenceResult:
    """Run PaddleOCR-VL-1.5 document parser."""
    if use_full_pdf:
        return _run_on_pdf(pdf_path, output_dir)
    return _run_on_images(page_images, output_dir)


def _run_on_pdf(pdf_path: Path, output_dir: Path) -> InferenceResult:
    def _execute() -> dict[str, str]:
        from paddleocr import PaddleOCRVL

        logger.info("Loading PaddleOCR-VL pipeline...")
        pipeline = PaddleOCRVL()
        results = list(pipeline.predict(str(pdf_path)))

        artifacts: dict[str, str] = {}
        for index, result in enumerate(results):
            page_dir = output_dir / f"result_{index:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            result.save_to_json(save_path=str(page_dir))
            result.save_to_markdown(save_path=str(page_dir))
            result.save_to_img(save_path=str(page_dir))
            artifacts[f"result_{index:04d}"] = str(page_dir)

        return artifacts

    return timed_run("paddleocr-vl-1.5", output_dir, _execute)


def _run_on_images(page_images: list[Path], output_dir: Path) -> InferenceResult:
    def _execute() -> dict[str, str]:
        from paddleocr import PaddleOCRVL

        logger.info("Loading PaddleOCR-VL pipeline...")
        pipeline = PaddleOCRVL()
        artifacts: dict[str, str] = {}

        for image_path in page_images:
            page_index = int(image_path.stem.split("_")[-1])
            page_dir = output_dir / f"page_{page_index:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)

            results = list(pipeline.predict(str(image_path)))
            for result in results:
                result.save_to_json(save_path=str(page_dir))
                result.save_to_markdown(save_path=str(page_dir))
                result.save_to_img(save_path=str(page_dir))

            artifacts[f"page_{page_index:04d}"] = str(page_dir)

        return artifacts

    return timed_run("paddleocr-vl-1.5", output_dir, _execute)
