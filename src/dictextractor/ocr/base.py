"""
Abstract base class for all OCR backends.
Every backend must implement run() and return an OCRPageResult.
"""

from abc import ABC, abstractmethod
from dictextractor.schemas.ocr_result import OCRPageResult


class OCRBackend(ABC):
    """
    Plugin interface for OCR backends.

    To add a new backend:
    1. Create a new file in ocr/ (e.g. ocr/tesseract.py).
    2. Subclass OCRBackend and implement run().
    3. Optionally override the property methods to declare capabilities.

    The extraction strategies only consume OCRPageResult, so any backend
    can be paired with any extraction strategy without code changes.
    """

    @abstractmethod
    def run(self, image_path: str, **kwargs) -> OCRPageResult:
        """
        Run OCR on a single image.

        Args:
            image_path: Path to the image to process.
            **kwargs: Backend-specific keyword arguments.

        Returns:
            OCRPageResult containing detected blocks and/or raw text.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier used in reports and output paths."""
        ...

    @property
    def requires_api_key(self) -> bool:
        """True if this backend needs a remote API key to function."""
        return False

    @property
    def supports_layout_analysis(self) -> bool:
        """True if this backend performs document layout analysis (columns, blocks, etc.)."""
        return False

    @property
    def is_open_source(self) -> bool:
        """True if the underlying model weights are publicly available."""
        return True
