"""
Abstract base class for all extraction strategies.
Every strategy must implement extract() and return a DictionaryPage.
"""

from abc import ABC, abstractmethod
from dictextractor.schemas.entry import DictionaryPage
from dictextractor.schemas.ocr_result import OCRPageResult


class ExtractionStrategy(ABC):
    """
    Plugin interface for extraction strategies.

    An extraction strategy receives OCR output + the source image and
    produces a structured DictionaryPage.

    To add a new strategy:
    1. Create a new file in extraction/ (e.g. extraction/my_strategy.py).
    2. Subclass ExtractionStrategy and implement extract().
    3. Pass the instance to the CLI or pipeline runner.
    """

    @abstractmethod
    def extract(
        self,
        ocr_result: OCRPageResult,
        image_path: str,
        page_number: int = 1,
        **kwargs,
    ) -> DictionaryPage:
        """
        Extract structured dictionary entries from OCR output + source image.

        Args:
            ocr_result: Unified OCR output from any OCRBackend.
            image_path: Path to the original (or preprocessed) image.
            page_number: Page number for provenance tracking.
            **kwargs: Strategy-specific keyword arguments.

        Returns:
            DictionaryPage containing all extracted entries.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name used in report filenames."""
        ...
