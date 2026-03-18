"""
Manual LLM extraction strategy.
Uses a hand-tuned prompt with explicit knowledge of the Chukchi dictionary structure.
Migrated from src/extract_dictionary.py.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

from dictextractor.extraction.base import ExtractionStrategy
from dictextractor.schemas.entry import DictionaryEntry, DictionaryPage
from dictextractor.schemas.ocr_result import OCRPageResult
from dictextractor.llm import client as llm
from dictextractor.llm.prompts import MANUAL_SYSTEM_PROMPT, manual_user_prompt
from dictextractor.utils.image import image_data_url, resolve_mime_type
from dictextractor.utils.io import read_text_file


class ManualLLMExtraction(ExtractionStrategy):
    """
    Extraction strategy that sends the image plus OCR text to an LLM with
    a detailed, hand-tuned system prompt describing the Chukchi dictionary structure.

    This is the current baseline strategy.
    """

    def __init__(self, model: str = "openrouter/qwen/qwen3-vl-30b-a3b-thinking"):
        self.model = model

    @property
    def name(self) -> str:
        return "llm_manual"

    # ------------------------------------------------------------------
    # ExtractionStrategy interface
    # ------------------------------------------------------------------

    def extract(
        self,
        ocr_result: OCRPageResult,
        image_path: str,
        page_number: int = 1,
        **kwargs,
    ) -> DictionaryPage:
        """
        Extract structured entries from the full page image + OCR text.

        If the OCR result contains multiple text blocks (e.g. from PaddleOCR VL),
        each block is processed individually and the results are merged.
        """
        text_blocks = ocr_result.text_blocks

        if text_blocks:
            # Per-block extraction (PaddleOCR VL or similar layout-aware backends)
            return self._extract_from_blocks(text_blocks, image_path, page_number)
        else:
            # Whole-page extraction (Mathpix plain text or no OCR)
            ocr_text = ocr_result.raw_text
            return self._extract_single_image(image_path, ocr_text, page_number)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_from_blocks(self, blocks, image_path: str, page_number: int) -> DictionaryPage:
        """Process each OCR block independently and merge all entries."""
        from PIL import Image

        all_entries: List[DictionaryEntry] = []

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img = Image.open(image_path)

            print(f"Found {len(blocks)} text blocks to process")
            print("=" * 60)

            for idx, block in enumerate(blocks, 1):
                if not block.content:
                    print(f"\n[{idx}/{len(blocks)}] Block {block.block_id}: no content, skipping")
                    continue

                print(f"\n[{idx}/{len(blocks)}] Processing block {block.block_id}...")

                b = block.bbox
                if block.crop_image is not None:
                    import cv2
                    crop_path = tmp_path / f"block_{block.block_id}.png"
                    cv2.imwrite(str(crop_path), block.crop_image)
                    block_image_path = str(crop_path)
                else:
                    cropped = img.crop((b.x_min, b.y_min, b.x_max, b.y_max))
                    crop_path = tmp_path / f"block_{block.block_id}.png"
                    cropped.save(crop_path)
                    block_image_path = str(crop_path)

                try:
                    page = self._extract_single_image(block_image_path, block.content, page_number)
                    all_entries.extend(page.entries)
                    print(f"  Extracted {len(page.entries)} entries from block")
                except Exception as e:
                    print(f"  Error processing block {block.block_id}: {e}")

        print("\n" + "=" * 60)
        print(f"Total entries extracted from all blocks: {len(all_entries)}")
        return DictionaryPage(entries=all_entries, page_number=page_number, source_file=image_path)

    def _extract_single_image(
        self, image_path: str, ocr_text: str = "", page_number: int = 1
    ) -> DictionaryPage:
        """Send one image to the LLM and parse the JSON response."""
        mime = resolve_mime_type(image_path)
        data_url = image_data_url(image_path, mime)

        messages = [
            {"role": "system", "content": MANUAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": manual_user_prompt(ocr_text)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        content = llm.complete(model=self.model, messages=messages)
        entries = self._parse_response(content, image_path)
        return DictionaryPage(entries=entries, page_number=page_number, source_file=image_path)

    def _parse_response(self, content: str, source: str) -> List[DictionaryEntry]:
        """Parse and validate the LLM JSON response into DictionaryEntry objects."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Content (first 500 chars): {content[:500]}")
            raise

        if isinstance(raw, dict) and "entries" in raw:
            raw = raw["entries"]
        elif isinstance(raw, dict):
            raw = [raw]

        entries = []
        for item in raw:
            cleaned = {
                "headword_phrase": item.get("headword_phrase") or "",
                "entry_type": item.get("entry_type") or "word",
                "pos": item.get("pos") or "",
                "translation_ru": item.get("translation_ru") or "",
                "literal_meaning": item.get("literal_meaning") or "",
                "grammar_notes": item.get("grammar_notes") or "",
            }
            if cleaned["headword_phrase"]:
                entries.append(DictionaryEntry(**cleaned))

        print(f"Validated {len(entries)} entries (filtered {len(raw) - len(entries)} incomplete)")
        return entries
