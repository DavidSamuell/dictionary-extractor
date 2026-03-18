"""
Join method extraction strategy.

Pipeline:
  Stage 1 — LLM(image + intro page) → structure description
  Stage 2 — LLM(structure description + alphabet + OCR text + image) → structured JSON output

This strategy automates prompt construction by first inferring the dictionary's
structural conventions from the image, then using that description to guide extraction.
"""

import json
from typing import List, Optional

from dictextractor.extraction.base import ExtractionStrategy
from dictextractor.schemas.entry import DictionaryEntry, DictionaryPage
from dictextractor.schemas.ocr_result import OCRPageResult
from dictextractor.llm import client as llm
from dictextractor.llm.prompts import (
    JOIN_STRUCTURE_SYSTEM_PROMPT,
    JOIN_STRUCTURE_USER_PROMPT,
    join_extraction_system_prompt,
    join_extraction_user_prompt,
)
from dictextractor.utils.image import image_data_url, resolve_mime_type


class JoinLLMExtraction(ExtractionStrategy):
    """
    Join method: the LLM first describes the dictionary structure from the image,
    then uses that description to extract and structure entries.
    """

    def __init__(
        self,
        model: str = "openrouter/qwen/qwen3-vl-30b-a3b-thinking",
        structure_model: Optional[str] = None,
        alphabet_hint: str = "",
    ):
        self.model = model
        self.structure_model = structure_model or model
        self.alphabet_hint = alphabet_hint

    @property
    def name(self) -> str:
        return "llm_join"

    def extract(
        self,
        ocr_result: OCRPageResult,
        image_path: str,
        page_number: int = 1,
        intro_image_path: Optional[str] = None,
        **kwargs,
    ) -> DictionaryPage:
        """
        Two-call extraction pipeline.

        Args:
            ocr_result: OCR output from any backend.
            image_path: Path to the dictionary page image.
            page_number: Page number for provenance.
            intro_image_path: Optional intro/legend page image to help infer structure.
        """
        mime = resolve_mime_type(image_path)
        data_url = image_data_url(image_path, mime)

        # Stage 1: infer structure
        structure_image = image_data_url(intro_image_path, resolve_mime_type(intro_image_path)) \
            if intro_image_path else data_url

        structure_messages = [
            {"role": "system", "content": JOIN_STRUCTURE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": JOIN_STRUCTURE_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": structure_image}},
                ],
            },
        ]
        print("Stage 1: Inferring dictionary structure...")
        structure_description = llm.complete(model=self.structure_model, messages=structure_messages)
        print(f"Structure description:\n{structure_description[:400]}...")

        # Stage 2: extract with the inferred structure
        ocr_text = ocr_result.raw_text or "\n".join(
            b.content for b in ocr_result.text_blocks
        )
        extraction_messages = [
            {"role": "system", "content": join_extraction_system_prompt(structure_description)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": join_extraction_user_prompt(ocr_text, self.alphabet_hint)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        print("Stage 2: Extracting entries using inferred structure...")
        content = llm.complete(model=self.model, messages=extraction_messages)
        entries = self._parse_response(content)

        return DictionaryPage(entries=entries, page_number=page_number, source_file=image_path)

    def _parse_response(self, content: str) -> List[DictionaryEntry]:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        raw = json.loads(content)
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
        return entries
