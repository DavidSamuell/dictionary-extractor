"""OCR layout adapters → flat stage-1 transcript (spec v2)."""

from dictextractor.ocr.adapters.flat_export import (
    page_dir_to_flat_lines,
    write_stage1_flat_for_page,
)
from dictextractor.ocr.adapters.layout_to_transcript_v1 import (
    FlatTranscriptParts,
    layout_to_transcript_v1,
)

__all__ = [
    "FlatTranscriptParts",
    "layout_to_transcript_v1",
    "page_dir_to_flat_lines",
    "write_stage1_flat_for_page",
]
