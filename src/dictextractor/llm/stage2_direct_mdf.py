"""
Pass 2: direct page transcription → Toolbox MDF text using a field profile.

Used by ``TwoStageLLMExtraction`` when ``stage2_mode='direct_mdf'``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from dictextractor.llm import client as llm
from dictextractor.llm.mdf_marker_reference import MDF_MARKER_REFERENCE
from dictextractor.schemas.field_map import FieldMapPrompt
from dictextractor.utils.image import image_data_url, resolve_mime_type
from dictextractor.utils.mdf_export import normalize_mdf_text
from dictextractor.utils.pdf_render import needs_pdf_rasterization

logger = logging.getLogger(__name__)

DIRECT_MDF_SYSTEM = """\
You digitize dictionary pages into SIL Toolbox MDF (Multi-Dictionary Formatter) text.

The flat transcription is human gold-label OCR text. Copy every vernacular and gloss
character from it exactly — do NOT re-read, correct, normalise, or substitute letters
from the page image. Use the image and introduction only to decide entry boundaries,
field roles, and MDF marker assignment.

Output ONLY MDF text — no JSON, no markdown fences, no commentary.
Use one blank line between lexicon records (between main entry blocks).

Allowed changes (structure only — never alter alphabet/characters):
  - Strip typography markup (<b>, <i>) when emitting MDF lines; do not change the
    underlying characters inside those spans.
  - Rejoin hyphenated line breaks from the transcription when forming headwords/glosses.
  - Normalise sense/homograph numbers: \\sn and \\hm as integers 1, 2, 3, …
    (convert Roman numerals; strip trailing ')' or '.').
  - Strip end-of-sentence punctuation (., !, ?) from gloss, definition, and example
    field values — MDF does not require them.
  - Split or merge lines across MDF marker fields per the field map (e.g. \\un vs \\gn).
  - For note fields (\\np, \\nt, \\ng, etc.): when the field map or transcription includes a
    printed English label (e.g. "Tone:", "Commonly used classifier:"), keep that label in the
    field value unless the field map explicitly says otherwise.

Do NOT invent entries or text spans absent from the transcription.
"""

DIRECT_MDF_USER_TEMPLATE = """\
Dictionary page extraction.

Inputs:
1. Gold-label OCR transcription (bold = <b>…</b>, italic = <i>…</i>) or column TSV
   (column_id, line_number, text). Treat this as authoritative for all characters.
2. Photo of the dictionary page (image below) — layout and field-boundary reference only.
3. Dictionary introduction pages (following images).
{toolbox_section}

{field_block}

Parse every dictionary entry on the page into MDF text using the field map above.
Copy headword and gloss characters verbatim from the transcription except for the
structural normalisations listed in the system prompt.

<transcription>
{transcription}
</transcription>
{guides_block}
"""


def strip_markdown_fences(text: str) -> str:
    """Remove optional markdown code fences from model output."""
    text = text.strip()
    fence = re.search(r"```(?:mdf|text)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _toolbox_content_parts(
    toolbox_pdf: Path,
    *,
    model: str,
) -> tuple[str, list[dict]]:
    """Build prompt section and optional vision parts for the Toolbox MDF manual."""
    if needs_pdf_rasterization(model):
        # OpenRouter / OpenAI cannot ingest PDF; attaching every rasterized page
        # (~65 images) routinely blows prompt limits and causes empty API responses.
        logger.info(
            "Toolbox PDF %s attached as text reference for model %s",
            toolbox_pdf.name,
            model,
        )
        section = (
            "4. SIL Toolbox MDF Reference Manual (text excerpt) — marker reference:\n\n"
            f"{MDF_MARKER_REFERENCE}\n"
        )
        return section, []

    section = (
        "4. Attached SIL Toolbox MDF Reference Manual (PDF) — marker reference.\n"
    )
    return section, [
        {
            "type": "image_url",
            "image_url": {
                "url": image_data_url(str(toolbox_pdf), "application/pdf"),
            },
        }
    ]


def build_direct_mdf_messages(
    *,
    transcription: str,
    image_path: str,
    intro_image_paths: List[str],
    field_map: FieldMapPrompt,
    model: str,
    guides: str = "",
    toolbox_pdf: Optional[Path] = None,
) -> list[dict]:
    """Build LLM messages for Pass 2 direct MDF extraction."""
    guides_block = f"\n\nUSER DEFINED GUIDELINES\n{guides}" if guides.strip() else ""
    toolbox_section = ""
    toolbox_parts: list[dict] = []
    if toolbox_pdf and toolbox_pdf.is_file():
        toolbox_section, toolbox_parts = _toolbox_content_parts(
            toolbox_pdf,
            model=model,
        )
    user_text = DIRECT_MDF_USER_TEMPLATE.format(
        transcription=transcription.strip(),
        field_block=field_map.format_prompt_block(),
        guides_block=guides_block,
        toolbox_section=toolbox_section,
    )
    mime = resolve_mime_type(image_path)
    content: list[dict] = [{"type": "text", "text": user_text}]
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": image_data_url(image_path, mime)},
        }
    )
    for intro_img in intro_image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url(intro_img, resolve_mime_type(intro_img))
                },
            }
        )
    content.extend(toolbox_parts)
    return [
        {"role": "system", "content": DIRECT_MDF_SYSTEM},
        {"role": "user", "content": content},
    ]


def extract_direct_mdf(
    *,
    transcription: str,
    image_path: str,
    intro_image_paths: List[str],
    field_map: FieldMapPrompt,
    model: str,
    reasoning_effort: str,
    guides: str = "",
    toolbox_pdf: Optional[Path] = None,
) -> tuple[str, str, dict, list]:
    """
    Run Pass 2 direct MDF extraction.

    Returns:
        (mdf_text, raw_response, usage_dict, sanitized_messages)
    """
    messages = build_direct_mdf_messages(
        transcription=transcription,
        image_path=image_path,
        intro_image_paths=intro_image_paths,
        field_map=field_map,
        model=model,
        guides=guides,
        toolbox_pdf=toolbox_pdf,
    )
    raw, usage = llm.complete_with_usage(
        model=model,
        messages=messages,
        reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
    )
    mdf_text = normalize_mdf_text(strip_markdown_fences(raw))
    return mdf_text, raw, usage, messages
