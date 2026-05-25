"""
Pass 1: discover which MDF markers this dictionary uses.

Output is a compact marker list + structure rules. Cached per stage-2 experiment as
``outputs/stage-2/<experiment>/field_cheatsheet.json``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from dictextractor.llm.client import complete
from dictextractor.llm.mdf_marker_reference import MDF_MARKER_REFERENCE
from dictextractor.schemas.dictionary_languages import DictionaryLanguagesConfig
from dictextractor.schemas.field_cheatsheet import DictionaryMarkerCheatsheet
from dictextractor.utils.image import image_data_url, resolve_mime_type

logger = logging.getLogger(__name__)

DISCOVERY_SYSTEM = f"""\
You analyse dictionary front matter and a sample page to list which SIL Toolbox MDF
markers this dictionary uses and how entries are structured.

Output a single JSON object — NOT a full typography map. For each marker that appears
in this dictionary, give a one-line description (role + language tier). Add structure
rules for entry boundaries (homographs, senses, subentries, gloss lines).

Use MDF marker codes from the reference below. Pick gloss/usage/definition tiers to
match the dictionary's target languages (e=English, n=national, r=regional, v=vernacular).
Do NOT assign English-tier markers (ge, de, ue) when the dictionary has no English target.

{MDF_MARKER_REFERENCE}
"""

DISCOVERY_USER_TEMPLATE = """\
Study this dictionary and return a marker cheat sheet JSON object.

Inputs:
1. Introduction pages (images below).
2. One sample dictionary page (image + flat transcription).
{config_hint}

Return JSON with this shape:
{{
  "dictionary_name": "short name",
  "markers": [
    {{"marker": "lx", "description": "headword / lexeme (vernacular lemma)"}},
    {{"marker": "gn", "description": "Russian gloss / translation"}},
    {{"marker": "un", "description": "usage note (parenthetical Russian expansion)"}}
  ],
  "rules": [
    "Separate homograph mains: two \\\\lx blocks with same lemma, each with \\\\hm N.",
    "Numbered senses: one \\\\lx, then \\\\sn 1, \\\\gn …, \\\\sn 2, \\\\gn …",
    "Run-on sub-entries: \\\\se sub-lemma then \\\\gn … inside the same record block."
  ],
  "abbreviations": {{"мн.": "plural"}}
}}

Include ONLY markers that appear on the sample page or are defined in the intro.
Omit markers not used. Keep descriptions short (one line each). Rules should cover
entry structure and gloss-line conventions (semicolons, hyphen rejoining).
Return JSON only — no markdown fences.

<sample_transcription>
{transcription}
</sample_transcription>
"""


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in field discovery response.")
    return json.loads(text[start : end + 1])


def _config_hint(config: Optional[DictionaryLanguagesConfig]) -> str:
    if config is None:
        return ""
    target_names = ", ".join(t.language for t in config.targets)
    return (
        f"\n3. Language roles: layout={config.layout}, "
        f"source={config.source.language}, targets=[{target_names}].\n"
    )


def discover_field_cheatsheet(
    *,
    transcription: str,
    sample_image: Path,
    intro_images: List[Path],
    model: str,
    reasoning_effort: str = "high",
    languages_config: Optional[DictionaryLanguagesConfig] = None,
    dictionary_name: str = "",
) -> DictionaryMarkerCheatsheet:
    """Pass 1: discover markers + rules for this dictionary."""
    user_text = DISCOVERY_USER_TEMPLATE.format(
        transcription=transcription.strip(),
        config_hint=_config_hint(languages_config),
    )
    content: list[dict] = [{"type": "text", "text": user_text}]
    for intro_img in intro_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url(str(intro_img), resolve_mime_type(str(intro_img)))
                },
            }
        )
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": image_data_url(str(sample_image), resolve_mime_type(str(sample_image)))
            },
        }
    )
    messages = [
        {"role": "system", "content": DISCOVERY_SYSTEM},
        {"role": "user", "content": content},
    ]
    logger.info("Field discovery: model=%s sample=%s", model, sample_image.name)
    raw = complete(model=model, messages=messages, reasoning_effort=reasoning_effort)  # type: ignore[arg-type]
    data = _extract_json_object(raw)
    sheet = DictionaryMarkerCheatsheet.model_validate(data)
    if dictionary_name and not sheet.dictionary_name:
        sheet = sheet.model_copy(update={"dictionary_name": dictionary_name})
    return sheet


def gold_cheatsheet_path(entry_dir: Path) -> Path:
    """Path to the human-authored marker cheat sheet under ``outputs/stage-2-gold/``."""
    return entry_dir / "outputs" / "stage-2-gold" / "field_cheatsheet.json"


def load_gold_cheatsheet(entry_dir: Path) -> DictionaryMarkerCheatsheet:
    """Load the gold Pass-1 marker cheat sheet for a dictionary entry."""
    path = gold_cheatsheet_path(entry_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Gold field cheat sheet not found: {path}")
    logger.info("Loading gold marker cheat sheet: %s", path)
    return DictionaryMarkerCheatsheet.model_validate_json(path.read_text(encoding="utf-8"))


def load_or_discover_cheatsheet(
    cache_path: Path,
    *,
    force_refresh: bool = False,
    **discover_kwargs,
) -> DictionaryMarkerCheatsheet:
    """Load cached cheat sheet or run Pass 1 and save."""
    if cache_path.is_file() and not force_refresh:
        logger.info("Loading cached marker cheat sheet: %s", cache_path)
        return DictionaryMarkerCheatsheet.model_validate_json(
            cache_path.read_text(encoding="utf-8")
        )
    sheet = discover_field_cheatsheet(**discover_kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(sheet.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved marker cheat sheet → %s", cache_path)
    return sheet
