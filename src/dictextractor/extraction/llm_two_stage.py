"""
Two-stage extraction strategy.

Pipeline
--------
Stage 1 — Transcription  (low/minimal reasoning, structured output)
  Inputs : image + alphabet (text or image) + optional OCR hint (txt/md/docx)
  Task   : Faithfully reproduce every character visible on the page (no interpretation).
  Output : TranscriptionResponse → list of lines → joined into plain text

Stage 2 — Structuring  (high reasoning, structured output)
  Inputs : Stage-1 transcript + dictionary intro text/images + image of the page
  Task   : Identify entities, infer the entry structure, and map to the schema.
  Output : EntriesResponse → List[DictionaryEntry]

Reasoning budget rationale
--------------------------
Stage 1 is a copying/transcription task — creativity and inference are harmful.
  → reasoning_effort="low"  (maps to thinking_level: low on Gemini 3)

Stage 2 requires understanding multi-column layouts, abbreviations, cross-references,
and mapping ambiguous text spans to typed schema fields.
  → reasoning_effort="high"  (maps to thinking_level: high on Gemini 3)

Structured output rationale
---------------------------
Both stages use response_format with a Pydantic schema enforced by the API:
  - Stage 1: TranscriptionResponse(lines: List[str])
      Forces line-by-line enumeration; structurally prevents preamble/postamble.
  - Stage 2: EntriesResponse(entries: List[DictionaryEntry])
      Guarantees valid JSON matching the schema; eliminates all parsing heuristics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from dictextractor.extraction.base import ExtractionStrategy
from dictextractor.schemas.entry import (
    DictionaryEntry,
    DictionaryPage,
    EntriesResponse,
    TranscriptionResponse,
    ColumnTranscription,
)
from dictextractor.schemas.ocr_result import OCRPageResult
from dictextractor.llm import client as llm
from dictextractor.llm.prompts import (
    TWO_STAGE_TRANSCRIBE_SYSTEM,
    TWO_STAGE_STRUCTURE_USER,
    two_stage_transcribe_user,
    two_stage_structure_system_prompt,
)
from dictextractor.utils.image import image_data_url, resolve_mime_type
from dictextractor.utils.io import read_docx_text


def _sum_costs(c1, c2) -> Optional[float]:
    """Sum two nullable cost values."""
    if c1 is None and c2 is None:
        return None
    return round((c1 or 0.0) + (c2 or 0.0), 8)


def _print_usage_summary(s1: dict, s2: dict, total: Optional[float]) -> None:
    print("\n  ── Usage ──────────────────────────────────────────")
    for label, u in [("Stage 1", s1), ("Stage 2", s2)]:
        img = f"  img={u.get('image_tokens')}" if u.get("image_tokens") else ""
        cost = f"  ${u.get('cost_usd'):.6f}" if u.get("cost_usd") is not None else ""
        print(f"  {label}: {u.get('total_tokens')} tokens{img}{cost}")
    if total is not None:
        print(f"  Page total: ${total:.6f}")
    print()


def _transcription_to_tsv(result: TranscriptionResponse) -> str:
    """
    Flatten a column-aware TranscriptionResponse into a TSV string.

    Format: column_id \\t line_number \\t text
    Line numbers reset to 1 at the start of each column.

    This format is passed verbatim to Stage 2 — each line is unambiguously
    labelled with its column and position, so the structuring LLM never needs
    to infer column boundaries from separator text.

    Example output for a two-column page:
        column_id\\tline_number\\ttext
        left\\t1\\tac-úkwʌn (сущ.) кремень
        left\\t2\\tбукв. жирный камень
        right\\t1\\tac-ékwəŋ (гл.) дробить
        right\\t2\\tсм. ac/æc
    """
    rows = ["column_id\tline_number\ttext"]
    for col in result.columns:
        for i, line in enumerate(col.lines, start=1):
            rows.append(f"{col.column_id}\t{i}\t{line}")
    return "\n".join(rows)


class TwoStageLLMExtraction(ExtractionStrategy):
    """
    Two-stage strategy: Stage 1 transcribes faithfully, Stage 2 structures the result.

    Args:
        transcribe_model:   Model used for Stage 1 (transcription).
        structure_model:    Model used for Stage 2 (structuring). Defaults to transcribe_model.
        alphabet_path:      Path to the alphabet file (.txt / .png / .jpg).
                            If an image, it is sent as a vision input to Stage 1.
                            If text, it is embedded in the prompt.
        intro_text:         Introduction/preface of the dictionary (plain text). Passed to
                            Stage 2 so the model understands conventions and abbreviations.
        intro_image_paths:  Intro page images sent as vision context to Stage 2.
                            Loaded once, shared across all pages.
    """

    def __init__(
        self,
        transcribe_model: str = "gemini/gemini-3-flash-preview",
        structure_model: Optional[str] = None,
        alphabet_path: Optional[str] = None,
        intro_text: str = "",
        intro_image_paths: Optional[List[str]] = None,
    ):
        self.transcribe_model = transcribe_model
        self.structure_model = structure_model or transcribe_model
        self.alphabet_path = alphabet_path
        self.intro_text = intro_text
        self.intro_image_paths = intro_image_paths or []

    @property
    def name(self) -> str:
        return "llm_two_stage"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        ocr_result: OCRPageResult,
        image_path: str,
        page_number: int = 1,
        intro_text: Optional[str] = None,
        stage1_output_path: Optional[str] = None,
        **kwargs,
    ) -> DictionaryPage:
        """
        Run the two-stage pipeline.

        Args:
            ocr_result:          OCR output from any backend. The plain text is passed
                                 to Stage 1 as an optional character-shape reference.
            image_path:          Path to the dictionary page image.
            page_number:         Page number for provenance.
            intro_text:          Override for the instance-level intro_text (if provided).
            stage1_output_path:  If given, the Stage 1 transcription is written to this
                                 path as a UTF-8 .txt file before Stage 2 runs.
                                 Useful for inspection and avoiding repeat API calls.
        """
        effective_intro = intro_text if intro_text is not None else self.intro_text

        # ── Stage 1: transcription ─────────────────────────────────────────────
        print("=" * 60)
        print("Stage 1: Transcribing page image …")
        transcribed_text, stage1_raw, stage1_usage = self._stage1_transcribe(ocr_result, image_path)
        print(f"Transcription ({len(transcribed_text)} chars):\n{transcribed_text[:500]}…\n")

        if stage1_output_path:
            base = Path(stage1_output_path)
            base.parent.mkdir(parents=True, exist_ok=True)
            base.write_text(transcribed_text, encoding="utf-8")
            raw1_path = base.with_name(base.stem + "_raw.json")
            raw1_path.write_text(stage1_raw, encoding="utf-8")
            print(f"Stage 1 saved → {base.name}  |  raw → {raw1_path.name}")

        # ── Stage 2: structuring ───────────────────────────────────────────────
        print("Stage 2: Structuring transcribed text …")
        entries, stage2_raw, stage2_usage = self._stage2_structure(
            transcribed_text, image_path, effective_intro, self.intro_image_paths
        )
        print(f"Extracted {len(entries)} entries.")

        if stage1_output_path:
            base = Path(stage1_output_path)
            raw2_path = base.with_name(base.stem.replace("_stage1", "") + "_stage2_raw.json")
            raw2_path.write_text(stage2_raw, encoding="utf-8")
            print(f"Stage 2 raw saved → {raw2_path.name}")

            # ── Per-page usage summary ─────────────────────────────────────────
            total_cost = _sum_costs(stage1_usage.get("cost_usd"), stage2_usage.get("cost_usd"))
            page_usage = {
                "stage1": stage1_usage,
                "stage2": stage2_usage,
                "total_cost_usd": total_cost,
            }
            usage_path = base.with_name(base.stem.replace("_stage1", "") + "_usage.json")
            usage_path.write_text(
                json.dumps(page_usage, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _print_usage_summary(stage1_usage, stage2_usage, total_cost)

        return DictionaryPage(entries=entries, page_number=page_number, source_file=image_path)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage1_transcribe(self, ocr_result: OCRPageResult, image_path: str) -> tuple[str, str, dict]:
        """
        Call the transcription LLM (Stage 1) with structured output.
        Returns (transcribed_text, raw_json_str, usage_dict).
        """
        mime = resolve_mime_type(image_path)
        page_data_url = image_data_url(image_path, mime)

        alphabet_text, alphabet_image_url = self._load_alphabet()
        ocr_hint = ocr_result.raw_text if ocr_result else ""

        user_text = two_stage_transcribe_user(
            alphabet_text=alphabet_text,
            ocr_hint=ocr_hint,
        )

        content: list = [{"type": "text", "text": user_text}]
        if alphabet_image_url:
            content.append({"type": "image_url", "image_url": {"url": alphabet_image_url}})
        content.append({"type": "image_url", "image_url": {"url": page_data_url}})

        messages = [
            {"role": "system", "content": TWO_STAGE_TRANSCRIBE_SYSTEM},
            {"role": "user", "content": content},
        ]

        result, raw, usage = llm.complete_structured(
            model=self.transcribe_model,
            messages=messages,
            response_schema=TranscriptionResponse,
            reasoning_effort="low",
        )
        return _transcription_to_tsv(result), raw, usage

    def _stage2_structure(
        self,
        transcribed_text: str,
        image_path: str,
        intro_text: str,
        intro_image_paths: Optional[List[str]] = None,
    ) -> tuple[List[DictionaryEntry], str, dict]:
        """
        Call the structuring LLM (Stage 2) with structured output.
        Returns (entries, raw_json_str, usage_dict).
        """
        mime = resolve_mime_type(image_path)
        page_data_url = image_data_url(image_path, mime)

        system_prompt = two_stage_structure_system_prompt(
            transcribed_text=transcribed_text,
            intro_text=intro_text,
        )

        content: list = [
            {"type": "text", "text": TWO_STAGE_STRUCTURE_USER},
            {"type": "image_url", "image_url": {"url": page_data_url}},
        ]
        for intro_img in (intro_image_paths or []):
            intro_mime = resolve_mime_type(intro_img)
            content.append(
                {"type": "image_url", "image_url": {"url": image_data_url(intro_img, intro_mime)}}
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        result, raw, usage = llm.complete_structured(
            model=self.structure_model,
            messages=messages,
            response_schema=EntriesResponse,
            reasoning_effort="high",
        )
        return result.entries, raw, usage

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_alphabet(self) -> tuple[str, Optional[str]]:
        """
        Load the alphabet hint.  Returns (alphabet_text, alphabet_image_data_url).
        Exactly one of the two will be non-empty; the other will be ""/None.
        """
        if not self.alphabet_path:
            return "", None

        p = Path(self.alphabet_path)
        suffix = p.suffix.lower()

        if suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            mime = resolve_mime_type(str(p))
            return "", image_data_url(str(p), mime)

        if suffix == ".docx":
            return read_docx_text(str(p)), None

        # .txt / .md / anything else — read as plain text
        return p.read_text(encoding="utf-8"), None
