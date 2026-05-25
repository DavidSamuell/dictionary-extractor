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
  → reasoning_effort="medium" by default (maps to thinking_level: medium on Gemini 3).
  Tunable via the strategy ctor / CLI: too-high reasoning has been observed to
  leak chain-of-thought into JSON string fields under structured output, so we
  default to medium and only bump up explicitly when needed.

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

from dictextractor.evaluation.stage1.flatten import flat_transcription_to_text
from dictextractor.extraction.base import ExtractionStrategy
from dictextractor.llm.field_discovery import load_gold_cheatsheet, load_or_discover_cheatsheet
from dictextractor.llm.stage2_direct_mdf import extract_direct_mdf
from dictextractor.schemas.field_map import FieldMapPrompt
from dictextractor.utils.stage1_input import read_stage1_transcript_text
from dictextractor.schemas.dictionary_languages import DictionaryLanguagesConfig
from dictextractor.schemas.entry import (
    DictionaryEntry,
    DictionaryPage,
    EntriesResponse,
    FlatTranscriptionResponse,
    TranscriptionResponse,
)
from dictextractor.schemas.ocr_result import OCRPageResult
from dictextractor.llm import client as llm
from dictextractor.llm.prompts import (
    STAGE_1_FLAT_SYSTEM,
    STAGE_1_SYSTEM,
    STAGE_2_SYSTEM,
    stage_1_user,
    stage_2_user,
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


def _sanitize_messages(messages: list) -> list:
    """
    Return a JSON-safe copy of the LLM messages with base64 image data replaced
    by a compact placeholder.  Keeps the full prompt text intact for debugging.
    """
    import copy, re
    sanitized = copy.deepcopy(messages)
    b64_pattern = re.compile(r"(data:[^;]+;base64,)(.+)")
    for msg in sanitized:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                m = b64_pattern.match(url)
                if m:
                    b64_len = len(m.group(2))
                    part["image_url"]["url"] = (
                        f"{m.group(1)}<{b64_len} chars omitted>"
                    )
    return sanitized


def _transcription_to_tsv(result: TranscriptionResponse) -> str:
    """
    Flatten a TranscriptionResponse into a TSV string.

    Format: column_id \\t line_number \\t text

    Header rows: column_id="header", line_number empty, one row per header line.
    Body rows:   column_id in {"left","center","right","single"}, line_number 1..N
                 within each column.
    Footer rows: column_id="footer", line_number empty, one row per footer line.

    Header/footer are page-level metadata (running title, page number, chapter
    abbreviation, etc.); Stage 2 ignores them, and Stage 1 evaluation excludes
    them from character/markup/read-order metrics.

    Example output for a two-column page with a header and a footer:
        column_id\\tline_number\\ttext
        header\\t\\tCHUKCHI-RUSSIAN DICTIONARY    A
        left\\t1\\tac-úkwʌn (сущ.) кремень
        left\\t2\\tбукв. жирный камень
        right\\t1\\tac-ékwəŋ (гл.) дробить
        right\\t2\\tсм. ac/æc
        footer\\t\\t— 12 —
    """
    rows = ["column_id\tline_number\ttext"]
    for line in result.header or []:
        rows.append(f"header\t\t{line}")
    for col in result.columns:
        for i, line in enumerate(col.lines, start=1):
            rows.append(f"{col.column_id}\t{i}\t{line}")
    for line in result.footer or []:
        rows.append(f"footer\t\t{line}")
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
        discover_extra_fields: bool = False,
        stage1_reasoning_effort: str = "low",
        stage2_reasoning_effort: str = "low",
        stage1_guides: str = "",
        stage2_guides: str = "",
        stage1_mode: str = "column",
        stage2_mode: str = "direct_mdf",
        dictionary_languages: Optional[DictionaryLanguagesConfig] = None,
        entry_dir: Optional[str] = None,
        stage2_experiment_dir: Optional[str] = None,
        overwrite: bool = False,
        stage2_toolbox_pdf: Optional[str] = None,
        field_cheatsheet_gold: bool = False,
    ):
        if stage1_mode not in ("column", "flat"):
            raise ValueError(f"stage1_mode must be 'column' or 'flat', got {stage1_mode!r}")
        if stage2_mode not in ("schema", "direct_mdf"):
            raise ValueError(
                f"stage2_mode must be 'schema' or 'direct_mdf', got {stage2_mode!r}"
            )
        self.transcribe_model = transcribe_model
        self.structure_model = structure_model or transcribe_model
        self.alphabet_path = alphabet_path
        self.intro_text = intro_text
        self.intro_image_paths = intro_image_paths or []
        self.discover_extra_fields = discover_extra_fields
        self.stage1_reasoning_effort = stage1_reasoning_effort
        self.stage2_reasoning_effort = stage2_reasoning_effort
        self.stage1_guides = stage1_guides
        self.stage2_guides = stage2_guides
        self.stage1_mode = stage1_mode
        self.stage2_mode = stage2_mode
        self.dictionary_languages = dictionary_languages
        self.entry_dir = Path(entry_dir) if entry_dir else None
        self.stage2_experiment_dir = (
            Path(stage2_experiment_dir) if stage2_experiment_dir else None
        )
        self.overwrite = overwrite
        self.stage2_toolbox_pdf = (
            Path(stage2_toolbox_pdf) if stage2_toolbox_pdf else None
        )
        self.field_cheatsheet_gold = field_cheatsheet_gold
        self._field_map: Optional[FieldMapPrompt] = None

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
        stage2_output_path: Optional[str] = None,
        run_stage: str = "both",
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
            stage1_output_path:  Path for the Stage 1 transcription TSV (e.g.
                                 ``<...>/stage-1/<page>/<page>_stage1.tsv``). The Stage 1
                                 raw/input JSONs are written next to it.
                                 Stage-2-only reads the transcription from this path.
            stage2_output_path:  Path for the final Stage 2 TSV (e.g.
                                 ``<...>/stage-2/<page>/<page>.tsv``). The Stage 2
                                 raw/input/usage JSONs are derived from this path's
                                 stem. If omitted, Stage 2 artifacts fall back to
                                 living next to ``stage1_output_path`` (legacy layout).
            run_stage:           "1" = stage 1 only, "2" = stage 2 only, "both" = full pipeline.
                                 Stage-2-only reads transcription from stage1_output_path.
        """
        effective_intro = intro_text if intro_text is not None else self.intro_text
        stage1_usage: Dict[str, Any] = {}
        stage2_usage: Dict[str, Any] = {}
        entries: List[DictionaryEntry] = []
        mdf_text = ""
        discovery_usage: Dict[str, Any] = {}

        # ── Stage 1: transcription ─────────────────────────────────────────────
        if run_stage in ("1", "both"):
            print("=" * 60)
            print("Stage 1: Transcribing page image …")
            transcribed_text, stage1_raw, stage1_usage, stage1_msgs = (
                self._stage1_transcribe(ocr_result, image_path)
            )
            print(
                f"Transcription ({len(transcribed_text)} chars):\n{transcribed_text[:500]}…\n"
            )

            if stage1_output_path:
                base = Path(stage1_output_path)
                base.parent.mkdir(parents=True, exist_ok=True)
                base.write_text(transcribed_text, encoding="utf-8")
                stem_base = base.stem.replace("_stage1_flat", "").replace("_stage1", "")
                raw1_path = base.parent / f"{stem_base}_stage1_raw.json"
                input1_path = base.parent / f"{stem_base}_stage1_input.json"
                input1_path.write_text(
                    json.dumps(stage1_msgs, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"Stage 1 saved → {base.name}  |  raw → {raw1_path.name}  |  input → {input1_path.name}")
        elif run_stage == "2":
            if not stage1_output_path or not Path(stage1_output_path).exists():
                raise FileNotFoundError(
                    f"Stage-2-only requires existing stage 1 transcript: {stage1_output_path}"
                )
            transcribed_text = read_stage1_transcript_text(Path(stage1_output_path))
            print("=" * 60)
            print(
                f"Stage 2 only: loaded existing transcription from {stage1_output_path} "
                f"({len(transcribed_text)} chars)"
            )

        # ── Stage 2: structuring ───────────────────────────────────────────────
        stage2_base: Optional[Path] = None
        if run_stage in ("2", "both"):
            # Resolve where Stage 2 artifacts live:
            #   - explicit stage2_output_path → use it (its stem becomes the artifact prefix).
            #   - else fall back to stage1 dir with the "_stage1" suffix stripped (legacy).
            if stage2_output_path:
                stage2_base = Path(stage2_output_path)
            elif stage1_output_path:
                s1 = Path(stage1_output_path)
                stage2_base = s1.with_name(
                    s1.stem.replace("_stage1_flat", "").replace("_stage1", "")
                    + s1.suffix
                )

            print("Stage 2: Structuring transcribed text …")
            if self.stage2_mode == "direct_mdf":
                field_map = self._ensure_field_map(transcribed_text, image_path)
                mdf_text, stage2_raw, stage2_usage, stage2_msgs = self._stage2_direct_mdf(
                    transcribed_text,
                    image_path,
                    effective_intro,
                    self.intro_image_paths,
                    field_map,
                )
                print(f"Direct MDF ({len(mdf_text)} chars).")
            else:
                entries, stage2_raw, stage2_usage, stage2_msgs = self._stage2_structure(
                    transcribed_text, image_path, effective_intro, self.intro_image_paths
                )
                print(f"Extracted {len(entries)} entries.")

            if stage2_base:
                stage2_base.parent.mkdir(parents=True, exist_ok=True)
                raw_ext = "txt" if self.stage2_mode == "direct_mdf" else "json"
                raw2_path = stage2_base.with_name(
                    stage2_base.stem + f"_stage2_raw.{raw_ext}"
                )
                raw2_path.write_text(stage2_raw, encoding="utf-8")
                input2_path = stage2_base.with_name(stage2_base.stem + "_stage2_input.json")
                input2_path.write_text(
                    json.dumps(stage2_msgs, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"Stage 2 raw saved → {raw2_path.name}  |  input → {input2_path.name}")

        # ── Per-page usage summary ────────────────────────────────────────────
        usage_path: Optional[Path] = None
        if run_stage in ("2", "both") and stage2_base:
            usage_path = stage2_base.with_name(stage2_base.stem + "_usage.json")
        elif run_stage == "1" and stage1_output_path:
            s1 = Path(stage1_output_path)
            stem_base = s1.stem.replace("_stage1_flat", "").replace("_stage1", "")
            usage_path = s1.parent / f"{stem_base}_usage.json"

        if usage_path and (stage1_usage or stage2_usage or discovery_usage):
            total_cost = _sum_costs(
                _sum_costs(stage1_usage.get("cost_usd"), stage2_usage.get("cost_usd")),
                discovery_usage.get("cost_usd"),
            )
            page_usage = {
                "stage1": stage1_usage or None,
                "field_discovery": discovery_usage or None,
                "stage2": stage2_usage or None,
                "total_cost_usd": total_cost,
            }
            usage_path.parent.mkdir(parents=True, exist_ok=True)
            usage_path.write_text(
                json.dumps(page_usage, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if stage1_usage and stage2_usage:
                _print_usage_summary(stage1_usage, stage2_usage, total_cost)

        return DictionaryPage(
            entries=entries,
            page_number=page_number,
            source_file=image_path,
            mdf_text=mdf_text,
        )

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage1_transcribe(
        self, ocr_result: OCRPageResult, image_path: str
    ) -> tuple[str, str, dict, list]:
        """
        Call the transcription LLM (Stage 1) with structured output.
        Returns (transcribed_text, raw_json_str, usage_dict, sanitized_messages).
        """
        mime = resolve_mime_type(image_path)
        page_data_url = image_data_url(image_path, mime)

        alphabet_text, alphabet_image_url = self._load_alphabet()
        ocr_hint = ocr_result.raw_text if ocr_result else ""

        user_text = stage_1_user(
            alphabet_text=alphabet_text,
            ocr_hint=ocr_hint,
            guides=self.stage1_guides,
        )

        content: list = [{"type": "text", "text": user_text}]
        if alphabet_image_url:
            content.append(
                {"type": "image_url", "image_url": {"url": alphabet_image_url}}
            )
        content.append({"type": "image_url", "image_url": {"url": page_data_url}})

        if self.stage1_mode == "flat":
            messages = [
                {"role": "system", "content": STAGE_1_FLAT_SYSTEM},
                {"role": "user", "content": content},
            ]
            result, raw, usage = llm.complete_structured(
                model=self.transcribe_model,
                messages=messages,
                response_schema=FlatTranscriptionResponse,
                reasoning_effort=self.stage1_reasoning_effort,
            )
            flat_text = flat_transcription_to_text(
                result.header, result.lines, result.footer
            )
            return flat_text, raw, usage, _sanitize_messages(messages)

        messages = [
            {"role": "system", "content": STAGE_1_SYSTEM},
            {"role": "user", "content": content},
        ]

        result, raw, usage = llm.complete_structured(
            model=self.transcribe_model,
            messages=messages,
            response_schema=TranscriptionResponse,
            reasoning_effort=self.stage1_reasoning_effort,
        )
        return _transcription_to_tsv(result), raw, usage, _sanitize_messages(messages)

    def _stage2_structure(
        self,
        transcribed_text: str,
        image_path: str,
        intro_text: str,
        intro_image_paths: Optional[List[str]] = None,
    ) -> tuple[List[DictionaryEntry], str, dict, list]:
        """
        Call the structuring LLM (Stage 2) with structured output.
        Returns (entries, raw_json_str, usage_dict, sanitized_messages).
        """
        mime = resolve_mime_type(image_path)
        page_data_url = image_data_url(image_path, mime)

        lang_block = (
            self.dictionary_languages.format_prompt_block()
            if self.dictionary_languages
            else ""
        )
        user_text = stage_2_user(
            transcribed_text=transcribed_text,
            intro_text=intro_text,
            discover_extra_fields=self.discover_extra_fields,
            guides=self.stage2_guides,
            dictionary_languages=lang_block,
        )

        content: list = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": page_data_url}},
        ]
        for intro_img in intro_image_paths or []:
            intro_mime = resolve_mime_type(intro_img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(intro_img, intro_mime)},
                }
            )

        messages = [
            {"role": "system", "content": STAGE_2_SYSTEM},
            {"role": "user", "content": content},
        ]

        result, raw, usage = llm.complete_structured(
            model=self.structure_model,
            messages=messages,
            response_schema=EntriesResponse,
            reasoning_effort=self.stage2_reasoning_effort,
        )
        return result.entries, raw, usage, _sanitize_messages(messages)

    def _ensure_field_map(
        self,
        transcribed_text: str,
        image_path: str,
    ) -> FieldMapPrompt:
        """Pass 1: load or discover field map once per dictionary."""
        if self._field_map is not None:
            return self._field_map

        if not self.stage2_experiment_dir:
            raise ValueError(
                "direct_mdf stage2_mode requires stage2_experiment_dir for field map cache."
            )

        cache_path = self.stage2_experiment_dir / "field_cheatsheet.json"
        if self.field_cheatsheet_gold:
            if self.entry_dir is None:
                raise ValueError(
                    "field_cheatsheet_gold requires entry_dir to locate outputs/stage-2-gold/"
                )
            print(
                "Pass 1: using gold marker cheat sheet "
                f"(outputs/stage-2-gold/field_cheatsheet.json) …"
            )
            self._field_map = load_gold_cheatsheet(self.entry_dir)
            if self.overwrite or not cache_path.is_file():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(self._field_map.model_dump(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        else:
            intro_paths = [Path(p) for p in self.intro_image_paths]
            dictionary_name = self.entry_dir.name if self.entry_dir else ""
            discover_kwargs = dict(
                force_refresh=self.overwrite,
                transcription=transcribed_text,
                sample_image=Path(image_path),
                intro_images=intro_paths,
                model=self.structure_model,
                reasoning_effort=self.stage2_reasoning_effort,
                languages_config=self.dictionary_languages,
                dictionary_name=dictionary_name,
            )
            print(f"Pass 1: marker cheat sheet discovery (cache → {cache_path}) …")
            self._field_map = load_or_discover_cheatsheet(cache_path, **discover_kwargs)

        print(self._field_map.format_prompt_block())
        return self._field_map

    def _stage2_direct_mdf(
        self,
        transcribed_text: str,
        image_path: str,
        intro_text: str,
        intro_image_paths: Optional[List[str]],
        field_map: FieldMapPrompt,
    ) -> tuple[str, str, dict, list]:
        """Pass 2: direct MDF extraction using a field map."""
        del intro_text  # intro images carry layout context; text unused here
        mdf_text, raw, usage, messages = extract_direct_mdf(
            transcription=transcribed_text,
            image_path=image_path,
            intro_image_paths=intro_image_paths or [],
            field_map=field_map,
            model=self.structure_model,
            reasoning_effort=self.stage2_reasoning_effort,
            guides=self.stage2_guides,
            toolbox_pdf=self.stage2_toolbox_pdf,
        )
        return mdf_text, raw, usage, _sanitize_messages(messages)

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
