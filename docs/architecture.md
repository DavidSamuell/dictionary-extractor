# Architecture

This document walks through the TransLex pipeline module by module so a new contributor can map any CLI flag to the code that implements it. For the high-level "what the project does" pitch, see [`README.md`](../README.md); for stage-specific methodology, see [`stage_1_methodology.md`](stage_1_methodology.md) and [`stage_2_methodology.md`](stage_2_methodology.md).

---

## 1. High-level data flow

```
                          ┌─────────────────────────────┐
                          │  Samples tree (per dict.)   │
                          │  snippets/                  │
                          │  introduction/              │
                          │  alphabet.txt               │
                          │  dictionary_languages.yaml  │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │  cli/extract.py  (dictextractor-extract)                        │
       │   • parse args, resolve per-page inputs                         │
       │   • dispatch by --strategy:                                     │
       │       two_stage    → extraction/llm_two_stage.py                │
       │       vlm_ocr      → extraction/vlm_ocr.py    + ocr/vlm/        │
       │       mathpix_ocr  → extraction/mathpix_ocr.py + ocr/mathpix*   │
       │       manual/join  → legacy single-shot strategies              │
       └────────────────────┬────────────────────────────────────────────┘
                            │
        ┌───────────────────┴────────────────────┐
        ▼                                        ▼
  STAGE 1 — transcription                  STAGE 2 — direct MDF
  llm_two_stage.run_stage1(...)            llm_two_stage.run_stage2(...)
     └─ llm/prompts.py                        ├─ llm/field_discovery.py  (Pass 1)
     └─ llm/client.py                         │   └─ outputs/.../field_cheatsheet.json
                                              └─ llm/stage2_direct_mdf.py (Pass 2)
                                                  └─ outputs/.../{page}.mdf.txt
        │                                        │
        ▼                                        ▼
  evaluation/stage1/  (eval-flat)         evaluation/stage2/  (eval-stage2-mdf)
```

The strategy abstraction in [`extraction/base.py`](../src/dictextractor/extraction/base.py) guarantees every backend exposes the same `extract(ocr_result, image_path)` surface; the unified [`OCRPageResult`](../src/dictextractor/schemas/ocr_result.py) is the only interchange type between OCR and extraction.

---

## 2. CLI layer (`src/dictextractor/cli/`)

Every console script is a thin argparse wrapper with **no business logic** — it resolves paths, loads `dictionary_languages.yaml`, and forwards to the strategy.

| Script                          | Module                                      | Role                                               |
|---------------------------------|---------------------------------------------|----------------------------------------------------|
| `dictextractor-extract`         | `cli/extract.py`                            | Stage 1 / Stage 2 extraction (batch or single page) |
| `dictextractor-eval-flat`       | `cli/evaluate_stage_flat.py`                | Stage 1 evaluation                                 |
| `dictextractor-eval-stage2-mdf` | `cli/evaluate_stage2_mdf.py`                | Stage 2 MDF evaluation                             |
| `dictextractor-evaluate`        | `cli/evaluate.py`                           | Legacy TSV evaluation (schema mode)                |
| `dictextractor-mathpix-convert` | `cli/run_mathpix_convert.py`                | Mathpix Convert batch driver                       |
| `dictextractor-run-ocr`         | `cli/run_ocr.py`                            | Stand-alone OCR backend runner                     |
| `dictextractor-preprocess`      | `cli/preprocess.py`                         | cv2 preprocessing                                  |
| `dictextractor-annotate`        | `cli/annotate.py`                           | Visualise OCR geometry                             |

Console-script registrations live in `pyproject.toml`.

---

## 3. Extraction strategies (`src/dictextractor/extraction/`)

```
extraction/
├── base.py              ExtractionStrategy ABC
├── llm_two_stage.py     Default — Stage 1 + Stage 2 LLM pipeline
├── llm_manual.py        Legacy single-shot Chukchi-specific prompt
├── llm_join.py          Legacy image-first + OCR-text join
├── vlm_ocr.py           Specialised VLM (MinerU / Paddle / GLM)
├── mathpix_ocr.py       Commercial Mathpix OCR
└── sample_entry.py      Per-language entry helpers (snippets, intro, alphabet)
```

### `llm_two_stage.TwoStageLLMExtraction`

The default pipeline that produces the numbers in the paper.

- **Stage 1** (`run_stage1` and `_run_stage1_flat`): builds prompts via [`llm/prompts.py`](../src/dictextractor/llm/prompts.py), calls the LLM through [`llm/client.py`](../src/dictextractor/llm/client.py) using `litellm.complete_structured`, deserialises into `TranscriptionResponse` (column) or `FlatTranscriptionResponse` (flat), and serialises to `*_stage1.tsv` / `*_stage1_flat.txt`.
- **Stage 2** branches on `stage2_mode`:
  - `direct_mdf` (default) → Pass 1 [`llm/field_discovery.load_or_discover_cheatsheet`](../src/dictextractor/llm/field_discovery.py), Pass 2 [`llm/stage2_direct_mdf.extract_direct_mdf`](../src/dictextractor/llm/stage2_direct_mdf.py).
  - `schema` (legacy) → single structured call returning `EntriesResponse`, serialised to JSON + TSV via [`utils/io.py`](../src/dictextractor/utils/io.py).

### `vlm_ocr.run_vlm_ocr_entry`

Drives the specialised document VLMs out of isolated virtual envs (one per model: `.venv-mineru-vllm`, `.venv-paddleocr`, `.venv-glmocr`). Spec resolution and runtime construction happen in [`ocr/vlm/registry.py`](../src/dictextractor/ocr/vlm/registry.py) and [`ocr/vlm/runner.py`](../src/dictextractor/ocr/vlm/runner.py). Raw backend outputs are then funnelled through the **flat adapter** (see §5) so OCR results compete on the same flat contract as LLM transcriptions.

### `mathpix_ocr`

Two-step: `dictextractor-mathpix-convert` calls the Mathpix Convert API and caches `.md` + `.lines.json` files; then `--strategy mathpix_ocr` runs the adapter that converts those caches to `*_stage1_flat.txt`.

---

## 4. LLM layer (`src/dictextractor/llm/`)

```
llm/
├── client.py                litellm wrapper (model routing, structured output, retries)
├── prompts.py               Stage 1 + Stage 2 (legacy schema) prompts
├── field_discovery.py       Pass 1 — discover MDF markers per dictionary
├── stage2_direct_mdf.py     Pass 2 — transcript → Toolbox MDF text
└── mdf_marker_reference.py  Curated MDF marker inventory used in Pass 1 prompt
```

- `client.complete(...)` and `client.complete_structured(...)` are the two call sites used by extraction. Provider keys are resolved by substring match on the model string. Gemini 3 quirks (fixed temperature, thinking-config) and OpenRouter token budgets are centralised here.
- Pass 1 (`field_discovery.py`) caches the cheat sheet under `outputs/stage-2/<experiment>/field_cheatsheet.json` so Pass 2 runs on every page without re-discovering markers. Use `--overwrite` to refresh.
- Pass 2 (`stage2_direct_mdf.py`) is **free-form** — the LLM emits raw MDF text rather than JSON. Allowed structural changes (strip `<b>` / `<i>`, rejoin hyphens, normalise homograph / sense numbers, strip end-of-sentence punctuation) are enforced via the system prompt.

---

## 5. OCR backends (`src/dictextractor/ocr/`)

```
ocr/
├── base.py                  OCRBackend ABC
├── mathpix.py               Mathpix backend (image / PDF → text + lines)
├── mathpix_convert.py       Convert-API wrapper used by the convert CLI
├── paddle_traditional.py    Classic PaddleOCR (legacy)
├── paddle_vl.py             PaddleOCR-VL spec / orchestration
├── vlm/                     Pluggable VLM-OCR runtime
│   ├── registry.py          Backend keys (mineru2.5-pro, paddleocr-vl-1.5, glm-ocr)
│   ├── runner.py            Common runtime (image batching, JSON sidecars)
│   ├── mineru.py            MinerU 2.5 Pro driver
│   ├── paddle_vl.py         PaddleOCR-VL driver
│   ├── glm_ocr.py           GLM-OCR driver
│   ├── glm_vllm_server.py   Optional vLLM server harness for GLM-OCR
│   ├── paddle_genai_server.py vLLM server hookup for PaddleOCR-VL
│   ├── prompts.py           VLM prompt builders (alphabet list, OCR hint)
│   ├── completion.py        vLLM/OpenAI-compatible chat completion helper
│   └── page_inputs.py       Page / snippet enumeration helpers
└── adapters/                OCR layout → flat .txt (frozen v1)
    ├── blocks.py            Normalised {bbox, text, category} blocks
    ├── layout_to_transcript_v1.py  Column-major flatten (ADAPTER_VERSION = "v1")
    ├── mineru.py, paddle.py, glm.py  Backend-specific block parsers
    ├── markdown_to_flat.py  Markdown OCR → flat text
    ├── mathpix_flat.py      Mathpix .md → flat
    ├── mathpix_lines.py     Mathpix .lines.json → flat
    └── flat_export.py       Single entry point used by Stage 1 evaluators
```

**Why an adapter layer?** Each specialised VLM emits its own layout JSON. To compare them apples-to-apples against LLM transcriptions, every backend funnels through `layout_to_transcript_v1`, which classifies header / body / footer blocks, clusters x-centres into at most three columns, and emits a column-major line list. The adapter is frozen at `v1` so retroactive changes don't silently move scores.

---

## 6. Stage 1 evaluation (`src/dictextractor/evaluation/stage1/`)

```
stage1/
├── flat_evaluator.py        Top-level eval driver
├── flatten.py               Gold flattening (FLAT_SPEC_VERSION = "v2")
├── alignment.py             quick_match + collapsed alignment modes
├── quick_match.py           OmniDocBench quick-match aligner
├── character_quality.py     TextEdit / GCER / WER
├── markup_quality.py        Bold / italic P/R/F1
├── read_order.py            ReadOrderEdit over gold line indices
├── normalize_typography.py  TYPOGRAPHY_SPEC_VERSION = "v1"
├── tag_parser.py            <b>/<i> tag tokenisation
├── stage1_eval_cache.py     Persistent cache (CACHE_FORMAT_VERSION = 12)
└── stage1_reports.py        Per-experiment CSV / JSON / TXT writers
```

Outputs: `evaluations/stage1_flat_eval/{stage1_flat_eval_summary.csv, stage1_flat_eval_detailed.csv, <experiment>/...}`. See [`stage_1_evaluation_metrics.md`](stage_1_evaluation_metrics.md) for metric definitions.

---

## 7. Stage 2 evaluation (`src/dictextractor/evaluation/stage2/`)

```
stage2/
├── mdf_evaluator.py         Top-level eval driver (Record Accuracy, MDF F1, ROE)
├── mdf_parser.py            Blank-line-delimited Toolbox MDF parser
├── mdf_align.py             Greedy record + line alignment
├── mdf_similarity.py        SequenceMatcher + value normalisation
├── mdf_marker_equiv.py      Marker substitution groups (e.g. \ge ↔ \de)
├── mdf_metrics.py           TP/FP/FN accounting + read-order edit
├── evaluator.py             Legacy TSV evaluator (schema mode)
├── metrics.py               Legacy metrics aggregator
└── error_analyzer.py        Character-level error breakdown (legacy)
```

Outputs: `evaluations/stage2_mdf_eval/{stage2_mdf_eval_summary.csv, <experiment>/...}`. See [`stage_2_evaluation_metrics.md`](stage_2_evaluation_metrics.md) for metric definitions.

---

## 8. Schemas (`src/dictextractor/schemas/`)

| File                       | Purpose                                                                              |
|----------------------------|--------------------------------------------------------------------------------------|
| `ocr_result.py`            | `OCRPageResult`, `OCRBlock`, `OCRLine`, `BBox` — unified OCR data carrier            |
| `entry.py`                 | `DictionaryEntry`, `DictionaryPage`, `TranscriptionResponse`, `FlatTranscriptionResponse`, `EntriesResponse` |
| `entry_numbers.py`         | Sense / homograph number normalisation (Roman → integer)                             |
| `field_cheatsheet.py`      | `DictionaryMarkerCheatsheet` — Pass 1 output schema                                  |
| `field_map.py`             | `FieldMapPrompt` — rendered into Pass 2 user prompt                                  |
| `dictionary_languages.py`  | `DictionaryLanguagesConfig` — per-dictionary YAML config                             |

---

## 9. Utilities (`src/dictextractor/utils/`)

| File                          | Purpose                                                                          |
|-------------------------------|----------------------------------------------------------------------------------|
| `io.py`                       | TSV / JSON I/O (`save_to_json`, `json_to_tsv`, `save_stage2_outputs`)            |
| `image.py`                    | `image_data_url`, MIME detection                                                 |
| `pdf_render.py`               | PyMuPDF page rasteriser (used when models don't accept PDF inline)               |
| `stage1_input.py`             | Resolve column / flat / gold Stage-1 transcripts for Stage 2                     |
| `stage2_direct_mdf_io.py`     | Write `{page}.mdf.txt` + sidecars                                                |
| `stage2_page_selection.py`    | `--one-page-per-entry` page picker                                               |
| `mdf_export.py`               | MDF normalisation helpers (used by Pass 2 and eval)                              |
| `mdf_compare.py`              | Dev-time MDF diff against gold                                                   |
| `dictionary_languages.py`     | Load / validate / autogenerate `dictionary_languages.yaml`                       |

---

## 10. Preprocessing (`src/dictextractor/preprocessing/`)

Optional cv2 pipeline (grayscale → deskew → denoise → contrast → sharpen). Off by default — without it, PDFs flow straight to the LLM as `application/pdf` inline data. Triggered by `--preprocess`.

---

## 11. Sample tree contract

```
assets/dictionaries/samples/<Source-Target>/
├── snippets/                          page images / PDFs (3 per dictionary in the paper)
├── introduction/                      intro pages (text, image, PDF)
├── alphabet.txt                       source-language alphabet list (optional)
├── alphabet.png                       alphabet rendered as image (optional)
├── mathpix/                           cached Mathpix .md / .lines.json (optional)
├── dictionary_languages.yaml          source/target roles + layout type
└── outputs/                           populated by the pipeline (gitignored locally)
    ├── stage-1/<experiment>/<page>/<page>_stage1_flat.txt
    ├── stage-1/<experiment>/run_config.json
    ├── stage-1-gold/<page>/<page>_stage1_GOLD.tsv          # column gold (hand-annotated)
    ├── stage-1-gold/<page>/<page>_stage1_GOLD_flat.txt     # derived via scripts/flatten_stage1_gold.py
    ├── stage-2/<experiment>/field_cheatsheet.json
    ├── stage-2/<experiment>/<page>/<page>.mdf.txt
    ├── stage-2/<experiment>/run_config.json
    └── stage-2-gold/<page>/<page>.mdf.txt                  # MDF gold (hand-annotated)
```

`<experiment>` is opaque; the convention used in the paper is `<model>_<reasoning>_<flat|mdf>_<alphabet-state>_<ocr-state>` for Stage 1 (e.g. `gemini31pro_flat_alpha`) and `<model>_high_mdf_<intro|nointro>_<toolbox|notoolbox>` for Stage 2 (e.g. `gemini31pro_high_mdf_intro_toolbox`).

---

## 12. Extending the pipeline

| To add …                  | Do this                                                                                                 |
|---------------------------|---------------------------------------------------------------------------------------------------------|
| A new LLM provider        | Add the routing rule to `llm/client.py`; nothing else changes.                                          |
| A new general-purpose LLM | Use existing routing — pass `--model <provider/model-id>`. Set provider key in `.env`.                  |
| A new specialised VLM     | (1) Register a runner under `ocr/vlm/<name>.py`; (2) add a block parser under `ocr/adapters/<name>.py`; (3) wire into `ocr/vlm/registry.py`. |
| A new extraction strategy | Subclass `ExtractionStrategy` in `extraction/<name>.py`; register in `_STRATEGIES` in `cli/extract.py`. |
| A new Stage 1 metric      | Add a `*_quality.py` module under `evaluation/stage1/` and surface it in `flat_evaluator.py` + `stage1_reports.py`. |
| A new MDF marker family   | Update `llm/mdf_marker_reference.py` (Pass 1 vocabulary) and `assets/evaluation/mdf_marker_sub_list.yaml` (eval-time equivalence). |

---

## 13. Versioning anchors

| Component                   | Version key                | Module                                          |
|-----------------------------|----------------------------|-------------------------------------------------|
| Flat line-order spec        | `FLAT_SPEC_VERSION = "v2"` | `evaluation/stage1/flatten.py`                  |
| OCR layout adapter          | `ADAPTER_VERSION = "v1"`   | `ocr/adapters/layout_to_transcript_v1.py`       |
| Typography normalisation    | `TYPOGRAPHY_SPEC_VERSION = "v1"` | `evaluation/stage1/normalize_typography.py` |
| Stage 1 eval cache          | `CACHE_FORMAT_VERSION = 12` | `evaluation/stage1/stage1_eval_cache.py`        |

Bumping any of these invalidates prior results — re-run the corresponding evaluation.
