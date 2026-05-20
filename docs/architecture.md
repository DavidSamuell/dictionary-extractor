# Proposed `src/` Architecture

This document describes the proposed modular refactoring of the `src/` directory into a proper Python package (`dictextractor`), designed to scale to 15+ OCR backends and multiple extraction strategies.

---

## Proposed Directory Tree

```
src/
  dictextractor/
    __init__.py

    schemas/
      __init__.py
      entry.py
      ocr_result.py

    preprocessing/
      __init__.py
      preprocess.py
      steps.py

    ocr/
      __init__.py
      base.py
      paddle_traditional.py
      paddle_vl.py
      mathpix.py
      mathpix_convert.py
      # Future:
      # tesseract.py
      # aws_textract.py
      # deepseek_ocr.py
      # olmocr.py
      # ocrflux.py
      # chandra.py
      # churro.py
      # numarkdown.py
      # dots_ocr.py
      # qwen_vl.py
      # gemini.py
      # claude.py
      # gpt.py
      # grobid.py

    extraction/
      __init__.py
      base.py
      llm_manual.py
      llm_join.py
      llm_two_stage.py

    llm/
      __init__.py
      client.py
      prompts.py

    evaluation/
      __init__.py
      evaluator.py
      error_analyzer.py
      metrics.py

    utils/
      __init__.py
      text.py
      image.py
      io.py
      visualization.py

    cli/
      __init__.py
      extract.py
      evaluate.py
      evaluate_stage1.py
      preprocess.py
      run_ocr.py
      run_mathpix_convert.py
      annotate.py
```

Everything outside `src/` stays unchanged: `assets/`, `outputs/`, `scripts/`, `requirements.txt`, `.env`.

---

## Subdirectory Reference

### `schemas/`

The single source of truth for all shared data models. Every other module imports from here rather than defining its own types.

| File | Contents |
|---|---|
| `entry.py` | `DictionaryEntry` (Pydantic) and `DictionaryPage` — the canonical schema for a structured dictionary entry (headword, POS, translation, literal meaning, grammar notes) |
| `ocr_result.py` | `BBox`, `OCRLine`, `OCRBlock`, `OCRPageResult` — the unified output type that every OCR backend produces |

**Why it exists:** Currently `DictionaryEntry` is defined twice with different structures in `extract_dictionary.py` (Pydantic model) and `paddle_ocr.py` (dataclass). `schemas/` eliminates this conflict and gives all modules a shared vocabulary.

---

### `preprocessing/`

Everything related to preparing a raw scanned image before it is passed to an OCR engine.

| File | Contents |
|---|---|
| `preprocess.py` | `DictionaryPreprocessor` class — chains preprocessing steps in sequence, with optional step-by-step visualization |
| `steps.py` | Individual pure functions: `to_grayscale()`, `deskew()`, `denoise()`, `contrast_normalization()`, `sharpen()` |

**Why it exists:** The preprocessing logic is currently duplicated across three files (`preprocessing.py`, `extract_dictionary.py`, and `paddle_ocr.py`) with slight variations. This consolidates it into one place. `steps.py` makes each step independently importable and testable.

**Usage note:** Preprocessing is **off by default** in `cli/extract.py` (`--preprocess` is a boolean opt-in). When off, snippet/intro PDFs flow straight to the LLM as `application/pdf` inline data. When on, PDFs are rasterized to PNG (cached under `{output}/.rendered_snippets/` and `.rendered_intro/`) before the full cv2 chain runs.

---

### `ocr/`

Pluggable OCR backends. Every backend implements the same abstract interface from `base.py`: given an image path, return an `OCRPageResult`.

| File | Contents |
|---|---|
| `base.py` | Abstract `OCRBackend` class with `run(image_path) -> OCRPageResult`, plus `name`, `requires_api_key`, and `supports_layout_analysis` properties |
| `paddle_traditional.py` | PaddleOCR detection + recognition pipeline (from `paddle_ocr.py`) |
| `paddle_vl.py` | PaddleOCR Vision-Language model (from `paddle_ocr_vl.py`) |
| `mathpix.py` | Mathpix reader — loads pre-existing `.docx` or `.txt` OCR output as an `OCRPageResult` |
| `mathpix_convert.py` | Mathpix Convert PDF API client — submits a PDF, polls for completion, downloads the `.docx`. Used by `cli/run_mathpix_convert.py` to generate the artifacts that `mathpix.py` later reads. |

**Planned backends by category:**

| Category | Backends |
|---|---|
| Traditional OCR | `tesseract.py` |
| Layout Analysis + OCR | *(non-VLM, TBD)* |
| Closed-source OCR | `aws_textract.py`, `mathpix.py` |
| Reasoning OCR VLMs | `numarkdown.py` (NuMarkdown-8B-Thinking), `dots_ocr.py` |
| Historical VLMs | `churro.py` (CHURRO) |
| Open-source OCR VLMs | `deepseek_ocr.py`, `paddle_vl.py`, `olmocr.py`, `ocrflux.py`, `chandra.py` |
| Generalist VLMs (open) | `qwen_vl.py` (Qwen3.5-VL), `gemma.py` (Gemma 3) |
| Generalist VLMs (closed) | `gemini.py` (Gemini 2.5 Pro), `claude.py` (Claude Sonnet), `gpt.py` (GPT-5) |
| End-to-end extractors | `grobid.py` (GROBID-Dictionaries) |

**Why it exists:** To add a new OCR method you drop a single file into `ocr/` that subclasses `OCRBackend`. Nothing else in the pipeline needs to change.

**Note on VLMs used for OCR:** Generalist VLMs like Gemini 2.5 Pro can appear here when used purely for raw text transcription (no structuring). When the same model is used for structured extraction, it belongs in `extraction/` instead — the distinction is in the prompt and the output format, not the model identity.

---

### `extraction/`

Strategies that turn OCR output (plus the source image) into structured `DictionaryPage` entries. All strategies implement the same abstract interface from `base.py`.

| File | Contents |
|---|---|
| `base.py` | Abstract `ExtractionStrategy` class with `extract(ocr_result, image_path) -> DictionaryPage` |
| `llm_manual.py` | Current approach: hand-tuned prompt with explicit knowledge of the Chukchi dictionary structure |
| `llm_join.py` | **Join method:** LLM(image + intro) → structure description → prompt(structure + alphabet + OCR) → LLM(OCR text + image) → structured output |
| `llm_two_stage.py` | **2-stage method:** LLM(alphabet + just extract text) → extracted text → prompt(interpret structure + map to format) → LLM(OCR + intro + image) → structured output |

**Why it exists:** The current `extract_dictionary.py` conflates I/O, image preprocessing, LLM calls, prompt construction, and output serialization into a single 1026-line file. `extraction/` isolates the strategy logic so new pipelines can be added without touching I/O or LLM plumbing.

---

### `llm/`

All LLM-specific concerns isolated from extraction logic.

| File | Contents |
|---|---|
| `client.py` | Unified wrapper around `litellm` — handles API key loading, model routing (Gemini / OpenRouter / Anthropic / OpenAI), and retry/error handling |
| `prompts.py` | Prompt templates for each extraction strategy, structured as Python objects rather than inline strings |

**Why it exists:** Prompt engineering and model selection are separate concerns from extraction strategy logic. Keeping them here lets you iterate on prompts or swap models without touching the extraction pipeline code.

---

### `evaluation/`

Evaluation of extraction results against ground truth, split into focused files.

| File | Contents |
|---|---|
| `metrics.py` | `EvaluationMetrics` dataclass — counts, accuracy, F1, precision, recall |
| `evaluator.py` | `DictionaryEvaluator` — entry-level matching (weighted similarity: 50% headword, 35% translation, 15% POS), generates evaluation reports |
| `error_analyzer.py` | `DetailedErrorAnalyzer` — character-level CER and WER analysis with substitution/deletion/insertion breakdown, uses shared `utils/text.py` for normalization |

**Why it exists:** The current `evaluate_extraction.py` is 1107 lines combining two distinct analysis concerns (entry-level and character-level) plus duplicated normalization logic. Splitting it makes each concern independently usable and testable.

---

### `utils/`

Shared helper functions deduplicated from across the codebase. No business logic — pure utilities.

| File | Contents |
|---|---|
| `text.py` | `normalize_text()` with the canonical homoglyph mapping (Latin B → Cyrillic В, Latin p → Cyrillic р, Latin ə → Cyrillic ә, etc.) |
| `image.py` | `encode_image_base64()`, image loading helpers |
| `io.py` | `read_docx_text()`, `read_text_file()`, `save_to_tsv()`, `json_to_tsv()` |
| `visualization.py` | `annotate_image()` (from `annotate_ocr_blocks.py`) and PaddleOCR stage visualizers (stage 1/2/3 from `paddle_ocr.py`) |

**Why it exists:** `normalize_text()` with the same homoglyph map is currently copy-pasted in two classes within `evaluate_extraction.py`. `preprocess_image()` is implemented three times across different files. `utils/` is the single import point for all shared logic.

---

### `cli/`

Thin command-line entry points. Each file has an `argparse`-based `main()` that wires together the appropriate modules and runs the pipeline. No business logic lives here.

| File | Command | Replaces |
|---|---|---|
| `extract.py` | `python -m dictextractor.cli.extract` | `src/extract_dictionary.py` main |
| `evaluate.py` | `python -m dictextractor.cli.evaluate` | `src/evaluate_extraction.py` main |
| `evaluate_stage1.py` | `python -m dictextractor.cli.evaluate_stage1` | Stage-1 transcription evaluator (`TextEdit`/`ReadOrderEdit`, GCER, WER, typography F1). Batch mode caches per-page metrics in `stage1_eval_cache.json` (see `--languages`, `--overwrite`). |
| `preprocess.py` | `python -m dictextractor.cli.preprocess` | `src/preprocessing.py` main |
| `run_ocr.py` | `python -m dictextractor.cli.run_ocr` | `src/paddle_ocr.py` main |
| `run_mathpix_convert.py` | `python -m dictextractor.cli.run_mathpix_convert` | Batch OCR of snippet PDFs via Mathpix Convert API |
| `annotate.py` | `python -m dictextractor.cli.annotate` | `src/annotate_ocr_blocks.py` main |

**`cli/extract.py` — batch mode:** supports both single-entry runs (`--input-image` + `--output`) and a batch mode (`--samples-dir <parent>`). In batch mode, every `{source}-{target...}` subfolder is processed using its default layout (`snippets/`, `introduction/`, `mathpix/`, `alphabet.txt`) and outputs land under `{entry}/outputs/stage-1/<stage1-experiment>/` and `{entry}/outputs/stage-2/<stage2-experiment>/`. `--languages A B C` filters to specific subfolders.

**Per-stage experiment slots:** both stages are namespaced by experiment name. `--experiment-name` (default `default`) sets the stage-1 slot AND the default stage-2 slot AND the stage-1 source consumed by stage 2. `--stage2-experiment-name` overrides the stage-2 slot only — use it to sweep stage-2 configs against a fixed stage-1 baseline; the stage-2 manifest records the lineage. `--no-alphabet` and `--no-ocr-hint` are stage-1 ablation toggles that suppress batch-mode auto-discovery of `alphabet.txt` / `mathpix/`. Gold lives under `{entry}/outputs/stage-1-gold/<stem>/<stem>_stage1_GOLD.tsv` — experiment-agnostic, so `--overwrite` of any experiment can never clobber it. `scripts/migrate_stage1_layout.sh` is a one-shot mover for pre-experiment trees (handles both stage 1 and stage 2).

**`run_config.json` manifest per stage per experiment.** Each `stage-1/<exp>/` and `stage-2/<exp>/` directory contains one manifest recording everything that could influence its outputs. Stage 1: model, reasoning, embedded alphabet text (or path for image alphabets), OCR-hint dir, embedded stage-1 guides text, per-page snippet + OCR-hint resolution, preprocess flag, git SHA. Stage 2: structure model, reasoning, `discover_extra_fields`, intro paths (always path-only, regardless of format), embedded stage-2 guides text, lineage to the stage-1 source experiment, per-page stage-1 TSV paths, git SHA. Manifests are written on first run and **preserved on resume** so they never drift from the predictions they document; pass `--overwrite` to refresh.

**`cli/run_mathpix_convert.py`:** walks a samples root (default `assets/dictionaries/samples-2`) and, for every entry folder missing a `mathpix/` subfolder, creates one and converts each PDF in `snippets/` to `page_N.docx` via the Mathpix Convert API. Requires `MATHPIX_APP_ID` and `MATHPIX_APP_KEY` (loaded from `.env`).

**Why it exists:** Separating CLI wiring from library code means any module can be imported and used programmatically (in notebooks, tests, or other scripts) without going through `argparse`. The shell scripts in `scripts/` call these CLI entry points.

---

## Data Flow

```
Input snippet (image OR pdf)
    │
    ▼ (opt-in via --preprocess; PDFs rendered to PNG first)
preprocessing/preprocess.py
    │ preprocessed image
    ▼
ocr/[backend].py  (implements OCRBackend)
    │ OCRPageResult
    ▼
extraction/[strategy].py  (implements ExtractionStrategy)
    │   uses llm/client.py + llm/prompts.py
    │ DictionaryPage
    ▼
utils/io.py  →  extracted_dictionary.tsv / .json
    │
    ▼
evaluation/evaluator.py + evaluation/error_analyzer.py
    │ vs. assets/gold_label_dictionary.tsv
    ▼
evaluation_report.txt / character_error_report.txt
```

The `ocr/` backends all produce `OCRPageResult`. The `extraction/` strategies all consume `OCRPageResult` and produce `DictionaryPage`. This means any OCR backend can be paired with any extraction strategy without code changes.

**Input formats:** `cli/extract.py` accepts both raster images (`.png`, `.jpg`, `.jpeg`, `.webp`) and PDFs in the snippets and introduction folders. When `--preprocess` is off (default), PDFs are passed through to the LLM as `application/pdf` inline data (`utils/image.py::resolve_mime_type` maps the extension). When `--preprocess` is on, PDFs are rasterized via PyMuPDF because the cv2 pipeline needs pixel input.

---

## Migration Map

| Current file | Migrated to |
|---|---|
| `src/preprocessing.py` | `preprocessing/preprocess.py` + `preprocessing/steps.py` |
| `src/paddle_ocr.py` | `ocr/paddle_traditional.py` |
| `src/paddle_ocr_vl.py` | `ocr/paddle_vl.py` |
| `src/extract_dictionary.py` | `extraction/llm_manual.py` + `llm/client.py` + `llm/prompts.py` + `schemas/entry.py` + `utils/io.py` + `utils/image.py` + `cli/extract.py` |
| `src/evaluate_extraction.py` | `evaluation/evaluator.py` + `evaluation/error_analyzer.py` + `evaluation/metrics.py` |
| `src/json_to_tsv.py` | `utils/io.py` (absorbed as a utility function) |
| `src/annotate_ocr_blocks.py` | `utils/visualization.py` + `cli/annotate.py` |
