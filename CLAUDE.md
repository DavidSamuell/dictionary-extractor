# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Experimental pipeline for extracting structured entries from scanned multilingual dictionary pages (originally Chukchi-Russian; now generalised to any source/target language pair). The package is `dictextractor`. `README.md` is the user-facing overview (keep it in sync with behaviour changes); `docs/architecture.md` is the per-module deep dive — read that before making non-trivial structural changes.

## Tooling: uv, not pip

This project uses `uv` for everything. **Never** invoke `pip`, `python`, or `python3` directly to run code in this repo — always go through `uv run`. The `.venv/` is managed by uv, and the registered console scripts (`dictextractor-extract`, `dictextractor-evaluate`, `dictextractor-eval-s1`, `dictextractor-mathpix-convert`, etc.) only resolve when invoked under `uv run`.

```bash
uv sync                          # install all deps into .venv
uv sync --extra paddle           # also install paddleocr/paddlepaddle
uv add <package>                 # add a dep (updates pyproject.toml + uv.lock)
uv run dictextractor-extract --help
uv run python -m dictextractor.cli.evaluate_stage1 --help
```

## Common commands

End-to-end batch over every language subfolder under a samples root (the primary use case):

```bash
# Stage 1 only (transcription) — outputs *_stage1.tsv per page
uv run dictextractor-extract \
    --strategy two_stage --stage 1 \
    --model gemini/gemini-3-flash-preview \
    --samples-dir assets/dictionaries/samples-2 \
    --overwrite

# Stage 2 only (structuring) — requires existing stage-1 TSVs
uv run dictextractor-extract \
    --strategy two_stage --stage 2 \
    --model gemini/gemini-3.1-pro-preview \
    --samples-dir assets/dictionaries/samples-2 \
    --languages Chepang-English \
    --discover-extra-fields --overwrite

# Both stages, single entry mode
uv run dictextractor-extract \
    --strategy two_stage \
    --input-image assets/pages/ --output outputs/run1/ \
    --intro assets/introduction/ --alphabet assets/alphabet.txt \
    --ocr-text assets/mathpix/

# Batch Mathpix OCR over every entry folder missing mathpix/
uv run dictextractor-mathpix-convert --samples-dir assets/dictionaries/samples-2

# Stage-1 evaluation (CER/WER/GCER/BLEU + markup F1 + read-order NED)
uv run python -m dictextractor.cli.evaluate_stage1 \
    --samples-dir assets/dictionaries/samples/ -o assets/dictionaries/samples/stage1_eval

# Stage-2 evaluation (entry-level matching vs. gold TSV)
uv run dictextractor-evaluate -e <pred>.tsv -g <gold>.tsv -o results/
```

Working examples for each command live in `examples/run_*.sh`. There is **no test suite** in this repo.

## Architecture in one screen

The pipeline is `image → OCR → extraction → evaluation`, with each stage pluggable behind an abstract base class. Read `docs/architecture.md` for full per-module rationale.

- **`schemas/`** — single source of truth for `DictionaryEntry`/`DictionaryPage` and `OCRPageResult`. All other modules import from here.
- **`preprocessing/`** — cv2 chain (`grayscale → deskew → denoise → contrast → sharpen`). **Off by default** in `cli/extract.py` (`--preprocess` opts in). When off, PDFs flow straight to the LLM as `application/pdf`; when on, PDFs are rasterised to PNG via PyMuPDF first (cached under `<output>/.rendered_snippets/` and `.rendered_intro/`).
- **`ocr/`** — OCR backends, all subclassing `OCRBackend.run(image_path) -> OCRPageResult`. Currently: `mathpix` (reads pre-existing `.docx`/`.txt`), `mathpix_convert` (Mathpix Convert PDF API client), `paddle_traditional`, `paddle_vl`. Many more backends planned — adding one is "drop a file in `ocr/`".
- **`extraction/`** — strategies that turn `OCRPageResult` + image into `DictionaryPage`. Three exist: `manual` (hand-tuned single-shot prompt), `join`, and **`two_stage`** (default). All implement `ExtractionStrategy.extract(...)`.
- **`llm/`** — `client.py` wraps `litellm` with provider-aware API key resolution and model-family-specific reasoning quirks (see below). `prompts.py` holds prompt templates as Python strings/builders.
- **`evaluation/stage1/`** — transcription quality (GCER/CER/WER/BLEU/NED, markup F1 for `<b>`/`<i>` tags, read-order NED). Metric definitions and rationale live in `docs/stage1_evaluation_metrics.md`.
- **`evaluation/stage2/`** — entry-level matching (weighted similarity: 50% headword, 35% translation, 15% POS) plus character-level error analysis.
- **`cli/`** — thin argparse wrappers; no business logic. Each maps to a `dictextractor-*` console script via `pyproject.toml`.
- **`label-studio/setup.py`** — provisions Label Studio projects for human post-editing of stage-1 transcriptions; one project per language pair.

## The two-stage extraction pipeline (the default, important)

`extraction/llm_two_stage.py` is the most-developed strategy and where most current work happens. It splits the LLM job in two:

- **Stage 1 (transcription)** — `reasoning_effort="low"`. Inputs: page image + alphabet + optional OCR hint. Output schema: `TranscriptionResponse` (columns of lines, with `<b>`/`<i>` tags). Goal: faithful copy, *no* interpretation.
- **Stage 2 (structuring)** — `reasoning_effort="medium"` by default. Inputs: stage-1 transcript + dictionary intro (text + images) + the page image. Output schema: `EntriesResponse(entries: List[DictionaryEntry])`. Goal: identify entry boundaries and map to typed fields.

**Subtle Stage-2 gotcha:** `reasoning_effort="high"` has been observed to leak chain-of-thought into JSON string fields (notably `semantic_domain`) on dense pages under structured output. Default is `medium`; only bump explicitly via `--reasoning high` when needed, and consider `--reasoning low` for problematic inputs.

**Resume behaviour:** `cli/extract.py` skips pages whose output already exists. Use `--overwrite` to re-process.

**Batch layout convention:** when `--samples-dir <root>` is used, each subfolder is treated as one dictionary entry with the layout:
```
<root>/<source-target>/
    snippets/         # page images or PDFs
    introduction/     # intro pages (images, PDFs, or .txt/.md/.docx)
    mathpix/          # optional OCR hints, file stem must match snippet stem
    alphabet.txt      # optional alphabet/legend for stage 1
    outputs/
        stage-1/<stem>/<stem>_stage1.tsv  + raw/input JSONs
        stage-2/<stem>/<stem>.tsv + .json + *_usage.json
```

## LLM client model-family rules (`llm/client.py`)

The client picks the API key based on substring of the model string (`gemini` / `claude` / `gpt` / `openrouter`) from env vars `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPEN_ROUTER_API_KEY` (loaded via `python-dotenv` from `.env`). Mathpix uses `MATHPIX_APP_ID` / `MATHPIX_APP_KEY`.

Model-specific quirks already encoded:
- **Gemini 3+** — temperature is locked to 1.0 by litellm (do not pass it). `reasoning_effort` maps to `thinking_level`; `none` collapses to `low` (Gemini 3 cannot fully disable thinking).
- **Gemini 2.5** — to disable thinking, send `extra_body={"generationConfig": {"thinking": {"thinkingConfig": {"mode": "DISABLED"}}}}` (handled when `reasoning_effort` is `none`/`low`/unset). Cannot be disabled on `gemini-2.5-pro`.

When adding a new model family, extend `_resolve_api_key()` and `_build_params()` rather than passing provider-specific kwargs from call sites.

## Known landmines

- **Keep `README.md` current** — when changing CLI flags, schema fields, the samples-dir layout, default models, or evaluation outputs, update the README in the same change. The user explicitly asked for this on 2026-05-07.
- **Don't run python directly** — `python -m dictextractor.cli.extract` will fail without `PYTHONPATH=src`. Always use `uv run` (which uses the installed package) or `uv run python -m ...`.
- **PDFs vs. images** — when adding new pipeline stages, remember that snippet inputs may be either; only enable PDF rasterisation when the downstream consumer (cv2) requires pixel input.
- **`assets/` is gitignored** — sample dictionaries, gold labels, model outputs are all local-only. Don't expect them to be present on a fresh checkout.
