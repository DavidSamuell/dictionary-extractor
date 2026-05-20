# Dictionary Extractor

Experimental pipeline for extracting structured entries from scanned multilingual dictionary pages using OCR and LLMs. Originally built for Chukchi-Russian dictionaries; now generalised to arbitrary source/target language pairs.

The pipeline is deliberately modular: any OCR backend can be paired with any extraction strategy. See [`docs/architecture.md`](docs/architecture.md) for the full module-by-module breakdown.

## Quick start

```bash
# Install (uv-managed; do not use pip directly)
uv sync                 # creates .venv + installs all deps
uv sync --extra paddle  # also install PaddleOCR / paddlepaddle

# Configure API keys (any subset, depending on which backends you use)
cat > .env <<'EOF'
GEMINI_API_KEY=...
OPEN_ROUTER_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
MATHPIX_APP_ID=...
MATHPIX_APP_KEY=...
EOF

# Run the default two-stage extraction over a folder of language samples
uv run dictextractor-extract \
    --strategy two_stage \
    --model gemini/gemini-3-flash-preview \
    --samples-dir assets/dictionaries/samples-2
```

Working examples for every CLI live in [`examples/`](examples/) (one shell script per workflow).

## Pipeline at a glance

```
Input snippet (image OR pdf)
    │  (cv2 preprocessing optional via --preprocess; PDFs rendered to PNG when on)
    ▼
ocr/[backend].py      ──►  OCRPageResult
    │
    ▼
extraction/[strategy].py  ──►  DictionaryPage  (uses llm/client.py + llm/prompts.py)
    │
    ▼
utils/io.py  →  <stem>.json + <stem>.tsv
    │
    ▼
evaluation/stage1/  (TextEdit/GCER/WER + markup F1 + ReadOrderEdit)
evaluation/stage2/  (entry-level matching vs. gold TSV)
```

OCR backends all produce `OCRPageResult`; extraction strategies all consume `OCRPageResult` and produce `DictionaryPage`. Adding a new OCR backend or extraction strategy is a single new file.

## Extraction strategies

The default and most-developed strategy is **`two_stage`**, which splits the LLM job into:

1. **Stage 1 — Transcription** (low reasoning). Faithfully copies every visible character into a structured `TranscriptionResponse` (columns of lines, with `<b>`/`<i>` tags for bold/italic). No interpretation.
2. **Stage 2 — Structuring** (medium reasoning by default). Takes the Stage 1 transcript + dictionary intro + page image and produces typed `DictionaryEntry` records.

Other strategies available via `--strategy`:
- `manual` — single-shot prompt with explicit Chukchi-dictionary structure baked in (legacy).
- `join` — image-first structure description joined with OCR text.

## Common workflows

### Batch extraction over a samples root

```bash
# Stage 1 only (transcription) — outputs *_stage1.tsv per page
uv run dictextractor-extract \
    --strategy two_stage --stage 1 \
    --samples-dir assets/dictionaries/samples-2 \
    --model gemini/gemini-3-flash-preview \
    --experiment-name baseline --overwrite

# Stage 2 only — requires existing stage-1 TSVs (same --experiment-name)
uv run dictextractor-extract \
    --strategy two_stage --stage 2 \
    --samples-dir assets/dictionaries/samples-2 \
    --languages Chepang-English \
    --model gemini/gemini-3.1-pro-preview \
    --experiment-name baseline \
    --discover-extra-fields --overwrite
```

When `--samples-dir <root>` is used, every subfolder under `<root>` is treated as one dictionary entry with the layout:

```
<root>/<source-target>/
    snippets/         # page images (.png/.jpg/.jpeg/.webp) or PDFs
    introduction/     # intro pages (images, PDFs, or .txt/.md/.docx)
    mathpix/          # optional pre-OCR'd hint files (stem must match snippet stem)
    alphabet.txt      # optional alphabet/legend (passed to Stage 1)
    outputs/
        stage-1/<stage1-experiment>/<stem>/<stem>_stage1.tsv  + raw/input JSONs
        stage-1/<stage1-experiment>/run_config.json           # stage-1 manifest
        stage-1-gold/<stem>/<stem>_stage1_GOLD.tsv            # gold (experiment-agnostic)
        stage-2/<stage2-experiment>/<stem>/<stem>.tsv + .json + *_usage.json
        stage-2/<stage2-experiment>/run_config.json           # stage-2 manifest
```

Already-processed pages are skipped automatically (per-experiment); pass `--overwrite` to force re-processing.

### Ablation experiments + reproducibility

Both stages are slotted by an experiment name so you can sweep configurations without overwriting prior runs:

- `--experiment-name` — Stage-1 slot. Also the default Stage-2 slot AND the stage-1 source that Stage 2 consumes.
- `--stage2-experiment-name` — optional override for the Stage-2 slot. Use this when you want to sweep Stage-2 configurations (intro, structure model, reasoning, `--discover-extra-fields`, stage-2 guides) against a fixed Stage-1 baseline; the stage-2 manifest records `--experiment-name` as its `stage1_source`.
- `--no-alphabet` / `--no-ocr-hint` — suppress those inputs for Stage 1 even when `alphabet.txt` / `mathpix/` exist in the language root.

Each experiment slot contains a single `run_config.json` capturing **every configurable parameter** used to produce its TSVs: model, reasoning effort, the verbatim alphabet text (or path if image), per-page snippet + OCR-hint resolution (Stage 1), intro paths + structure model + reasoning + lineage to the Stage-1 source (Stage 2), and the embedded contents of `--stage-1-guides` / `--stage-2-guides` when set. The manifest is written on first run and preserved on resume; pass `--overwrite` to refresh it.

```bash
# Baseline: alphabet + OCR hint + flash
uv run dictextractor-extract \
    --strategy two_stage --stage 1 \
    --samples-dir assets/dictionaries/samples-2 \
    --model gemini/gemini-3-flash-preview \
    --experiment-name gemini3flash_alpha_ocr

# Ablation: same model, no alphabet, no OCR hint
uv run dictextractor-extract \
    --strategy two_stage --stage 1 \
    --samples-dir assets/dictionaries/samples-2 \
    --model gemini/gemini-3-flash-preview \
    --no-alphabet --no-ocr-hint \
    --experiment-name gemini3flash_bare

# Stage-2 sweep against a fixed Stage-1 baseline
uv run dictextractor-extract \
    --strategy two_stage --stage 2 \
    --samples-dir assets/dictionaries/samples-2 \
    --model gemini/gemini-3.1-pro-preview \
    --experiment-name gemini3flash_alpha_ocr \
    --stage2-experiment-name pro_highreasoning \
    --stage2-reasoning high
```

```bash
# Baseline: alphabet + OCR hint + flash
uv run dictextractor-extract \
    --strategy two_stage --stage 1 \
    --samples-dir assets/dictionaries/samples-2 \
    --model gemini/gemini-3-flash-preview \
    --experiment-name gemini3flash_alpha_ocr

# Ablation: same model, no alphabet, no OCR hint
uv run dictextractor-extract \
    --strategy two_stage --stage 1 \
    --samples-dir assets/dictionaries/samples-2 \
    --model gemini/gemini-3-flash-preview \
    --no-alphabet --no-ocr-hint \
    --experiment-name gemini3flash_bare
```

If you have predictions and gold from before this layout existed, run [`scripts/migrate_stage1_layout.sh`](scripts/migrate_stage1_layout.sh) once to move gold to `stage-1-gold/` and pre-existing predictions to the `legacy` experiment slot. `assets/` is gitignored so the move is local-only.

### Single-entry extraction

```bash
uv run dictextractor-extract \
    --strategy two_stage \
    --input-image assets/pages/ \
    --output outputs/run1/ \
    --intro assets/introduction/ \
    --alphabet assets/alphabet.txt \
    --ocr-text assets/mathpix/
```

### OCR with Mathpix (batch)

```bash
uv run dictextractor-mathpix-convert --samples-dir assets/dictionaries/samples-2
```

Walks every entry subfolder and, for each one missing a `mathpix/` directory, runs every PDF in `snippets/` through the Mathpix Convert API, writing `page_N.docx` files. Requires `MATHPIX_APP_ID` and `MATHPIX_APP_KEY` in `.env`.

### Evaluation

**Stage 1** (transcription quality — character accuracy, markup preservation, read-order):

```bash
# Compare every experiment under each language root
uv run dictextractor-eval-s1 \
    --samples-dir assets/dictionaries/samples-2 --all-experiments \
    -o assets/dictionaries/samples-2/stage1_eval

# Restrict to specific experiments
uv run dictextractor-eval-s1 \
    --samples-dir assets/dictionaries/samples-2 \
    --experiment-name gemini3flash_alpha_ocr \
    --experiment-name gemini3flash_bare \
    -o assets/dictionaries/samples-2/stage1_eval
```

Reports cover OmniDocBench-style `TextEdit` and `ReadOrderEdit`, plus GCER,
WER, and per-tag bold/italic typography F1 after semantic adjacency alignment.
Outputs:

- `<out>/stage1_eval_detailed.csv` — long-format, one row per `(experiment, page_id)` plus a per-experiment `__aggregate__` row. Includes `alphabet` and `ocr-hint` booleans from each language's `run_config.json`.
- `<out>/stage1_eval_summary.csv` — same metrics aggregated per `(experiment, language)`, with `page_count`, `alphabet`, and `ocr-hint`.
- `--metrics minimal` — CSVs only include `TextEdit`, `GCER`, `WER`, `typography_f1` (bold+italic TP/FP/FN pooled), and `ReadOrderEdit`. Default is `full` (adds full bold/italic detail).
- `--alignment-threshold` / `--alignment-max-span-rows` — tune semantic adjacency matching (defaults: `0.5` and `3`).
- **Incremental batch runs:** metrics are stored under `<out>/stage1_eval_cache.json` (keyed by experiment, page, prediction/gold fingerprints, and alignment settings). The detailed and summary CSVs always cover **every** language that has gold+predictions under `--samples-dir`, but only pages in the current `--languages` selection (or **all** languages if omitted) plus any page whose cache is stale get recomputed. Use `--overwrite` to force recomputation for the current `--languages` / `--experiment-name` slice without dropping other cached pages. Delete the cache file to recompute everything.
- `<out>/<experiment>/` — the three per-component CSVs, the JSON report, and the human-readable text report (same shape as the pre-experiment evaluator), one drill-down folder per experiment.

Metric definitions and aggregation rules are in [`docs/stage1_evaluation_metrics.md`](docs/stage1_evaluation_metrics.md).

**Stage 2** (entry-level matching vs. gold TSV):

```bash
uv run dictextractor-evaluate -e <pred>.tsv -g <gold>.tsv -o results/
```

Computes weighted similarity (50% headword, 35% translation, 15% POS), precision/recall/F1, plus character-level error analysis.

### Label Studio for human post-editing

```bash
bash examples/run_label_studio_local.sh
```

Provisions one Label Studio project per language pair, uploads page images, and pre-fills tasks with Stage 1 transcriptions for human correction. See [`label-studio/setup.py`](label-studio/setup.py).

## Output schemas

`DictionaryEntry` (canonical, defined in [`src/dictextractor/schemas/entry.py`](src/dictextractor/schemas/entry.py)):

| Field                | Description                                                                  |
|----------------------|------------------------------------------------------------------------------|
| `headword`           | Headword/phrase with all diacritics preserved                                |
| `pos`                | Part-of-speech tag/abbreviation (empty if absent)                            |
| `meaning_description`| Primary definition; senses joined with `;`, sub-meanings with ` \| `         |
| `semantic_domain`    | Short domain label (`botany`, `colloquial`, …) — only when explicitly marked |
| `examples`           | List of usage examples                                                       |
| `extra_fields`       | Discovery slot for non-canonical fields (etymology, IPA, gender, …)          |

Use `--discover-extra-fields` on Stage 2 to populate `extra_fields`; otherwise it stays empty.

### Per-run prompt overrides

Pass `--stage-1-guides <path>` and/or `--stage-2-guides <path>` to append the contents of a `.txt`/`.md`/`.docx` file verbatim to the corresponding user prompt, under a `USER DEFINED GUIDELINES` header. Useful for one-off or per-language tweaks (e.g. "ignore page numbers", "treat `;` as a sense separator") without touching `src/dictextractor/llm/prompts.py`.

```bash
uv run dictextractor-extract \
    --strategy two_stage --stage 2 \
    --samples-dir assets/dictionaries/samples-2 \
    --languages Chepang-English \
    --stage-2-guides assets/dictionaries/samples-2/Chepang-English/guides_s2.md \
    --overwrite
```

Flags are optional — leaving them unset is identical to today's behaviour. The header section is omitted entirely when the file is not provided.

## CLI reference

All entry points are registered as console scripts (run with `uv run <name>`):

| Console script                  | Module                                       | Purpose                                   |
|---------------------------------|----------------------------------------------|-------------------------------------------|
| `dictextractor-extract`         | `dictextractor.cli.extract`                  | Run extraction (single entry or batch)    |
| `dictextractor-evaluate`        | `dictextractor.cli.evaluate`                 | Stage-2 entry-level evaluation            |
| `dictextractor-eval-s1`         | `dictextractor.cli.evaluate_stage1`          | Stage-1 transcription evaluation          |
| `dictextractor-mathpix-convert` | `dictextractor.cli.run_mathpix_convert`      | Batch OCR of PDFs via Mathpix Convert API |
| `dictextractor-preprocess`      | `dictextractor.cli.preprocess`               | Standalone cv2 image preprocessing        |
| `dictextractor-run-ocr`         | `dictextractor.cli.run_ocr`                  | Run a chosen OCR backend                  |
| `dictextractor-annotate`        | `dictextractor.cli.annotate`                 | Visualise OCR blocks on the page image    |

Pass `--help` to any of them for full options.

## Tooling notes

- This project uses [`uv`](https://docs.astral.sh/uv/). Never invoke `pip` or run `python` directly — always go through `uv run` (the registered console scripts only resolve when launched by uv). Quick reference: [`docs/uv.md`](docs/uv.md).
- LLM calls are routed through `litellm`, with provider keys resolved by substring of the model string (`gemini` → `GEMINI_API_KEY`, `claude` → `ANTHROPIC_API_KEY`, etc.). See [`src/dictextractor/llm/client.py`](src/dictextractor/llm/client.py) for model-family quirks (Gemini 3 fixed temperature, Gemini 2.5 thinking-config).
- Preprocessing is **off by default**. Without it, PDFs flow straight to the LLM as `application/pdf` inline data. With `--preprocess`, PDFs are rasterised via PyMuPDF first (cv2 needs pixels).
- `assets/` is gitignored — sample dictionaries, gold labels, and outputs are local-only.

## Repository layout

```
src/dictextractor/
    schemas/        # Pydantic schemas — single source of truth for entries + OCR results
    preprocessing/  # cv2 pipeline (grayscale → deskew → denoise → contrast → sharpen)
    ocr/            # Pluggable OCR backends (mathpix, paddle_traditional, paddle_vl, …)
    extraction/     # LLM extraction strategies (manual, join, two_stage)
    llm/            # litellm wrapper + prompt templates
    evaluation/
        stage1/     # Transcription quality (TextEdit/GCER/WER + markup F1 + ReadOrderEdit)
        stage2/     # Entry-level matching + character-level error analysis
    utils/          # Shared helpers (text normalisation, image helpers, IO, viz)
    cli/            # argparse entry points (no business logic)
docs/               # architecture.md, stage1_evaluation_metrics.md, uv.md
examples/           # Working shell scripts for every CLI workflow
label-studio/       # Provisioning script for human post-editing projects
scripts/            # Helper scripts (e.g. PDF page extraction from full dictionaries)
```
