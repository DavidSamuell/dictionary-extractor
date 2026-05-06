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
evaluation/stage1/  (CER/WER/GCER/BLEU/NED + markup F1 + read-order)
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
    --model gemini/gemini-3-flash-preview --overwrite

# Stage 2 only — requires existing stage-1 TSVs
uv run dictextractor-extract \
    --strategy two_stage --stage 2 \
    --samples-dir assets/dictionaries/samples-2 \
    --languages Chepang-English \
    --model gemini/gemini-3.1-pro-preview \
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
        stage-1/<stem>/<stem>_stage1.tsv   + raw/input JSONs
        stage-2/<stem>/<stem>.tsv + .json  + *_usage.json
```

Already-processed pages are skipped automatically; pass `--overwrite` to force re-processing.

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
uv run python -m dictextractor.cli.evaluate_stage1 \
    --samples-dir assets/dictionaries/samples/ \
    -o assets/dictionaries/samples/stage1_eval
```

Reports cover GCER (grapheme CER, OCR-D spec), CER, WER, BLEU, NED, plus per-tag bold/italic F1 and read-order NED. Metric definitions and aggregation rules are in [`docs/stage1_evaluation_metrics.md`](docs/stage1_evaluation_metrics.md).

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
        stage1/     # Transcription quality (GCER/CER/WER/BLEU + markup F1 + read-order)
        stage2/     # Entry-level matching + character-level error analysis
    utils/          # Shared helpers (text normalisation, image helpers, IO, viz)
    cli/            # argparse entry points (no business logic)
docs/               # architecture.md, stage1_evaluation_metrics.md, uv.md
examples/           # Working shell scripts for every CLI workflow
label-studio/       # Provisioning script for human post-editing projects
scripts/            # Helper scripts (e.g. PDF page extraction from full dictionaries)
```
