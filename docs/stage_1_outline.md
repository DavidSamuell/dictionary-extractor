# Stage 1 Outline (Quick Reference)

Stage 1 answers: **what text appears on the page, in what order, with what inline typography?** Stage 2 (see `stage_2_methodology.md`) parses that text into dictionary entries.

---

## Tracks

| Track | Mode | Primary output | Evaluation |
| --- | --- | --- | --- |
| **Column** | `--stage1-mode column` (default) | `*_stage1.tsv` | `dictextractor-eval-s1` |
| **Flat** | `--stage1-mode flat` | `*_stage1_flat.txt` | `dictextractor-eval-flat` |

Flat track is required for **fair comparison** between Gemini flat LLM runs and specialized VLM OCR (MinerU, Paddle, GLM).

---

## Flat spec v2 (one file per page)

Line order in every flat gold/pred file:

1. **Header** — TSV `column_id=header` rows, file order (or adapter-classified header blocks).
2. **Body** — column-major: `left` → `center`/`middle` → `right` → `single`; within column by `line_number`.
3. **Footer** — TSV `column_id=footer` rows, file order (or adapter-classified footer blocks).

Implementation: `src/dictextractor/evaluation/stage1/flatten.py` (`FLAT_SPEC_VERSION = "v2"`).

---

## File layout (samples tree)

```text
{lang}/
  outputs/
    stage-1-gold/{stem}/
      {stem}_stage1_GOLD.tsv          # human column gold
      {stem}_stage1_GOLD_flat.txt     # derived flat gold (eval-flat)
    stage-1/{experiment}/{stem}/
      {stem}_stage1_flat.txt          # flat pred (LLM or OCR adapter)
      {stem}_stage1.tsv               # column pred (LLM column mode only)
      {stem}_stage1_raw.json          # LLM structured response
      {stem}_stage1_input.json        # LLM request snapshot
      …                               # VLM artifacts (backend-specific)
```

---

## Pipelines (flat)

### A. General LLM (Gemini flat)

```text
Page image + optional alphabet/OCR hint
  → dictextractor-extract --stage1-mode flat
  → FlatTranscriptionResponse (header / lines / footer)
  → {stem}_stage1_flat.txt
```

- Script: `examples/stage-1/run_stage1_extraction_flat.sh`
- Prompt: `STAGE_1_FLAT_SYSTEM` in `src/dictextractor/llm/prompts.py`
- Experiments: `gemini3flash_flat_alpha_ocr`, `_noalpha_ocr`, `_alpha_noocr`, `_bare`

### B. VLM OCR + flatten

```text
Page image
  → VLM runner (MinerU / Paddle / GLM) → raw JSON/layout on disk
  → scripts/ocr_to_stage1_flat.py (or adapter at eval time)
  → layout blocks → OcrAdapter_v1 → normalize_typography v1
  → {stem}_stage1_flat.txt
```

- OCR run: `dictextractor` VLM CLI / batch jobs (see `src/dictextractor/ocr/vlm/`)
- Flatten script: `examples/helper/run_flatten_vlm_ocr.sh`
- Experiments: `MinerU2.5-Pro`, `PaddleOCR-VL-1.5`, `GLM-OCR`
- Adapter: `layout_to_transcript_v1` (`ADAPTER_VERSION = "v1"`)

---

## Gold flattening

```bash
uv run python scripts/flatten_stage1_gold.py \
  --samples-dir assets/dictionaries/samples
```

Always regenerate flat gold after gold TSV edits.

---

## Evaluation (flat)

```bash
bash examples/evaluation/run_stage1_eval_flat.sh
```

- Default in script: `--include-vlm-ocr` (4× Gemini flat + 3× OCR).
- **Do not** combine `--include-vlm-ocr` with `--experiment-name` for OCR only — that filters out Gemini.
- OCR-only refresh: uncomment the OCR block in the script (no `--include-vlm-ocr`).
- Cache: `evaluations/stage1_flat_eval/stage1_flat_eval_cache.json` — aggregate CSVs merge **all** valid cached experiments.

Metrics detail: `docs/stage_1_evaluation_metrics.md`.

---

## Versioning (frozen v1)

| Component | Version key | Module |
| --- | --- | --- |
| Flat line order | `v2` | `flatten.py` |
| OCR layout adapter | `v1` | `layout_to_transcript_v1.py` |
| Typography normalize | `v1` | `normalize_typography.py` |
| eval-flat cache | `3` | `stage1_eval_cache.py` |

---

## Related docs

- **Full methodology:** `docs/stage_1_methodology.md`
- **Metrics (eval-s1 vs eval-flat):** `docs/stage_1_evaluation_metrics.md`
- **Stage 2:** `docs/stage_2_methodology.md`
- **Engineering plan:** `PLAN.md` §3 (Layer 2 / adapter fairness)
