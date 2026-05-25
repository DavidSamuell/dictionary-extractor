# Stage 1 Methodology: Page Transcription

Stage 1 is the **faithful transcription** layer of the dictionary-extractor pipeline. It copies visible text from a dictionary page image into a machine-readable format. It does **not** segment entries, assign parts of speech, or map fields to MDF — that is Stage 2 (`docs/stage_2_methodology.md`).

This document covers:

1. **Column mode** (layout-aware TSV) — default LLM Stage 1 output; optional Stage 2 input via `--stage1-input column`.
2. **Flat mode** — one line per visible row, spec **v2**, used for OCR benchmarks, fair cross-paradigm comparison, and **default Stage 2 direct MDF input** (`--stage1-input flat`).
3. **Gold flattening** — deriving flat gold from column gold.
4. **VLM OCR flattening** — MinerU / Paddle / GLM → flat preds via a frozen geometry adapter.
5. **Typography normalization** — shared cleanup before flat write and eval.
6. **Evaluation** — `eval-flat` (metrics in `docs/stage_1_evaluation_metrics.md`).

Quick reference: `docs/stage_1_outline.md`.

---

## 1. Role in the pipeline

```text
                    ┌─────────────────────────────────────┐
Page image ────────►│ Stage 1: transcription              │
Optional alphabet   │  • column TSV  OR  flat .txt        │
Optional OCR hint   └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   eval-flat (flat)         Stage 2 direct MDF          Stage 2 schema
   layout-aware gold        flat or column transcript   (legacy JSON/TSV)
```

| Question | Stage 1 | Stage 2 |
| --- | --- | --- |
| What characters appear? | Yes | Uses Stage 1 transcript (characters copied verbatim in direct MDF) |
| Bold / italic on words? | Yes (`<b>`, `<i>`) | Tags stripped when emitting MDF |
| Column vs global read order? | Column TSV encodes grid; flat encodes one ordered line list | `--stage1-input` selects transcript source |
| Entry boundaries, glosses, MDF? | No | Yes (`direct_mdf` default) |

**Design principle:** Stage 1 must not “fix” dictionary content (no merging hyphenated lines across rows in flat mode, no paraphrase). Stage 2 may apply linguistic judgment.

---

## 2. Output formats

### 2.1 Column TSV (default LLM)

**Path:** `{lang}/outputs/stage-1/<experiment>/<stem>/<stem>_stage1.tsv`

**Schema:** tab-separated, header `column_id`, `line_number`, `text`.

| `column_id` | Meaning |
| --- | --- |
| `header` / `footer` | Page metadata; empty `line_number` |
| `left`, `center`, `right`, `single` | Body columns (`middle` aliases `center` in flatten) |

**Gold:** `{lang}/outputs/stage-1-gold/<stem>/<stem>_stage1_GOLD.tsv`

**Produced by:** `dictextractor-extract --stage1-mode column` with `TranscriptionResponse` (`src/dictextractor/schemas/entry.py`).

**Stage 2 input:** `--stage1-input auto|column|flat` (default `auto`). Batch direct MDF (`examples/stage-2/run_stage2_extraction.sh`) typically uses **`flat` gold** from `stage-1-gold/`. Column TSV is still preferred for column-trilingual layouts when `column_id` matters.

### 2.2 Flat text (spec v2)

**Path:** `{lang}/outputs/stage-1/<experiment>/<stem>/<stem>_stage1_flat.txt`

**Contract:** one string per visible line; lines separated by `\n`; order:

1. **Header lines** (top of page).
2. **Body lines** — for multi-column pages: complete **column-major** order (all lines in left column top→bottom, then next column, etc.). This matches the gold flatten rule from column TSV, not “read across rows” unless the source truly used that order.
3. **Footer lines** (bottom of page).

**Gold:** `{lang}/outputs/stage-1-gold/<stem>/<stem>_stage1_GOLD_flat.txt` — **derived** from column gold TSV, not hand-edited separately.

**Implementation:** `src/dictextractor/evaluation/stage1/flatten.py` (`FLAT_SPEC_VERSION = "v2"`).

**Produced by:**

- LLM: `--stage1-mode flat` → `FlatTranscriptionResponse` → `flat_transcription_to_text()`.
- OCR: VLM artifacts → `OcrAdapter_v1` → `write_stage1_flat_for_page()`.
- At eval time: column preds can be flattened on the fly (`flat_evaluator._load_pred_lines`).

**Evaluation:** `dictextractor-eval-flat` — see §6.

---

## 3. Flat gold generation

Gold flat files must follow the same v2 rule as predictions so OCR and LLM are scored against a single canonical line list.

### 3.1 Algorithm

From each `*_stage1_GOLD.tsv`:

1. Collect `header` rows → lines in **file order** (`text` column only).
2. Collect body rows → sort columns `left` (0) → `center`/`middle` (1) → `right` (2) → `single` (3); within column sort by `line_number`, then file index.
3. Collect `footer` rows → lines in **file order**.
4. Concatenate: `header + body + footer`; write `*_stage1_GOLD_flat.txt`.

### 3.2 Command

```bash
uv run python scripts/flatten_stage1_gold.py \
  --samples-dir assets/dictionaries/samples
```

Optional: `--languages Evenki-Russian Yiddish-English`, `--dry-run`, `-v`.

**When to re-run:** whenever column gold TSV changes.

---

## 4. Flat LLM transcription (general / Gemini)

### 4.1 CLI

```bash
uv run dictextractor-extract \
  --strategy two_stage \
  --stage 1 \
  --stage1-mode flat \
  --model gemini/gemini-3-flash-preview \
  --samples-dir assets/dictionaries/samples \
  --languages Evenki-Russian \
  --experiment-name gemini3flash_flat_alpha_ocr
```

**Constraints** (`src/dictextractor/cli/extract.py`):

- Requires `--strategy two_stage`.
- Supports `--stage 1` only (no Stage 2 in the same run).

Batch wrapper with ablation matrix: `examples/stage-1/run_stage1_extraction_flat.sh`.

### 4.2 Model I/O

| Piece | Location |
| --- | --- |
| System prompt | `STAGE_1_FLAT_SYSTEM` — `src/dictextractor/llm/prompts.py` |
| User prompt | `stage_1_user()` — alphabet, optional `<ocr_reference>`, optional guides |
| Response schema | `FlatTranscriptionResponse` — fields `header`, `lines`, `footer` |
| Serialization | `flat_transcription_to_text()` in `flatten.py` |
| Sidecar files | `{stem}_stage1_raw.json`, `{stem}_stage1_input.json` |

The model is instructed to:

- Emit **one string per visible printed line** in global reading order (column-major for multi-column pages).
- Preserve diacritics; use `<b>` / `<i>` when confident.
- Keep hyphenated wraps as **two lines** (e.g. `intelligi-` then `ble, adj. …`).
- Put running titles / page numbers in `header` or `footer`, not in body `lines`.

Optional inputs (same as column Stage 1):

- **Alphabet** — text and/or image under the language folder.
- **OCR hint** — reference text for ambiguous glyphs only; image takes priority.

### 4.3 Experiment matrix (samples)

Typical flat ablations under `outputs/stage-1/`:

| Experiment folder | Alphabet | OCR hint |
| --- | --- | --- |
| `gemini3flash_flat_alpha_ocr` | yes | yes |
| `gemini3flash_flat_noalpha_ocr` | no | yes |
| `gemini3flash_flat_alpha_noocr` | yes | no |
| `gemini3flash_flat_bare` | no | no |

`run_config.json` beside outputs records `alphabet` and `ocr-hint` for eval CSVs.

---

## 5. VLM OCR and flat flattening

Specialized document OCR models (MinerU 2.5 Pro, PaddleOCR-VL 1.5, GLM-OCR) emit **layout JSON or markdown**, not dictionary flat text. The repo normalizes them through a **frozen geometry adapter** so they compete on the same flat contract as Gemini flat LLM.

### 5.1 End-to-end flow

```text
Page image (snippet)
    │
    ▼
VlmOcrRunner.run_page()          # src/dictextractor/ocr/vlm/
    │  mineru: content.json, layout trees
    │  paddle: *_res.json, parsing_res_list
    │  glm: result.json, output.txt, layout_details
    ▼
Page directory under stage-1/<experiment>/<stem>/
    │
    ▼
scripts/ocr_to_stage1_flat.py    # or flat_export at import time
    │  detect_backend(page_dir)
    │  mineru_blocks / paddle_blocks / glm_transcript
    ▼
layout_to_transcript_v1(blocks)  # ADAPTER_VERSION = "v1"
    │  classify header / body / footer (category + y-bands)
    │  cluster x-centers → ≤3 columns
    │  body: column-major, y then x within column
    │  normalize_line() per line
    ▼
{stem}_stage1_flat.txt           # spec v2 line list
```

**Batch flatten** (after VLM jobs complete):

```bash
bash examples/helper/run_flatten_vlm_ocr.sh
```

Equivalent:

```bash
uv run python scripts/ocr_to_stage1_flat.py \
  --samples-dir assets/dictionaries/samples \
  --experiment MinerU2.5-Pro \
  --experiment PaddleOCR-VL-1.5 \
  --experiment GLM-OCR
```

### 5.2 Backend detection and readers

| Experiment folder | Backend enum | Artifact signals | Reader module |
| --- | --- | --- | --- |
| `MinerU2.5-Pro` | `mineru` | `content.json` (nested or flat) | `ocr/adapters/mineru.py` |
| `PaddleOCR-VL-1.5` | `paddle` | `*_res.json`, `parsing_res_list` | `ocr/adapters/paddle.py` |
| `GLM-OCR` | `glm` | `result.json`, `output.txt` | `ocr/adapters/glm.py` |

All layout backends share `layout_to_transcript_v1` (`ocr/adapters/layout_to_transcript_v1.py`). GLM may bypass block clustering when a line-oriented transcript is already available.

**Layout blocks** (`ocr/adapters/blocks.py`): normalized `{x0, y0, x1, y1, text, category}` in page-relative coordinates.

### 5.3 OcrAdapter_v1 rules (summary)

| Step | Rule |
| --- | --- |
| Classify | Map VLM categories to header / body / footer; skip figures; optional y-bands (top 12%, bottom 15%) |
| Columns | Cluster block x-centers; up to 3 columns; wide single column if spread &lt; 0.12 |
| Body order | Column-major: all blocks in column 0, then 1, then 2; within column sort by `y_min`, `x0` |
| Line split | Split block text on `\n`; strip HTML table tags to one line per row when needed |
| Typography | Per-line `normalize_line()` before append; empty/junk lines dropped |

**Explicit non-goals for v1:** no per-language gutter tuning; no oracle reorder to match gold; LTR column-major assumption (RTL may degrade).

Parser details and fairness rules: `PLAN.md` §3–§5.

### 5.4 Typography normalization (v1)

**Module:** `src/dictextractor/evaluation/stage1/normalize_typography.py` (`TYPOGRAPHY_SPEC_VERSION = "v1"`).

Applied to **every OCR flat line** (and available for any flat pred). Order:

1. **OCR garbage strip** — HTML entities; MinerU `x<i>{n}` runs; LaTeX unwrap (`\text`, `\overline`, subscripts `_{}`, braced spans, `arrayl` blocks); stray `\`, `{}`; swastika runs; checkboxes; modifier letters; `[Unreadable]`; U+FFFD.
2. **LaTeX / markdown / HTML** → dictionary `<b>` / `<i>` where applicable.
3. **Strip unknown tags** — keep only `<b>`, `<i>`, and POS marker `(<N>)`.
4. **Unicode NFC** — via `normalize_unicode`.
5. **Junk line filter** — drop lines that are digit spam, low-diversity symbol runs, 12+ char repeats (including Hebrew after stripping combining marks), CJK-only garbage lines, lone backslashes, or lines containing U+FFFD.

**Idempotent** on lines that already match gold tag conventions.

**Audit** residual patterns after flatten:

```bash
uv run python scripts/audit_ocr_flat_noise.py
```

### 5.5 Known OCR flatten limitations

| Issue | Cause | Mitigation |
| --- | --- | --- |
| Empty or tiny flat files | Paddle/MinerU labeled all blocks as header/footer/reference | Fix upstream VLM or accept missing body in metrics |
| Typography F1 ≈ 0 | VLM JSON rarely emits `<b>`/`<i>` | Expected; use GCER/WER for OCR typography-agnostic score |
| High ReadOrderEdit | Adapter column-major ≠ gold line order on complex scripts | Layout v2 / reading-order research track |
| Single-line arrow/glyph noise | OCR misread on citations | Rare; not expanded regex unless pattern recurs |

---

## 6. Evaluation

Stage 1 metrics are documented in **`docs/stage_1_evaluation_metrics.md`**. Overview of both tracks: **`docs/evaluation_metrics.md`**. Summary:

| CLI | Pred / gold | Character + markup alignment | Read order |
| --- | --- | --- | --- |
| `dictextractor-eval-flat` | Flat `.txt` | **Page collapsed** (one string per side) | **Gold line indices** (OmniDocBench-style) |

**Fair OCR comparison:** always use **eval-flat** with the same `*_stage1_GOLD_flat.txt` and v2 flatten rule.

### 6.1 Batch eval-flat

```bash
bash examples/evaluation/run_stage1_eval_flat.sh
```

Active block uses `--include-vlm-ocr` (four `gemini3flash_flat_*` + three OCR backends). Uncomment alternate blocks in the script for OCR-only or explicit experiment lists.

**Important:** `--include-vlm-ocr` together with `--experiment-name MinerU2.5-Pro` (etc.) **intersects** the lists — only OCR runs; Gemini is skipped. Use one mode or the other.

### 6.2 Incremental cache and merged CSVs

**Cache file:** `evaluations/stage1_flat_eval/stage1_flat_eval_cache.json` (format version **4**).

- Each page stores metrics plus fingerprints of pred/gold files and alignment settings.
- A run with `--experiment-name` limits **recomputation** to those experiments.
- **`stage1_flat_eval_detailed.csv` / `stage1_flat_eval_summary.csv`** are built from **all valid cache entries** for every experiment discovered on disk (`collect_valid_metrics`), not only the experiments selected on the CLI.

Workflow:

1. Full eval once with `--include-vlm-ocr` to populate Gemini + OCR cache.
2. After OCR adapter or flatten changes: `run_flatten_vlm_ocr.sh`, then OCR-only eval with `--overwrite` (Gemini rows stay in CSV from cache).

Re-run flat gold + full eval after gold TSV edits or metric code changes.

### 6.3 Outputs

Under `evaluations/stage1_flat_eval/`:

- `stage1_flat_eval_detailed.csv` — per (experiment, page)
- `stage1_flat_eval_summary.csv` — per (experiment, language)
- `<experiment>/stage1_flat_evaluation_report.{txt,json}`
- Per-experiment drill-down CSVs (`character_recognition.csv`, etc.)

---

## 7. Fairness and versioning

| Artifact | Version | Change policy |
| --- | --- | --- |
| Flat line order spec | **v2** | Bump `FLAT_SPEC_VERSION`; regenerate all `*_GOLD_flat.txt` |
| OCR layout adapter | **v1** | New behavior → `adapter_v2`, document in paper |
| Typography normalization | **v1** | Extend `normalize_typography.py`; re-flatten OCR preds |
| eval-flat cache | **4** | Bump `CACHE_FORMAT_VERSION` in `stage1_eval_cache.py` invalidates old cache |

**Allowed:** frozen prompts, frozen adapter, symmetric normalization for all OCR backends, per-language Stage 1 **guides** (LLM only).

**Not allowed on the eval set:** tuning adapter thresholds or regexes against eval scores; per-page hand fixes; different postprocess per model without labeling a new experiment.

---

## 8. Operational checklist

| Task | Command / script |
| --- | --- |
| Regenerate flat gold | `scripts/flatten_stage1_gold.py` |
| Run Gemini flat ablations | `examples/stage-1/run_stage1_extraction_flat.sh` |
| Run VLM OCR (per backend) | VLM CLI / batch (see `ocr/vlm/`) |
| OCR JSON → flat preds | `examples/helper/run_flatten_vlm_ocr.sh` |
| Audit OCR flat noise | `scripts/audit_ocr_flat_noise.py` |
| Eval flat (LLM + OCR) | `examples/evaluation/run_stage1_eval_flat.sh` |
| Eval Stage 2 MDF | `examples/evaluation/run_stage2_eval_mdf.sh` |

---

## 9. Code map

| Concern | Module / script |
| --- | --- |
| Flat spec + gold write | `evaluation/stage1/flatten.py` |
| Flat eval + discovery | `evaluation/stage1/flat_evaluator.py` |
| eval-flat CLI | `cli/evaluate_stage_flat.py` |
| Report generation | `evaluation/stage1/stage1_reports.py` |
| LLM flat extract | `extraction/llm_two_stage.py`, `llm/prompts.py` |
| OCR → flat export | `ocr/adapters/flat_export.py` |
| Layout adapter v1 | `ocr/adapters/layout_to_transcript_v1.py` |
| Backend parsers | `ocr/adapters/mineru.py`, `paddle.py`, `glm.py` |
| Typography v1 | `evaluation/stage1/normalize_typography.py` |
| VLM runners | `ocr/vlm/mineru.py`, `paddle_vl.py`, `glm_ocr.py` |

---

## 10. Related documentation

- `docs/stage_1_outline.md` — one-page quick reference
- `docs/stage_1_evaluation_metrics.md` — TextEdit, GCER, WER, typography F1, ReadOrderEdit
- `docs/stage_2_evaluation_metrics.md` — Record Accuracy, MDF Fields F1, ReadOrderEdit (Stage 2 MDF)
- `docs/evaluation_metrics.md` — overview of both evaluation tracks
- `docs/stage_2_methodology.md` — Pass 1 discovery + Pass 2 direct MDF (and legacy schema mode)
- `README.md` — CLI reference and repo layout
- `PLAN.md` — Layer 2 flat eval and fairness rules
