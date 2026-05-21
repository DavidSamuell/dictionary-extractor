# Stage 1 Evaluation Metrics

**Pipeline and flattening (LLM flat, VLM OCR, gold spec v2):** see `docs/stage_1_methodology.md` and `docs/stage_1_outline.md`.

Stage 1 evaluation measures how well a model transcribes a dictionary page into tagged text. We compare **predicted** output against **gold** (human-verified ground truth) on three independent dimensions:

1. **Text recognition quality** — characters and words (tags ignored)
2. **Markup / typography** — bold and italic on the right words
3. **Read order** — whether content appears in the correct sequence

Two CLIs implement the same metric definitions on different file formats:

| Track | CLI | Pred / gold format | Primary use |
| --- | --- | --- | --- |
| **eval-s1** | `dictextractor-eval-s1` | Column TSV (`column_id \| line_number \| text`) | LLM column transcription, layout-aware gold |
| **eval-flat** | `dictextractor-eval-flat` | One line per row in flat `.txt` (spec v2) | Flat LLM transcription, VLM OCR adapters, fair OCR comparison |

---

## Tracks at a glance

| Aspect | eval-s1 | eval-flat |
| --- | --- | --- |
| Gold path | `*/stage-1-gold/<stem>/<stem>_stage1_GOLD.tsv` | `*/stage-1-gold/<stem>/<stem>_stage1_GOLD_flat.txt` |
| Pred path | `*/stage-1/<exp>/<stem>/<stem>_stage1.tsv` | `*/stage-1/<exp>/<stem>/<stem>_stage1_flat.txt` (or column TSV flattened at eval time) |
| Header / footer | **Excluded** from all metrics | **Included** (header → body → footer per flat spec v2) |
| Character + typography alignment | Adjacent row spans (default up to 3 rows) | **Page collapsed** to one string per side |
| Read-order alignment | Adjacent row spans (default up to 3 rows) | **Per-line** adjacent spans (same as eval-s1) |
| ReadOrderEdit | Reported | Reported |

**eval-s1 — header/footer exclusion:** Rows with `column_id="header"` or `column_id="footer"` are dropped before metrics. They are page metadata, not dictionary body; many gold TSVs predate the header/footer split. Re-enable later if you want explicit metadata accuracy.

**eval-flat — flat spec v2 line order:** Header lines (file order) → body (column-major from gold TSV, or adapter reading order for OCR) → footer lines. Generate gold flats with `python scripts/flatten_stage1_gold.py`. OCR flat preds are written after `vlm_ocr` or via `python scripts/ocr_to_stage1_flat.py`.

---

## Shared text normalisation

Applied **symmetrically** to predicted and gold before alignment and metrics:

| Step | Where | What |
| --- | --- | --- |
| Span / page text | Semantic alignment, TextEdit, GCER, WER | Tag-strip, NFC, collapse whitespace, remove space before punctuation (`,.:;!?…`) |
| Tagged line | Before `parse_tagged_words` | Same line normalisation (tags preserved) |
| Word key | Typography `SequenceMatcher` + similarity | Additionally strip punctuation from the word surface for slot matching only |
| Tags | TP/FP/FN | Compared on the **original** parsed `(word, tags)` — not on the normalised key |

**Span shape is already equivalent:** `<b>hello world</b>` and `<b>hello</b> <b>world</b>` both yield `(hello, {b})` and `(world, {b})` after parsing.

---

## 1. Text Recognition Quality

Measures how accurately the model recognizes text content, **ignoring all typography tags** (`<b>`, `<i>`, etc.). Predicted and gold text are aligned before grapheme and word error rates are computed.

### eval-s1: semantic span alignment

1. Remove header/footer rows.
2. Build adjacent predicted and gold row spans (default: up to `--alignment-max-span-rows` rows, default `3`).
3. Score each candidate span pair with grapheme NED: `distance / max(len(pred), len(gold), 1)`.
4. Greedily select pairs with similarity `1 - NED` ≥ `--alignment-threshold` (default `0.5`), without overlapping source rows.
5. Unmatched gold spans → missing content; unmatched predicted spans → extra content.

### eval-flat: page-collapsed alignment (character metrics only)

1. Join **all** flat lines on each side with spaces (same join rule as multi-row spans in eval-s1).
2. Apply `clean_text` normalisation to produce one predicted string and one gold string.
3. Align as a **single span pair** (`align_page_collapsed`) — no multi-line fuzzy search, no line-split penalty.

Line boundaries do **not** affect GCER, WER, or TextEdit on eval-flat. Wrong global order still hurts character metrics because the collapsed strings differ in sequence.

### Metrics

| Metric | Definition | Interpretation |
| --- | --- | --- |
| **TextEdit** | Mean grapheme NED over aligned spans; unmatched spans score `1.0` | OmniDocBench-style headline text score. Lower is better; `0` is perfect. |
| **GCER** | `total_grapheme_edits / total_graphemes_gold` | OCR-D-style grapheme character error rate. Lower is better. |
| **WER** | Word-level error rate after alignment (jiwer over aligned span texts) | Lower is better. |

### Diagnostics (eval-s1; eval-flat page mode)

- **matched_spans** — Successfully matched units (eval-flat: `0` or `1` for the page pair).
- **missing_spans** — Gold with no matched prediction.
- **extra_spans** — Prediction with no matched gold.

---

## 2. Markup / Typography Preservation

Measures whether bold and italic are applied to the right words. Minor character errors are tolerated via fuzzy word alignment inside each aligned unit.

### Alignment

1. **eval-s1:** Use the semantic span alignment from §1.
2. **eval-flat:** Use the **page-collapsed** pair from §1 (one unit per page).
3. Within each matched unit, align words with `SequenceMatcher` on normalised tag-stripped text.
4. Reject word pairs with character similarity &lt; `0.5`; treat as separate insertion/deletion.

Example: gold `<b>hello</b>` vs pred `<b>hello ,</b>` — span text can match after space-before-punct normalisation; typography may still differ on the word key.

### Per-tag metrics

| Metric | Definition |
| --- | --- |
| **Precision** | `TP / (TP + FP)` |
| **Recall** | `TP / (TP + FN)` |
| **F1** | `2 * P * R / (P + R)` |

**Typography F1** pools bold and italic counts before P/R/F1 (`typography_TP = bold_TP + italic_TP`, etc.). Use this as the single markup headline in comparison CSVs; per-tag P/R/F1 appear in full reports.

- **TP:** Tagged in both pred and gold.
- **FP:** Tagged in pred only.
- **FN:** Tagged in gold only, or word/line missing.

Words in missing spans → FN; words in extra spans → FP.

---

## 3. Read Order (Structure Preservation)

Measures whether the model reads the page in the correct order — important for multi-column dictionaries (e.g. reading across columns instead of down each column).

### How it works

Uses **line-level** semantic span alignment (not page collapse):

1. Build adjacent spans on **per-line** rows (eval-s1: after header/footer drop; eval-flat: each flat file line is one row).
2. Greedily match spans with `--alignment-threshold` / `--alignment-max-span-rows` (defaults `0.5` / `3`).
3. Sort matched gold spans by gold order → canonical ID sequence.
4. Sort matched predicted spans by pred order; map each to its matched gold span ID (unmatched pred → `extra:<idx>`).
5. Normalized Levenshtein distance between sequences.

| Metric | Definition | Interpretation |
| --- | --- | --- |
| **ReadOrderEdit** | `levenshtein(pred_sequence, gold_sequence) / max(len(pred), len(gold), 1)` | Lower is better; `0` means all matched spans appear in canonical order. |

On **eval-flat**, character and typography intentionally ignore line splits; **ReadOrderEdit** is the metric that still penalises line reordering, merged lines, and column-order mistakes.

---

## Aggregation

When evaluating multiple pages:

| Component | Aggregation |
| --- | --- |
| TextEdit | **Span-weighted average** across pages |
| GCER, WER | **Micro-average** (sum numerators / denominators) |
| Markup P/R/F1 | **Micro-average** (sum TP, FP, FN) |
| ReadOrderEdit | **Macro-average** (mean per-page scores) |

---

## eval-s1 usage

```bash
# Single file
uv run dictextractor-eval-s1 \
    -p page_1_stage1.tsv -g page_1_stage1_GOLD.tsv

# Batch: all experiments under samples
uv run dictextractor-eval-s1 \
    --samples-dir assets/dictionaries/samples \
    --all-experiments \
    --alignment-threshold 0.5 --alignment-max-span-rows 3 \
    -o evaluations/stage1_eval
```

**Layout:** predictions at `<lang>/outputs/stage-1/<experiment>/<stem>/<stem>_stage1.tsv`, gold at `<lang>/outputs/stage-1-gold/<stem>/<stem>_stage1_GOLD.tsv`.

**Cache:** `<out>/stage1_eval_cache.json` — invalid on pred/gold mtime+size change, alignment flag change, or cache format bump in `stage1_eval_cache.py`.

**Outputs:**

- `stage1_eval_detailed.csv`, `stage1_eval_summary.csv`
- `<out>/<experiment>/stage1_evaluation_report.{txt,json}`
- `character_recognition.csv`, `markup_preservation.csv`, `structure_preservation.csv`

**Metrics profiles:** `--metrics full` (default) vs `--metrics minimal` (`TextEdit`, `GCER`, `WER`, `typography_f1`, `ReadOrderEdit`).

---

## eval-flat usage

```bash
# Single page
uv run dictextractor-eval-flat \
    -p path/to/page_stage1_flat.txt \
    -g path/to/page_stage1_GOLD_flat.txt

# Batch (example wrapper)
bash examples/evaluation/run_stage1_eval_flat.sh

# Flat LLM ablations + VLM OCR backends
bash examples/evaluation/run_stage1_eval_flat.sh --include-vlm-ocr

# Force recompute after metric code changes
bash examples/evaluation/run_stage1_eval_flat.sh --overwrite
```

**Experiment selection** (mutually exclusive modes):

| Flag | What gets evaluated |
| --- | --- |
| `--experiment-name NAME` (repeatable) | Named folders only |
| `--experiment-name-contains flat` | Folder names containing `flat` (LLM flat ablations) |
| `--include-vlm-ocr` | `*flat*` experiments **plus** `MinerU2.5-Pro`, `PaddleOCR-VL-1.5`, `GLM-OCR` if present |
| `--all-experiments` | Every folder under `outputs/stage-1` (includes column-mode `gemini3flash_*` — usually not wanted for flat benchmarks) |

**Alignment flags** (read order only; character/typography always use page collapse):

- `--alignment-threshold` (default `0.5`)
- `--alignment-max-span-rows` (default `3`)

**Cache:** `<out>/stage1_flat_eval_cache.json` (format version **3**: page collapse for char/markup + line-level read order). Use `--overwrite` after code changes.

**Outputs** (default `evaluations/stage1_flat_eval/`):

- `stage1_flat_eval_detailed.csv`, `stage1_flat_eval_summary.csv`
- `stage1_flat_eval_cache.json`
- `<out>/<experiment>/stage1_flat_evaluation_report.{txt,json}`
- Per-experiment CSV drill-downs via the shared report helper

**Metrics profiles:** `--metrics minimal` (default on CLI) vs `--metrics full` (per-tag bold/italic P/R/F1).

---

## Comparing experiments

Detailed and summary CSVs are the canonical artifacts for cross-configuration analysis. Each ablation (alphabet on/off, OCR hint, model, OCR backend) is a distinct `experiment` value. Use the summary CSV for per-language leaderboards; use the detailed CSV with `__aggregate__` rows for corpus-level scores. `alphabet` and `ocr-hint` columns come from `run_config.json` beside predictions where available.

**Fair OCR comparison:** Prefer **eval-flat** with `--include-vlm-ocr` so all systems share the same flat contract and gold flattening rule. **eval-s1** remains useful when scoring column TSV directly (layout-aware gold grid).
