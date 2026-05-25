# Stage 1 Evaluation Metrics

**Pipeline and flattening (LLM flat, VLM OCR, gold spec v2):** see `docs/stage_1_methodology.md` and `docs/stage_1_outline.md`.  
**Overview of both evaluation tracks:** see `docs/evaluation_metrics.md`.

Stage 1 evaluation measures how well a model transcribes a dictionary page into tagged text. We compare **predicted** flat output against **gold** flat on three dimensions:

1. **Text recognition quality** — characters and words (tags ignored)
2. **Markup / typography** — bold and italic on the right words
3. **Reading order** — OmniDocBench-style ReadOrderEdit over gold line indices

**CLI:** `dictextractor-eval-flat` (the sole Stage 1 evaluation track)  
**Format:** one line per row in flat `.txt` (spec v2)

| Path | Location |
| --- | --- |
| Gold | `*/outputs/stage-1-gold/<stem>/<stem>_stage1_GOLD_flat.txt` |
| Pred | `*/outputs/stage-1/<experiment>/<stem>/<stem>_stage1_flat.txt` (or column TSV flattened at eval time) |

**Flat spec v2 line order:** Header lines (file order) → body (column-major from gold TSV, or adapter reading order for OCR) → footer lines. Generate gold flats with `python scripts/flatten_stage1_gold.py`. OCR flat preds are written after `vlm_ocr` or via `python scripts/ocr_to_stage1_flat.py`.

---

## Alignment architecture

Character and typography metrics default to **OmniDocBench quick_match** (`align_lines_quick_match`, `--character-alignment quick_match`). Optional **page collapse** joins the whole page first.

| Mode | Function | Line splits/merges | Swapped lines (char score) |
| --- | --- | --- | --- |
| **quick_match** (default) | `align_lines_quick_match` | Adjacent pred merge (OmniDocBench) | Not penalized |
| **Collapsed** | `align_page_collapsed` | Ignored (one page string) | Penalized |

Flat files are loaded as synthetic single-column rows (`column_id=single`, `line_number=1…N`). Column TSV predictions can be flattened on the fly before evaluation.

---

## Shared text normalisation

Applied **symmetrically** to predicted and gold before alignment and metrics:

| Step | Where | What |
| --- | --- | --- |
| Span / page text | Semantic alignment, TextEdit, GCER, WER | Tag-strip, NFC, collapse whitespace, remove space before punctuation (`,.:;!?…`) |
| Tagged line | Before `parse_tagged_words` | Same line normalisation (tags preserved) |
| Word key | Typography `SequenceMatcher` + similarity | Additionally strip punctuation from the word surface for slot matching only |
| Tags | TP/FP/FN | Compared on the **original** parsed `(word, tags)` — not on the normalised key |

Line-level quick_match scores candidates with **grapheme normalised edit distance** (NED). Each flat line is one unit; adjacent pred lines may merge (Adjacency Search Match). Pairs above fixed OmniDocBench thresholds are accepted via Hungarian assignment plus fuzzy subset matching.

---

## 1. Text recognition quality

Character metrics default to **quick_match** (`--character-alignment quick_match`).

With **quick_match** (default):

1. Match predicted lines to gold lines by content (one line per unit; adjacent pred merge allowed).
2. Pool grapheme edits over matched units; unmatched units score as errors.

With **page collapse** (`--character-alignment collapsed`):

1. Join **all** flat lines on each side with spaces.
2. Apply `clean_text` normalisation to produce one predicted string and one gold string.
3. Align as a **single span pair** (`align_page_collapsed`).

quick_match ignores swapped lines but merges split OCR lines. Collapsed mode ignores all line boundaries but penalizes global character reordering.

| Metric | Definition | Interpretation |
| --- | --- | --- |
| **TextEdit** | Mean grapheme NED over aligned spans; unmatched spans score `1.0` | OmniDocBench-style headline text score. Lower is better. |
| **GCER** | `total_grapheme_edits / total_graphemes_gold` | Grapheme character error rate. Lower is better. |
| **WER** | Word-level error rate after alignment | Lower is better. |

---

## 2. Markup / typography preservation

Uses the same alignment as character metrics (quick_match or collapsed). Within each matched unit:

1. Parse inline `<b>` / `<i>` tags into `(word, tag-set)` tuples.
2. Align words with `SequenceMatcher` on normalised tag-stripped text.
3. Accept word pairs only when character similarity ≥ `0.5`.
4. Score bold and italic tags independently on aligned word pairs.

| Metric | Definition |
| --- | --- |
| **Bold / italic precision** | `TP / (TP + FP)` per tag type |
| **Bold / italic recall** | `TP / (TP + FN)` per tag type |
| **Bold / italic F1** | Harmonic mean per tag type |
| **Typography F1** | P/R/F1 with **bold and italic counts pooled** before computing precision and recall |

Use **Typography F1** as the single markup headline in comparison CSVs (`--metrics minimal`). Use `--metrics full` when you need separate bold and italic P/R/F1 columns.

---

## 3. Reading order (structure preservation)

OmniDocBench-style **ReadOrderEdit** over **gold line indices** (after quick_match alignment):

1. Take all line pairs matched at the alignment step.
2. Sort pairs by predicted line index (reading order in the pred file).
3. `gt = [0, 1, …, n−1]` — every gold line index (including unmatched).
4. `pred` — matched gold indices in predicted reading order (flattened when a span covers multiple lines).
5. Unmatched predicted lines are omitted from `pred`.

With **`--character-alignment collapsed`**, ReadOrderEdit uses sequential anchor search in the joined pred string instead (line boundaries ignored for matching).

| Metric | Definition | Interpretation |
| --- | --- | --- |
| **ReadOrderEdit** | `levenshtein(gt, pred) / max(len(gt), len(pred), 1)` | Lower is better; `0` = perfect order among detected lines. |

Missing lines hurt read order because their gold indices stay in `gt` but never appear in `pred`. Swapping two detected lines also increases edit distance. Content-correct reordering does **not** affect GCER under quick_match but **does** affect ReadOrderEdit.

---

## Aggregation

| Component | Aggregation |
| --- | --- |
| TextEdit | **Span-weighted average** across pages |
| GCER, WER | **Micro-average** |
| Markup P/R/F1 (bold, italic, typography) | **Micro-average** |
| ReadOrderEdit | **Macro-average** (mean of per-page scores) |

---

## Usage

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
| `--experiment-name-contains flat` | Folder names containing `flat` |
| `--include-vlm-ocr` | `*flat*` experiments plus MinerU / Paddle / GLM OCR if present |
| `--all-experiments` | Every folder under `outputs/stage-1` |

**Alignment flags** (span matching for character/typography):

- `--character-alignment collapsed|quick_match` (default `quick_match`)
- `--alignment-threshold` (deprecated; quick_match uses fixed OmniDocBench thresholds)

**Metrics profile:**

- `--metrics minimal` (default) — headline columns only
- `--metrics full` — separate bold and italic P/R/F1 in CSVs and JSON

**Cache:** `<out>/stage1_flat_eval_cache.json` (format version **12**). Use `--overwrite` after metric code changes.

---

## Outputs

Default directory: `evaluations/stage1_flat_eval/`

| File | Contents |
| --- | --- |
| `stage1_flat_eval_detailed.csv` | Per (experiment, page) metrics + `alphabet` / `ocr-hint` flags from `run_config.json` |
| `stage1_flat_eval_summary.csv` | Per (experiment, language) aggregates |
| `<experiment>/stage1_flat_evaluation_report.{txt,json}` | Human-readable and machine-readable per-page reports |
| `<experiment>/character_recognition.csv` | TextEdit, GCER, WER drill-down |
| `<experiment>/markup_preservation.csv` | Typography (and bold/italic in full profile) |

**Detailed / summary CSV columns (`--metrics minimal`):**

`experiment`, `page_id` (detailed only) or `language` + `page_count` (summary), `alphabet`, `ocr-hint`, `TextEdit`, `GCER`, `WER`, `typography_f1`, `ReadOrderEdit`

**Additional columns with `--metrics full`:**

`bold_precision`, `bold_recall`, `bold_f1`, `italic_precision`, `italic_recall`, `italic_f1`

**JSON structure (per page):**

- `character_quality` — TextEdit, GCER, WER, grapheme/word edit counts, matched/missing/extra spans
- `markup_quality` — bold, italic, and pooled typography P/R/F1 with TP/FP/FN

---

## Comparing experiments

Use the detailed and summary CSVs for cross-configuration analysis. Each ablation is a distinct `experiment` value. `alphabet` and `ocr-hint` columns come from `run_config.json` beside predictions where available.

**Fair OCR comparison:** use `--include-vlm-ocr` so all systems share the same flat contract and gold flattening rule.

**Fair comparison checklist:**

1. Same gold flat (`stage-1-gold`, spec v2).
2. Same `--character-alignment` and `--alignment-threshold`.
3. Same `--metrics` profile when comparing typography breakdowns.

**What Stage 1 evaluation does not measure:**

- Lexicon record detection or MDF marker assignment → Stage 2 `eval-stage2-mdf`
- Entry-level headword/gloss matching → legacy `dictextractor-evaluate` (TSV)
- Layout reconstruction quality beyond reading-order edit distance
- Semantic correctness of dictionary content (paraphrase, merged entries)

---

## Implementation reference

| Component | Module |
| --- | --- |
| Flat evaluator | `evaluation/stage1/flat_evaluator.py` |
| Page / line alignment | `evaluation/stage1/alignment.py` |
| Character metrics | `evaluation/stage1/character_quality.py` |
| Typography metrics | `evaluation/stage1/markup_quality.py` |
| Report generation | `evaluation/stage1/stage1_reports.py` |
| Eval cache | `evaluation/stage1/stage1_eval_cache.py` (format version **12**) |
| CLI | `cli/evaluate_stage_flat.py` |
| Gold flattening | `evaluation/stage1/flatten.py` |
