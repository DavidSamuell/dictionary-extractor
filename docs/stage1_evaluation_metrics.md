# Stage 1 Evaluation Metrics

Stage 1 evaluation measures how well a vision-language model (VLM) transcribes a dictionary page image into structured, tagged text. The evaluation compares a **predicted** TSV (model output) against a **gold** TSV (human-verified ground truth), both sharing the same format: `column_id | line_number | text`.

**Header / footer rows are excluded from every metric.** Stage 1 emits page-level metadata (running titles, page numbers, chapter abbreviations, footnotes) into rows with `column_id="header"` or `column_id="footer"` and empty `line_number`. These rows are dropped before computing character quality, markup quality, and read-order metrics — they are not transcription content, and existing gold TSVs predate the header/footer split. Backfill the golds and re-enable inclusion later if you want explicit metadata accuracy tracking.

We evaluate three independent quality dimensions:

---

## 1. Text Recognition Quality

Measures how accurately the model recognizes text content, **ignoring all typography tags** (`<b>`, `<i>`, etc.). Inspired by OmniDocBench, predicted and gold rows are first semantically aligned as adjacent spans so harmless line splits/merges do not dominate the score.

### Semantic Alignment

1. Header/footer rows are removed.
2. Adjacent predicted and gold row spans are generated (default: up to 3 TSV rows per span).
3. Each candidate span pair is scored by grapheme normalized edit distance (NED): `distance / max(len(pred), len(gold), 1)`.
4. Candidate pairs whose similarity (`1 - NED`) is at least `--alignment-threshold` (default: `0.5`) are selected greedily, disallowing overlapping source rows.
5. Unmatched gold spans count as missing content; unmatched predicted spans count as hallucinated/extra content.

### Metrics

| Metric | Definition | Interpretation |
| --- | --- | --- |
| **TextEdit** | Mean grapheme NED over aligned spans, with unmatched spans scored as `1.0` | OmniDocBench-style headline text score. Lower is better; `0` is perfect. |
| **GCER** | `total_grapheme_edits / total_graphemes_gold` over aligned spans | OCR-D-style grapheme character error rate. Lower is better. |
| **WER** | `total_word_edits / total_words_gold` over aligned spans | Word-level edit rate after semantic alignment. Lower is better. |

### Diagnostics

- **matched_spans**: Predicted/gold adjacent spans successfully matched.
- **missing_spans**: Gold spans with no matched prediction.
- **extra_spans**: Predicted spans with no matched gold content.

---

## 2. Markup / Typography Preservation

Measures whether the model correctly applies bold and italic formatting to the right words.

### Alignment

Since the model may introduce minor character errors (e.g., `аваскн` vs `аваски`), we cannot require exact word matches inside each aligned span. Instead:

1. Text spans are first matched by the semantic alignment pass above.
2. Within each matched span, words are aligned using `SequenceMatcher` on normalised tag-stripped text (see below).
3. Aligned word pairs with character-level similarity below 0.5 are rejected and treated as separate insertions/deletions.

**Typography-only normalisation** (text metrics use tag-stripped semantic span text; this extra punctuation handling is only for word/tag alignment):

| Step | Where | What |
|------|--------|------|
| Line | Before `parse_tagged_words` | NFC, collapse whitespace, remove space before punctuation (`,.:;!?…`) |
| Word key | `SequenceMatcher` + similarity | Strip the same punctuation characters from the word surface for alignment only |
| Tags | TP/FP/FN | Compared on the **original** parsed `(word, tags)` — not on the normalised key |

**Span shape is already equivalent:** `<b>hello world</b>` and `<b>hello</b> <b>world</b>` both yield `(hello, {b})` and `(world, {b})` after parsing; no extra rule is needed.

Example: gold `<b>hello</b>` vs pred `<b>hello.,</b>` → words align as the same slot (punctuation ignored for matching) → **typography TP** if both bold, while GCER may still penalise the extra `.,`.

### Per-tag metrics

For each tag type (`bold` and `italic`), we compute:


| Metric        | Definition            | Interpretation                                                         |
| ------------- | --------------------- | ---------------------------------------------------------------------- |
| **Precision** | `TP / (TP + FP)`      | Of all words the model tagged, how many should have been tagged?       |
| **Recall**    | `TP / (TP + FN)`      | Of all words that should have been tagged, how many did the model tag? |
| **F1**        | `2 * P * R / (P + R)` | Harmonic mean of precision and recall.                                 |

**Typography F1** pools bold and italic counts before computing P/R/F1 (`typography_TP = bold_TP + italic_TP`, etc.). Use this as the single markup headline in comparison CSVs; keep per-tag P/R/F1 in full reports for diagnosis.


Where:

- **TP** (True Positive): Word is tagged in both predicted and gold.
- **FP** (False Positive): Word is tagged in predicted but not in gold.
- **FN** (False Negative): Word is tagged in gold but not in predicted (or the line/word is missing entirely).

Words in missing spans count as false negatives; words in extra spans count as false positives.

---

## 3. Read Order (Structure Preservation)

Measures whether the model reads the page in the correct order — particularly important for multi-column dictionary layouts where a model might incorrectly read across columns instead of down each column sequentially.

### How it works

1. Reuse semantic span alignment from TextEdit.
2. Sort matched gold spans by gold TSV order to form the canonical sequence.
3. Sort matched predicted spans by predicted TSV order, replacing each with its matched gold span ID.
4. Insert unique IDs for unmatched predicted spans.
5. Compute normalized edit distance between the predicted ID sequence and canonical gold ID sequence.

| Metric | Definition | Interpretation |
| --- | --- | --- |
| **ReadOrderEdit** | `levenshtein(pred_gold_id_sequence, gold_id_sequence) / max(len(pred), len(gold), 1)` | OmniDocBench-style order score over aligned text components. Lower is better; `0` means all matched spans appear in canonical order. |

---

## Aggregation

When evaluating multiple pages, metrics are aggregated as follows:


| Component                               | Aggregation method                                                                                 |
| --------------------------------------- | -------------------------------------------------------------------------------------------------- |
| TextEdit                                | **Span-weighted average**: sum per-page TextEdit by aligned span count.                            |
| Character quality (GCER, WER)           | **Micro-average**: sum numerators and denominators across all pages. Longer pages contribute more. |
| Markup quality (P/R/F1)                 | **Micro-average**: sum TP, FP, FN across all pages, then compute P/R/F1.                           |
| ReadOrderEdit                           | **Macro-average**: average per-page scores. Each page contributes equally regardless of length.    |


---

## Usage

```bash
# Single file
uv run dictextractor-eval-s1 \
    -p page_1_stage1.tsv -g page_1_stage1_GOLD.tsv

# Compare every experiment under each language root
uv run dictextractor-eval-s1 \
    --samples-dir assets/dictionaries/samples-2 --all-experiments \
    --alignment-threshold 0.5 --alignment-max-span-rows 3 \
    -o assets/dictionaries/samples-2/stage1_eval

# Restrict to a subset of experiments
uv run dictextractor-eval-s1 \
    --samples-dir assets/dictionaries/samples-2 \
    --experiment-name gemini3flash_alpha_ocr \
    --experiment-name gemini3flash_bare \
    -o assets/dictionaries/samples-2/stage1_eval
```

Batch mode expects the experiment-aware layout: predictions at `<lang>/outputs/stage-1/<experiment>/<stem>/<stem>_stage1.tsv` and gold at `<lang>/outputs/stage-1-gold/<stem>/<stem>_stage1_GOLD.tsv`. Run `scripts/migrate_stage1_layout.sh` once if you have a pre-experiment tree.

**Incremental evaluation:** `<out>/stage1_eval_cache.json` stores per-page metrics until the prediction or gold file changes (mtime + size), or `--alignment-threshold` / `--alignment-max-span-rows` change. Detailed and summary CSVs are regenerated from **all** languages that have paired gold+predictions on disk; `--languages` limits **which pages are recomputed** in this invocation (unless the cache is already invalid). `--overwrite` forces recomputation for that language/experiment selection even if the cache is still valid. Removing an experiment/page from disk prunes its cache entry on the next run. If you change evaluation code meaningfully, bump the cache format in `stage1_eval_cache.py` or delete the cache file.

Output files:

- `<out>/stage1_eval_detailed.csv` — long-format CSV. **Full** (default): `TextEdit`, `GCER`, `WER`, full typography columns, and `ReadOrderEdit`. **Minimal** (`--metrics minimal`): `TextEdit`, `GCER`, `WER`, `typography_f1`, `ReadOrderEdit`. One row per `(experiment, page_id)` plus one `__aggregate__` row per experiment.
- `<out>/stage1_eval_summary.csv` — same column sets, aggregated per `(experiment, language)`, with `page_count`, `alphabet`, and `ocr-hint`.
- `<out>/<experiment>/stage1_evaluation_report.txt` — Human-readable summary
- `<out>/<experiment>/stage1_evaluation_report.json` — Full metrics with raw counts
- `<out>/<experiment>/character_recognition.csv`, `markup_preservation.csv`, `structure_preservation.csv` — Per-component drill-down tables (one row per page + aggregate)

### Comparing experiments

The detailed and summary CSVs are the canonical artifacts for cross-configuration analysis. Each ablation (alphabet on/off, OCR hint on/off, different model / reasoning level) appears as a distinct value in the `experiment` column. Use the summary CSV for per-language leaderboard-style comparisons; use the detailed CSV to pivot by `(experiment, __aggregate__)` for overall quality or filter to a single `page_id` for page-level ablations. The `alphabet` and `ocr-hint` columns mirror `run_config.json` beside the predictions.