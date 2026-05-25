# Stage 2 Evaluation Metrics

**Pipeline and MDF output:** see `docs/stage_2_methodology.md` and `docs/stage_2_outline.md`.  
**Overview of both evaluation tracks:** see `docs/evaluation_metrics.md`.

Stage 2 MDF evaluation measures how well a model extracts **Toolbox MDF** from a dictionary page. We compare **predicted** `.mdf.txt` against **gold** `.mdf.txt` on three headline metrics:

1. **Record Accuracy** — fraction of gold lexicon blocks correctly matched
2. **MDF Fields F1** — F1 over `\marker` assignment on field lines within matched records
3. **ReadOrderEdit** — whether matched records appear in the correct sequence

**CLI:** `dictextractor-eval-stage2-mdf`  
**Format:** blank-line-delimited Toolbox MDF (`\marker value` lines)

| Path | Location |
| --- | --- |
| Gold | `*/outputs/stage-2-gold/<stem>/<stem>.mdf.txt` |
| Pred | `*/outputs/stage-2/<experiment>/<stem>/<stem>.mdf.txt` |

---

## Record and line definitions

### Record

One **record** is a blank-line-delimited lexicon block — a headword entry plus homographs, subentries, and attached field lines, exactly as Toolbox would store one lexical unit.

Records are parsed in file order with 0-based indices. The headword is taken from the first `\lx` line in the block (if present).

### Field line

One **field line** is a single `\marker value` row inside a record (e.g. `\ge translation`, `\ps n.`).

### Fingerprint (record matching)

For record alignment, each block is reduced to a **value-only fingerprint**:

1. Take every field value in block order.
2. Apply `normalize_field_value` (see below).
3. Join with spaces.

**Markers are excluded** from the fingerprint. Record matching therefore depends on lexical content, not on whether the model chose `\ge` vs `\gn`.

---

## Shared value normalisation

Applied before fingerprinting and line alignment:

| Step | What |
| --- | --- |
| NFC | Unicode normal form |
| Typography strip | Remove `<b>`, `</b>`, `<i>`, `</i>` |
| Whitespace | Collapse runs to a single space, strip ends |
| Project text cleanup | `normalize_text` (homoglyph / punctuation normalisation shared with Stage 1) |

Similarity between two normalised strings uses `difflib.SequenceMatcher` ratio in `[0, 1]`.

---

## Alignment

Both record and line alignment use the same **greedy one-to-one** procedure:

1. Enumerate all candidate pairs whose similarity ≥ threshold.
2. Sort candidates by similarity (descending), then gold index, then pred index.
3. Greedily accept pairs without reusing either side.
4. Unmatched gold indices → **false negatives**; unmatched pred indices → **false positives**.

### 1. Record-level alignment

| Setting | Default | Meaning |
| --- | --- | --- |
| `--record-threshold` | `0.6` | Minimum fingerprint similarity to pair two records |

Each accepted pair counts as one **record true positive**. Unmatched gold records are **record false negatives**; unmatched predicted records are **record false positives**.

### 2. Line-level alignment (within matched records)

For each matched record pair, align field lines by **normalised value only** (markers ignored for pairing):

| Setting | Default | Meaning |
| --- | --- | --- |
| `--line-threshold` | `0.7` | Minimum value similarity to pair two lines |

After pairing, markers are scored on matched lines only.

### 3. Marker substitution list

Some dictionaries use interchangeable gloss markers (e.g. Russian `\gn` vs English `\ge`). The evaluator treats markers in the same **equivalence group** as correct.

Default file: `assets/evaluation/mdf_marker_sub_list.yaml`

```yaml
equivalence_groups:
  - [gn, dn]
  - [de, ge]
```

Override with `--marker-sub-list <path>`. If the file is missing or empty, the same built-in groups apply.

**Scoring on matched lines:**

| Outcome | Marker counts |
| --- | --- |
| Markers equal or in the same group | **TP** +1 |
| Markers differ (not substitutable) | **FP** +1, **FN** +1 |
| Gold line unmatched | **FN** +1 |
| Pred line unmatched | **FP** +1 |

---

## Metrics

### 1. Record Accuracy

**Record Accuracy** = `TP / (TP + FN)` = matched gold records / all gold records.

This answers: *“Out of every entry that should be on the page, how many did we get right?”* Extra predicted blocks that do not match gold (`FP`) do not affect this score.

JSON reports also include `record_counts` (`tp`, `fp`, `fn`) for diagnostics.

### 2. MDF Fields F1

**MDF Fields F1** is micro-averaged F1 over field lines within matched records (after line alignment). Precision and recall are computed internally from marker TP/FP/FN counts but are not reported as headline metrics.

Wrong markers, missing lines, and extra lines all reduce MDF Fields F1. Use the JSON report’s **marker confusion** table to see systematic `\ps` ↔ `\ge` swaps. `marker_counts` (`tp`, `fp`, `fn`) is included in JSON for drill-down.

### 3. Read order (structure preservation)

OmniDocBench-style **ReadOrderEdit** over **gold record indices**:

1. Take all record pairs matched at the record-alignment step.
2. Sort pairs by predicted record index (reading order in the pred file).
3. `gt = [0, 1, …, n−1]` — every gold record index (including unmatched).
4. `pred` — matched gold indices in predicted reading order.
5. Unmatched predicted records are omitted from `pred`.

| Metric | Definition | Interpretation |
| --- | --- | --- |
| **ReadOrderEdit** | `levenshtein(gt, pred) / max(len(gt), len(pred), 1)` | Lower is better; `0` = perfect order among detected records. |

Missing records hurt read order because their gold indices stay in `gt` but never appear in `pred`. Swapping two detected entries also increases edit distance.

---

## Aggregation

When multiple pages are evaluated together:

| Component | Aggregation |
| --- | --- |
| Record Accuracy | **Micro-average** (`sum(TP) / sum(TP + FN)` across pages) |
| MDF Fields F1 | **Micro-average** (pooled marker TP/FP/FN, then F1) |
| ReadOrderEdit | **Macro-average** (mean of per-page scores) |

The summary CSV includes a `__aggregate__` row per experiment when more than one page is evaluated.

---

## Usage

```bash
# Single page
uv run dictextractor-eval-stage2-mdf \
    -p path/to/page.mdf.txt \
    -g path/to/page.mdf.txt

# Batch (benchmark wrapper)
bash examples/evaluation/run_stage2_eval_mdf.sh

# Batch with custom marker substitution list
uv run dictextractor-eval-stage2-mdf \
    --samples-dir assets/dictionaries/samples \
    --experiment-name gemini31pro_high_mdf_intro_notoolbox \
    --experiment-name gemini31pro_high_mdf_intro_toolbox \
    --languages Chukchi-Russian Evenki-Russian \
    --marker-sub-list assets/evaluation/mdf_marker_sub_list.yaml \
    --record-threshold 0.6 \
    --line-threshold 0.7 \
    -o evaluations/stage2_mdf_eval
```

**Experiment selection:**

| Flag | What gets evaluated |
| --- | --- |
| `--experiment-name NAME` (repeatable) | Named folders under `outputs/stage-2/` only |
| `--all-experiments` | Every experiment folder that has matching gold/pred pairs |
| `--languages LANG …` | Restrict to named language directories |

**Threshold flags:**

- `--record-threshold` (default `0.6`)
- `--line-threshold` (default `0.7`)
- `--marker-sub-list` (default: `assets/evaluation/mdf_marker_sub_list.yaml`)

---

## Outputs

Default directory: `evaluations/stage2_mdf_eval/`

| File | Contents |
| --- | --- |
| `stage2_mdf_eval_summary.csv` | Per (experiment, page): Record Accuracy, MDF Fields F1, ReadOrderEdit |
| `<experiment>/stage2_mdf_evaluation_report.json` | Full metrics, confusion matrix, diagnostic samples |
| `<experiment>/stage2_mdf_evaluation_report.txt` | Human-readable per-page summary |

**Summary CSV columns:**

`experiment`, `page_id`, `Record_Accuracy`, `MDF_Fields_F1`, `ReadOrderEdit`

**JSON diagnostics** (per page):

- `record_samples` — matched pairs with similarity &lt; 0.95 (content drift)
- `missing_record_samples` / `extra_record_samples` — unmatched blocks
- `marker_error_samples` — wrong marker on aligned lines
- `marker_confusion` — gold marker → predicted marker counts

---

## Comparing experiments

Use the summary CSV for cross-configuration tables. Each ablation is a distinct `experiment` folder under `outputs/stage-2/`.

**Fair comparison checklist:**

1. Same gold MDF (`stage-2-gold`).
2. Same Stage 1 transcript input (document `stage1_source` in `run_config.json`).
3. Same `--record-threshold`, `--line-threshold`, and `--marker-sub-list`.
4. Report Record Accuracy and MDF Fields F1 together — high record accuracy with low MDF Fields F1 usually means entries were found but field markers are wrong.

**What Stage 2 MDF evaluation does not measure:**

- Character-level OCR quality → Stage 1 `eval-flat`
- Inline bold/italic preservation → Stage 1 typography metrics
- Validity of Toolbox export syntax beyond `\marker value` lines
- Entry-level TSV column matching → legacy `dictextractor-evaluate`

---

## Implementation reference

| Component | Module |
| --- | --- |
| MDF parser | `evaluation/stage2/mdf_parser.py` |
| Record/line alignment | `evaluation/stage2/mdf_align.py` |
| Marker substitution | `evaluation/stage2/mdf_marker_equiv.py` |
| Metrics + reports | `evaluation/stage2/mdf_evaluator.py` |
| CLI | `cli/evaluate_stage2_mdf.py` |
| Default sub list | `assets/evaluation/mdf_marker_sub_list.yaml` |
