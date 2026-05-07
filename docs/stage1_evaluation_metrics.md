# Stage 1 Evaluation Metrics

Stage 1 evaluation measures how well a vision-language model (VLM) transcribes a dictionary page image into structured, tagged text. The evaluation compares a **predicted** TSV (model output) against a **gold** TSV (human-verified ground truth), both sharing the same format: `column_id | line_number | text`.

**Header / footer rows are excluded from every metric.** Stage 1 emits page-level metadata (running titles, page numbers, chapter abbreviations, footnotes) into rows with `column_id="header"` or `column_id="footer"` and empty `line_number`. These rows are dropped before computing character quality, markup quality, and read-order metrics — they are not transcription content, and existing gold TSVs predate the header/footer split. Backfill the golds and re-enable inclusion later if you want explicit metadata accuracy tracking.

We evaluate three independent quality dimensions:

---

## 1. Character Recognition Quality

Measures how accurately the model recognizes individual characters and words, **ignoring all typography tags** (`<b>`, `<i>`, etc.). Both predicted and gold text are tag-stripped and Unicode NFC-normalized before comparison.

Rows are aligned by their `(column_id, line_number)` key. Missing or extra rows in the prediction are counted as full-length errors.

| Metric | Definition | Implementation | Interpretation |
|--------|-----------|----------------|----------------|
| **GCER** (Grapheme Character Error Rate) | `total_grapheme_edits / total_graphemes_gold` | Custom: `grapheme` (UAX #29) + `python-Levenshtein` | Primary OCR quality metric. Operates on user-perceived characters rather than raw code points — critical for scripts like Devanagari, Arabic, and Thai where one visual character spans multiple Unicode code points. Follows the OCR-D standard definition. 0 = perfect. |
| **CER** (Character Error Rate) | `total_char_edits / total_chars_gold` | [`jiwer.process_characters`](https://jitsi.github.io/jiwer) | Traditional OCR metric operating on raw Unicode code points. Kept for comparability with benchmarks using the ISRI/ocreval convention. |
| **WER** (Word Error Rate) | `total_word_edits / total_words_gold` | [`jiwer.process_words`](https://jitsi.github.io/jiwer) | Fraction of gold words that need editing. Whitespace tokenisation, consistent with the ISRI/ocreval word boundary convention. |
| **BLEU** | Corpus-level BLEU-4 | [`sacrebleu`](https://github.com/mjpost/sacrebleu) `tokenize="intl"` | Translation-inspired fluency metric. Uses the Moses v14 international tokeniser, which handles Latin, Cyrillic, Arabic, and CJK scripts without relying on whitespace. Reported as a fraction (0–1). |
| **NED** (Normalized Edit Distance) | `total_grapheme_edits / max(total_graphemes_gold, total_graphemes_pred)` | Custom: same grapheme-level edits as GCER | Symmetric variant of GCER — normalises by the *longer* string, penalizing both extra and missing content equally. |

All five metrics are **micro-averaged** across lines: numerators and denominators are summed across all lines before division, so longer lines contribute proportionally more. The exception is BLEU, which is computed as a single corpus-level score over all lines concatenated.

### Metric choices and reproducibility

- **GCER** aligns with the [OCR-D specification](https://ocr-d.de/en/spec/ocrd_eval.html), which explicitly defines CER at the grapheme-cluster level. No mainstream library exposes this natively, so it is computed with `grapheme` + `python-Levenshtein`.
- **CER / WER** use `jiwer` (RapidFuzz C++ backend) for reproducibility and performance. The default jiwer transformations are bypassed; text is pre-cleaned once by `_clean()` (strip tags → NFC → collapse whitespace) before being passed to jiwer.
- **BLEU** uses `sacrebleu` with `tokenize="intl"` for language-agnostic, reproducible measurement consistent with multilingual MT evaluation papers. The score is divided by 100 to normalize to the [0, 1] range used by the other metrics.

### Additional diagnostics

- **matched_lines**: Lines present in both predicted and gold.
- **missing_lines**: Lines in gold but absent from prediction.
- **extra_lines**: Lines in prediction but absent from gold.

---

## 2. Markup / Typography Preservation

Measures whether the model correctly applies bold and italic formatting to the right words.

### Alignment

Since the model may introduce minor character errors (e.g., `аваскн` vs `аваски`), we cannot require exact word matches for alignment. Instead:

1. Lines are matched by `(column_id, line_number)`.
2. Within each matched line, words are aligned using `SequenceMatcher` on the tag-stripped text.
3. Aligned word pairs with character-level similarity below 0.5 are rejected and treated as separate insertions/deletions.

### Per-tag metrics

For each tag type (`bold` and `italic`), we compute:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Precision** | `TP / (TP + FP)` | Of all words the model tagged, how many should have been tagged? |
| **Recall** | `TP / (TP + FN)` | Of all words that should have been tagged, how many did the model tag? |
| **F1** | `2 * P * R / (P + R)` | Harmonic mean of precision and recall. |

Where:
- **TP** (True Positive): Word is tagged in both predicted and gold.
- **FP** (False Positive): Word is tagged in predicted but not in gold.
- **FN** (False Negative): Word is tagged in gold but not in predicted (or the line/word is missing entirely).

Words on missing lines count as false negatives; words on extra lines count as false positives.

---

## 3. Read Order (Structure Preservation)

Measures whether the model reads the page in the correct order — particularly important for multi-column dictionary layouts where a model might incorrectly read across columns instead of down each column sequentially.

### How it works

1. Strip all HTML tags from both predicted and gold text.
2. Concatenate all lines into a single continuous string in **canonical reading order**: `left` column top-to-bottom, then `center`, then `right` (or `single` for single-column pages).
3. Compute the **Normalized Edit Distance** between the two concatenated strings.

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Read Order NED** | `levenshtein(pred_concat, gold_concat) / max(len(pred), len(gold))` | Overall distance between the two reading-order strings. Conflates character errors with order errors. |
| **Character NED baseline** | The NED from Component 1 | Baseline error attributable purely to character misrecognition. |
| **Isolated Order Error** | `max(Read_Order_NED - Character_NED, 0)` | The portion of Read Order NED attributable to structural/ordering mistakes rather than character errors. |

### Intuition

If the model reads in the correct order but makes some character errors, Read Order NED will be close to Character NED, and the isolated order error will be near zero.

If the model scrambles the reading order (e.g., reads across columns row-by-row, or jumps from line 3 to line 15), large chunks of text will be displaced in the concatenated string, causing the Read Order NED to spike well above the Character NED baseline.

---

## Aggregation

When evaluating multiple pages, metrics are aggregated as follows:

| Component | Aggregation method |
|-----------|--------------------|
| Character quality (GCER, CER, WER, NED) | **Micro-average**: sum numerators and denominators across all pages. Longer pages contribute more. |
| Character quality (BLEU) | **Macro-average**: mean of per-page BLEU scores. Each page contributes equally. |
| Markup quality (P/R/F1) | **Micro-average**: sum TP, FP, FN across all pages, then compute P/R/F1. |
| Read order (NED, isolated error) | **Macro-average**: average per-page scores. Each page contributes equally regardless of length. |

---

## Usage

```bash
# Single file
uv run python -m dictextractor.cli.evaluate_stage1 \
    -p page_1_stage1.tsv -g page_1_stage1_GOLD.tsv

# Batch (auto-discovers all *_stage1.tsv / *_stage1_GOLD.tsv pairs)
uv run python -m dictextractor.cli.evaluate_stage1 \
    --samples-dir assets/dictionaries/samples/

# Custom output directory
uv run python -m dictextractor.cli.evaluate_stage1 \
    --samples-dir assets/dictionaries/samples/ \
    -o results/stage1_eval/
```

Output files:
- `stage1_evaluation_report.txt` — Human-readable summary
- `stage1_evaluation_report.json` — Full metrics with raw counts
- `stage1_evaluation_report.csv` — Flat table (one row per page + aggregate)
