# Dictionary parsing benchmark — design guide

This document captures the evaluation and pipeline plan for a benchmark paper on **parsing scanned dictionary pages into structured digital entries** (headword, POS, semantics, examples, etc.). It reconciles the existing `dictextractor` two-stage pipeline with fair comparison of **specialized document OCR** (MinerU, PaddleOCR-VL, GLM-OCR, …) and **general VLMs** (Gemini, Ovis, …).

---

## 1. What we are measuring (three questions)

| ID | Question | Primary metric |
|----|----------|----------------|
| **A** | Who best **recognizes** page text (characters, typography)? | Flat transcription eval |
| **B** | Who best **structures** dictionary entries? | Stage 2 vs entry gold |
| **C** | Does a **universal transcription layer** help structuring? | Stage 2 ablation (transcript vs none; flat vs column) |

**Paper headline:** **B** (digital dictionary entries).  
**A** is supporting analysis. **C** is a high-value systems section.

Do **not** claim that stage-1 column TSV scores alone equal “best dictionary parser.”

---

## 2. Reframe stage 1: `PageTranscript` (not “the OCR contest”)

Stage 1 in the benchmark is **not** only a VLM transcription step. It is a **canonical interchange layer** that stage 2 always consumes (as the main text hint or equivalent).

### `PageTranscript` (v1 contract)

- **Ordered lines** of page body text (top → bottom in a fixed global reading contract).
- **Typography:** `<b>`, `<i>` where the producer supplies them (VLMs via prompt; parsers via markdown/HTML normalization).
- **Optional metadata:** `producer` (`vlm` | `ocr_adapter`), `source_model`, `spec_version`.
- **Not required:** `column_id` / `line_number` in the benchmark contract (those remain useful for human gold and optional `eval-s1`).

### Two producers (universal stage-1 pipeline)

```text
snippet image
       │
       ├─ General VLM ──► TranscribePrompt_v1 ──────────┐
       │                                                 ├──► PageTranscript
       └─ Specialized OCR ──► parser API ──► OcrAdapter_v1 ─┘
```

- **General VLM:** one frozen transcription prompt + structured output → normalize to `PageTranscript`.
- **Specialized OCR:** not promptable; **parser-specific reader** (thin) → shared **`layout_to_transcript_v1`** (language-agnostic, geometry-only) → `PageTranscript`.

### Stage 2 (fixed for all entrants)

```text
page image + dictionary intro + PageTranscript (+ optional guides)
       → fixed StructureModel / prompts
       → DictionaryEntry list
```

**Specialized OCR path for the paper:** `OCR → PageTranscript → Stage 2` **without** a second VLM transcription pass (unless running an explicit ablation).

---

## 3. Evaluation layers (three leaderboards)

**Unit of evaluation:** always **one snippet image = one page**. Gold, predictions, and metrics are per `page_<n>` (or equivalent stem). Do not concatenate multiple pages into one flat file or one score; aggregate across pages only when reporting corpus-level tables (mean/median over pages).

### Layer 1 — Primary: dictionary structuring

- **Metric:** entry-level vs gold (`dictextractor-evaluate`): headword, POS, translation, examples, extra fields, etc.
- **Protocol:** one structure model, one prompt, one reasoning setting; only swap **PageTranscript** producer.
- **Contenders:** each parser + `OcrAdapter_v1`; each general VLM via `TranscribePrompt_v1`; optional baselines (image-only stage 2, no transcript).

This answers: *who parses scanned dictionaries into digital format best?*

### Layer 2 — Secondary: transcription / OCR fidelity (fair across paradigms)

**New `eval-flat` (to implement):**

- **Gold:** derive `<stem>_stage1_GOLD_flat.txt` **per page** from that page’s column TSV with **flat spec v2** (do not hand-edit eval pages):
  1. `header` rows in file order (one line per row, `text` only).
  2. Body columns in order: `left` → `center` / `middle` → `right` → `single`; within each column sort by `line_number`.
  3. `footer` rows in file order.
  4. Join with `\n` inside the single-page flat file. Typography (`<b>`, `<i>`) preserved.
- **Pred:** same v2 layout for column TSV (flatten) or `FlatTranscriptionResponse` / `OcrAdapter_v1` → `<stem>_stage1_flat.txt`; per-line typography normalization before write.
- **Metrics:** TextEdit, GCER, WER, typography F1 per page (reuse stage-1 metric logic on that page’s flat line list; line-aware alignment — implementer choice, document in spec).
- **Omit or de-emphasize:** ReadOrderEdit on flat track (order is baked into flatten rule).

This answers: *character recognition and typography preservation without column-adapter bias.*

### Layer 3 — Optional: layout-aware transcription (`eval-s1` today)

- **Keep** existing `dictextractor-eval-s1` on `column_id | line_number | text` gold.
- **Label:** “dictionary layout-aware transcription,” not “raw OCR only.”
- **Use for:** VLMs that emit column TSV; parsers after **mechanical** TSV projection from `PageTranscript` (cluster id → `left`/`right`/`single`), not per-language tuning.

---

## 4. Fairness rules (paper + engineering)

### Allowed

- **Frozen `TranscribePrompt_v1`** and **`OcrAdapter_v1`** (versioned in repo).
- **Language-agnostic adapter:** geometry + format only (see §5).
- **Same stage 2** for every method.
- **Symmetric imperfection:** every parser uses the same adapter; every VLM uses the same transcribe prompt.

### Not allowed on the eval set

- Tuning gutter thresholds, line splits, or regexes by watching **eval** `eval-s1` / `eval-flat` scores.
- Per-page hand fixes.
- Oracle reordering pred to match gold read order.
- Different postprocess for model A vs B unless labeled as a separate experiment.

### Allowed once, document clearly

- **Dev-only** refinement → **`adapter_v2`** (new spec version), eval set untouched.
- Per-language **`guides.md`** for stage 2 (symmetric with allowing per-language intro — not adapter tuning).

### Claims to avoid

| Weak claim | Stronger claim |
|------------|----------------|
| “Best OCR model” from TSV `eval-s1` only on parsers | “Best **PageTranscript** under adapter v1” + flat OCR metrics |
| “Stage 1 winner = best dictionary digitization” | “Stage 2 entry metrics (primary)” |
| “No postprocessing” | “No **eval-set-tuned** postprocessing; frozen v1 adapter” |

---

## 5. General OCR adapter (`OcrAdapter_v1`) — no per-language tuning

One shared function after normalizing parser output to blocks:

`{x0, y0, x1, y1, text, category}`

| Step | Rule |
|------|------|
| Filter | Skip or passthrough non-text categories (`figure`, etc.) by category id |
| Columns | Cluster block x-centers → 1–3 columns (gap/histogram; cap at 3; else `single`) |
| Within column | Sort by `y_min`, then `x_min` |
| Serialize | **Column-major** lines: all lines in column 1, then 2, then 3 (global reading contract) |
| Line split | Split block `text` on `\n`; optional tall-block split by height ÷ median line height |
| Typography | `**…**` → `<b>`, `*…*` → `<i>`; passthrough HTML bold/italic |
| Header/footer | Default **omit** (or top/bottom 5% y-band only — no language-specific rules) |

**Thin parser readers (not tunable):**

- MinerU → blocks from `middle_json` / layout JSON  
- PaddleOCR-VL → `parsing_res_list`  
- GLM-OCR → `layout_details` + `md_results` fallback  

**Limitations (state in paper):** LTR assumed; RTL/odd layouts may degrade; tables may be one blob; adapter v1 is geometric, not linguistically tuned.

---

## 6. General VLM transcription (`TranscribePrompt_v1`)

- Task: faithful OCR / transcription only — **no** entry parsing.
- Output: ordered lines with `<b>` / `<i>`; same **column-major reading order** sentence as in adapter v1 (complete left column, then right, etc.).
- Same hyphenation policy as current `STAGE_1_SYSTEM` (preserve printed line breaks; do not merge hyphenated wraps).
- Flatten to `PageTranscript`; do not require column schema in the benchmark contract.

---

## 7. Relationship to OmniDocBench / Real5

| Aspect | OmniDocBench / Real5 | This benchmark |
|--------|----------------------|----------------|
| Gold | JSON layout + order + typed elements | Entry gold + transcription gold (TSV; flat derived) |
| Pred | Page markdown → element extraction + match | `PageTranscript` or TSV |
| Primary score | Mixed text + table TEDS + formula CDM | **Structured dictionary entries** |
| OCR-ish metric | TextEdit after element match | **eval-flat** + optional **eval-s1** |
| Specialized vs VLM | Same md-in / JSON-GT matcher | Same **PageTranscript** schema + same stage 2 |

Real5 inherits OmniDocBench metrics on **degraded images**; we inherit the **idea** of adjacency/hybrid matching in `eval-s1`, but add **dictionary entry** evaluation and a **frozen geometric adapter** for parsers.

---

## 8. Recommended experiments (paper tables)

### Table 1 — Structured entries (primary)

| Method | Transcript source | Stage 1 LLM? | Stage 2 |
|--------|-------------------|----------------|---------|
| MinerU + adapter v1 | OCR | No | Fixed |
| Paddle-VL-1.5 + adapter v1 | OCR | No | Fixed |
| GLM-OCR + adapter v1 | OCR | No | Fixed |
| Gemini / Ovis | VLM transcribe | Yes (or transcript-only arm) | Fixed |
| Ablations | flat vs column transcript; no transcript | — | Fixed |

### Table 2 — Transcription fidelity (secondary)

- All rows: **eval-flat** per page vs that page’s flat gold (same flatten rule); report corpus stats as mean/median over pages.
- Optional columns: typography F1, GCER, TextEdit.

### Table 3 — Layout-aware transcription (optional)

- **eval-s1** on column TSV (VLMs native; parsers via mechanical TSV from line clusters).

### Ablations (short section)

1. Flat vs column `PageTranscript` → same stage 2.  
2. With vs without transcript → stage 2.  
3. Parser transcript vs VLM transcript → stage 2 (hint quality).  

---

## 9. Dataset / repo layout (conventions)

Keep existing layout; extend experiment naming:

```text
<lang>/
  snippets/
  introduction/
  outputs/
    stage-1-gold/<stem>/<stem>_stage1_GOLD.tsv
    stage-1-gold/<stem>/<stem>_stage1_GOLD_flat.txt   # derived, versioned rule
    stage-1/<experiment>/<stem>/<stem>_stage1.tsv      # optional column view
    stage-1/<experiment>/<stem>/page_transcript.txt  # canonical benchmark artifact
    ocr-raw/<backend>/                               # optional: md + layout JSON
    stage-2/<experiment>/...
  ocr_postprocess.yaml                               # NOT per-language tuning for v1 — omit or spec-only constants
```

**Experiment names** should encode lineage, e.g.:

- `s1-flat-gemini` / `s1-tsv-gemini`  
- `s1-ocr-mineru-adapter-v1`  
- `s2-baseline-tsv` / `s2-baseline-flat` (stage 2 slot + stage-1 source in `run_config.json`)

---

## 10. Implementation order (repo)

1. Document **`PageTranscript`** schema (JSON or `.txt` + manifest).  
2. **`flatten_stage1_tsv()`** + generate flat gold; **`eval-flat`** CLI (or `--mode flat` on `eval-s1`).  
3. **`layout_to_transcript_v1()`** + parser readers (MinerU, Paddle, GLM).  
4. Batch CLI: run over many snippets, but write/eval **one** `page_transcript.txt` (and flat/TSV preds) **per page** per backend.  
5. **`TranscribePrompt_v1`** + experiment slot for flat VLM transcripts.  
6. Stage 2 path: consume `page_transcript.txt` without requiring column TSV (prompt update).  
7. Keep **`eval-s1`** unchanged for column gold; optional TSV export from transcript clusters.

---

## 11. What we are **not** doing (v1)

- Multi-page concatenated transcription eval (one flat file or one metric across merged pages).
- Per-language adapter tuning on eval PDFs.  
- Replacing column gold or Label Studio workflow.  
- Requiring specialized OCR models to accept chat prompts.  
- Single leaderboard mixing entry F1 with flat OCR without labeling tracks.  
- Claiming OmniDocBench Overall score equals dictionary parsing quality.

---

## 12. One-paragraph paper framing

We introduce a multilingual **dictionary digitization** benchmark with gold **entries** and gold **transcriptions**, and a two-layer evaluation: (1) **structured parsing** via a fixed stage-2 model fed by a standardized **`PageTranscript`**; (2) **transcription fidelity** via a language-agnostic flat metric comparable across promptable VLMs and non-promptable document OCR engines linked by a frozen geometric **adapter v1**. This separates OCR fidelity from lexicographic structuring and avoids conflating layout reconstruction tuning with character recognition quality.

---

## 13. Related docs in this repo

- `docs/stage1_evaluation_metrics.md` — current column TSV metrics (eval-s1).  
- `docs/architecture.md` — pipeline modules.  
- `CLAUDE.md` — commands and samples-dir layout.  
- `src/dictextractor/extraction/llm_two_stage.py` — current stage 1/2 implementation.  

When implementation diverges from this guide, update this file in the same PR.
