# Evaluation Metrics Overview

The dictionary-extractor pipeline has **two independent evaluation tracks**. Stage 1 measures transcription fidelity; Stage 2 measures structured lexicon output. Scores on one track do not subsume the other.

| Track | Question | Input format | CLI | Detailed doc |
| --- | --- | --- | --- | --- |
| **Stage 1** | What text appears, with what typography? | Flat `.txt` (spec v2) | `dictextractor-eval-flat` | [`stage_1_evaluation_metrics.md`](stage_1_evaluation_metrics.md) |
| **Stage 2** | Which lexicon records were found, with correct markers and order? | Toolbox MDF `.mdf.txt` | `dictextractor-eval-stage2-mdf` | [`stage_2_evaluation_metrics.md`](stage_2_evaluation_metrics.md) |

**Legacy:** `dictextractor-evaluate` supports **entry-level TSV** matching (headword + gloss columns) for the old `--stage2-mode schema` JSON/TSV path. Use the MDF track for direct MDF extraction experiments.

---

## Stage 1 (summary)

Stage 1 flat evaluation compares predicted and gold page transcriptions on two dimensions:

1. **Text recognition** — `TextEdit`, `GCER`, `WER` (page-collapsed by default; line split/merge invariant)
2. **Typography** — bold/italic P/R/F1; pooled headline **Typography F1** (`--metrics minimal`)

Batch wrapper: `bash examples/evaluation/run_stage1_eval_flat.sh`  
Cache format version: **6** (use `--overwrite` after metric code changes).

Full definitions: **`docs/stage_1_evaluation_metrics.md`**.

---

## Stage 2 (summary)

Stage 2 MDF evaluation compares predicted and gold Toolbox MDF on three headline metrics:

1. **Record Accuracy** — fraction of gold lexicon blocks correctly matched
2. **MDF Fields F1** — marker assignment F1 on field lines within matched records
3. **ReadOrderEdit** — OmniDocBench-style edit distance over gold record indices

Gold: `*/outputs/stage-2-gold/<stem>/<stem>.mdf.txt`  
Pred: `*/outputs/stage-2/<experiment>/<stem>/<stem>.mdf.txt`

Default alignment thresholds: record **0.6**, line **0.7**. Marker substitutions (`gn`↔`dn`, `de`↔`ge`): `assets/evaluation/mdf_marker_sub_list.yaml`.

Batch wrapper: `bash examples/evaluation/run_stage2_eval_mdf.sh`

Full definitions: **`docs/stage_2_evaluation_metrics.md`**.

---

## Typical workflow

```text
Page image
    │
    ▼
Stage 1 extraction ──► eval-flat ──► TextEdit, GCER, WER, Typography F1
    │
    ▼
Stage 2 direct MDF ──► eval-stage2-mdf ──► Record Accuracy, MDF Fields F1, ReadOrderEdit
```

Hold Stage 1 fixed when sweeping Stage 2 model or prompt settings. Document both stage-1 and stage-2 experiment names in result tables.

---

## Output locations (defaults)

| Track | Default output dir | Summary CSV |
| --- | --- | --- |
| Stage 1 flat | `evaluations/stage1_flat_eval/` | `stage1_flat_eval_summary.csv` |
| Stage 2 MDF | `evaluations/stage2_mdf_eval/` | `stage2_mdf_eval_summary.csv` |

Each experiment also gets per-page JSON and text reports under `<output_dir>/<experiment>/`.

---

## Related docs

- [`stage_1_methodology.md`](stage_1_methodology.md) — transcription pipeline and flat spec
- [`stage_2_methodology.md`](stage_2_methodology.md) — Pass 1 discovery + Pass 2 direct MDF (and legacy schema mode)
- [`stage_2_outline.md`](stage_2_outline.md) — JSON/MDF field mapping
- [`mdf_field_reference.md`](mdf_field_reference.md) — Toolbox marker reference
