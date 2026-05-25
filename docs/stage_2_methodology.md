# Stage 2 Methodology: Structured Dictionary Parsing

Stage 2 turns a **faithful page transcription** (Stage 1) into a **list of structured dictionary entries** with MDF-aligned typed fields. It is the lexicographic-parsing layer of the two-stage pipeline: Stage 1 answers “what characters appear, in what order?”; Stage 2 answers “which spans are entries, and what is each headword, POS, gloss per target language, example, etc.?”

Evaluation for Stage 2 is measured on the **MDF track** (Record Accuracy, MDF Fields F1, ReadOrderEdit) or the legacy **TSV track** (entry-level headword/gloss matching for `--stage2-mode schema`). Both are separate from Stage 1 transcription metrics (`docs/stage_1_evaluation_metrics.md`).

Field-to-Toolbox mapping: `docs/stage_2_outline.md`. Evaluation overview: `docs/evaluation_metrics.md`.

---

## Stage 2 modes

| Mode | CLI flag | Primary output | Status |
| --- | --- | --- | --- |
| **`direct_mdf`** | `--stage2-mode direct_mdf` (default) | `{stem}.mdf.txt` + per-experiment `field_cheatsheet.json` | **Primary** batch path |
| **`schema`** | `--stage2-mode schema` | `{stem}.json` + `{stem}.tsv` | Legacy structured JSON export |

Unless noted, **§1–§7 below describe `direct_mdf`**. Schema-mode details are kept in §5 (JSON fields) and §8 (legacy TSV eval).

**Direct MDF pipeline:**

```text
Stage-1 transcript (flat or column) + page image + intro images
        │
        ▼
Pass 1 (once per experiment) ──► outputs/stage-2/<experiment>/field_cheatsheet.json
        │                          (markers + structure rules)
        ▼
Pass 2 (per page) ──► Toolbox MDF text ({stem}.mdf.txt)
```

Implementation: Pass 1 `src/dictextractor/llm/field_discovery.py`; Pass 2 `src/dictextractor/llm/stage2_direct_mdf.py`; orchestration `src/dictextractor/extraction/llm_two_stage.py`.

---

## 1. Role in the pipeline

```text
Page image ──┬──► Stage 1 (transcription) ──► flat .txt or column TSV ──┐
Introduction ┴──► (text + optional images) ─────────────────────────────┤
                      dictionary_languages.yaml ──────────────────────────┤
                                                                          ▼
                                                    Stage 2 (structuring)
                                                      direct_mdf (default):
                                                        Pass 1 → field_cheatsheet.json
                                                        Pass 2 → {stem}.mdf.txt
                                                      schema (legacy):
                                                        → entry JSON + review TSV
```

| Stage | Task type | Typical reasoning | Default output (`direct_mdf`) |
| --- | --- | --- | --- |
| **1** | Faithful copy of visible text | `low` | `*_stage1_flat.txt` or `*_stage1.tsv` |
| **2 Pass 1** | Discover MDF markers + entry rules | `high` (batch script) | `field_cheatsheet.json` |
| **2 Pass 2** | Map transcript → Toolbox MDF | `high` (batch script) | `{stem}.mdf.txt` |

**Separation of concerns:** Stage 1 must not “fix” or normalize dictionary content; Stage 2 may apply linguistic judgment to split subentries/senses, join hyphenated line breaks, and assign MDF markers.

Orchestration: `src/dictextractor/extraction/llm_two_stage.py`. Legacy schema prompts: `src/dictextractor/llm/prompts.py` (`STAGE_2_SYSTEM`, `EntriesResponse`).

---

## 2. Inputs

Stage 2 is **multimodal**. For each dictionary page:

### 2.1 Stage-1 transcription (required)

Stage 2 reads one transcript per page from the Stage-1 experiment slot:

| Artifact | Path | Format |
| --- | --- | --- |
| Column (preferred) | `{entry}/outputs/stage-1/<experiment>/<stem>/<stem>_stage1.tsv` | TSV: `column_id`, `line_number`, `text` |
| Flat | `…/<stem>/<stem>_stage1_flat.txt` | One line per row (eval-flat spec) |

**CLI:** `--stage1-input auto|column|flat` (default `auto` = TSV if present, else flat). `--stage1-mode` only controls what Stage 1 **writes** when you run stage 1 or both.

Stage-2-only runs (`--stage 2`) load the resolved file from disk; they do not re-run Stage 1.

Body rows use `column_id` ∈ `{left, center, right, single}` with `line_number` 1…N per column. Page metadata uses `column_id` ∈ `{header, footer}` with empty `line_number`; the Stage 2 system prompt instructs the model to **ignore** these rows.

The `text` column preserves inline markup from Stage 1:

- `<b>…</b>` — typically headwords / entry starts  
- `<i>…</i>` — typically POS, examples, cross-references  

Stage 2 uses these tags as boundary hints, then **strips** them from extracted field values.

**Flat Stage 1** (`--stage1-mode flat`, `*_stage1_flat.txt`) is supported for transcription benchmarks and for Stage 2 when you pass `--stage1-input flat` or `auto` (default: use column TSV if present, else flat). Column TSV is preferred for multi-column trilingual pages because `column_id` is preserved.

### 2.2 Page image (required)

The snippet image for the page is always attached in the user message (first image). The model is told to prioritise the image for character accuracy and entry boundaries when the transcript disagrees.

### 2.3 Dictionary introduction (optional, batch mode)

When `{entry}/introduction/` exists, the batch CLI loads it once per language:

- **Text** (`.txt`, `.md`, `.docx`) → embedded in the Stage 2 user prompt under `<dictionary_introduction>…</dictionary_introduction>`
- **Images / PDFs** → rendered to images and appended after the page image in the user message

Introduction material explains abbreviations, entry layout, POS conventions, and semantic-domain markers. It is **not** re-sent to Stage 1.

Pass `--no-intro` to skip `{entry}/introduction/` for Pass 1 discovery and Pass 2 extraction. Pass `--toolbox-pdf PATH` to attach the SIL Toolbox MDF Reference Manual during **Pass 2 only**. The batch wrapper `examples/stage-2/run_stage2_extraction.sh` runs a full intro × toolbox ablation by default:

| Experiment suffix | Introduction | Toolbox PDF (Pass 2) |
| --- | --- | --- |
| `_intro_notoolbox` | yes | no |
| `_intro_toolbox` | yes | yes |
| `_nointro_notoolbox` | `--no-intro` | no |
| `_nointro_toolbox` | `--no-intro` | yes |

Example: `gemini31pro_high_mdf_intro_toolbox`.

### 2.4 Dictionary language config (batch mode)

When running with `--samples-dir` and per-entry batching, the CLI loads:

- **Path:** `{entry}/dictionary_languages.yaml`
- **Fallback:** If missing, build from folder name + `dictionary_metadata.csv` and write the file (see `load_dictionary_languages` in `src/dictextractor/utils/dictionary_languages.py`).

The config defines:

- **Source language** — headword language (`\lx`), optional `column_id` for column-trilingual layouts  
- **Target languages** — codes for `target_glosses` keys and MDF gloss markers (`ge`, `gn`, `gf`, …)  
- **Layout** — `bilingual`, `inline_trilingual`, or `column_trilingual`

Regenerate all sample YAMLs after metadata or folder renames:

```bash
uv run python scripts/generate_dictionary_languages_yaml.py --overwrite
```

The rendered `<dictionary_languages>` block is appended to the **schema-mode** Stage 2 user prompt. In **`direct_mdf`**, the same config is passed as a **Pass 1 discovery hint** (`languages_config` in `field_discovery.py`).

### 2.5 User-defined guidelines (optional)

`--stage-2-guides <path>` appends a `.txt` / `.md` / `.docx` file verbatim under `USER DEFINED GUIDELINES` at the end of the user prompt. Use for per-language parsing rules without editing `prompts.py`.

### 2.6 What Stage 2 does *not* use

- `alphabet.txt` and `mathpix/` OCR hints (Stage 1 only)  
- Stage 1 raw/input JSON (debug artifacts only)

---

## 3. Model call

| Parameter | CLI flag | Default | Notes |
| --- | --- | --- | --- |
| Stage 2 mode | `--stage2-mode` | `direct_mdf` | `direct_mdf` \| `schema` |
| Structure model | `--model` / `--structure-model` | (required) | Used for Pass 1 + Pass 2 in `direct_mdf`; `--structure-model` overrides `--model` for Stage 2 only |
| Reasoning effort | `--stage2-reasoning` | `low` | `low` \| `medium` \| `high` — `examples/stage-2/run_stage2_extraction.sh` uses `high` |
| One page per entry | `--one-page-per-entry` | off | Stage 2 sweeps: prefer lowest stage-2-gold page, else lowest stage-1 gold snippet, else lowest page number |
| Pass 1 refresh | `--overwrite` | off | Re-run Pass 1 discovery and Pass 2 for this experiment slot |
| Toolbox PDF | `--toolbox-pdf` | — | Optional MDF manual attached in **Pass 2 only** |
| Extra fields | `--discover-extra-fields` | off | **Schema mode only** — populates `extra_fields` from frozen allowlist |
| Stage | `--stage 2` or `both` | — | `both` runs Stage 1 then Stage 2 on the same page |

**`direct_mdf`:** Pass 1 and Pass 2 use `llm.complete` (free-form MDF text / JSON cheat sheet). Pass 1 vocabulary comes from `src/dictextractor/llm/mdf_marker_reference.py` (curated marker list in the discovery system prompt).

**`schema` (legacy):** single Pass 2 call via `llm.complete_structured` with `response_schema=EntriesResponse`.

**Reasoning budget:** Higher reasoning helps on dense pages but can **leak chain-of-thought into output** on some models. Validate before large sweeps.

Stage 1 and Stage 2 can use **different models** via `--model` + `--structure-model`.

---

## 4. Prompt design

### 4.1 Direct MDF — Pass 1 (field discovery)

System prompt embeds the curated **`MDF_MARKER_REFERENCE`** (`src/dictextractor/llm/mdf_marker_reference.py`) plus instructions to output a JSON cheat sheet: markers used on this dictionary, one-line descriptions, and structure rules.

User message includes:

1. Introduction page images  
2. Sample page image  
3. Sample page **Stage-1 transcript** (gold flat or column, depending on `--stage1-input`)  
4. Optional hint from `dictionary_languages.yaml` (source/target languages, layout)

Output is cached as `{entry}/outputs/stage-2/<experiment>/field_cheatsheet.json`. Pass 1 does **not** attach the Toolbox PDF unless you add that separately in custom tooling.

### 4.2 Direct MDF — Pass 2 (page extraction)

System prompt (`DIRECT_MDF_SYSTEM` in `stage2_direct_mdf.py`): copy transcript characters verbatim; use image/intro for boundaries and marker assignment only; emit blank-line-delimited MDF.

User message includes:

1. `<transcription>` — Stage-1 gold or pred transcript  
2. Page image  
3. Introduction images  
4. **Field map block** — rendered from `field_cheatsheet.json` (`format_prompt_block()`)  
5. Optional `--toolbox-pdf` (Pass 2 only)  
6. Optional `--stage-2-guides`

`dictionary_languages.yaml` informs Pass 1 discovery; **Pass 2 marker assignment follows the cheat sheet**, not the YAML directly.

### 4.3 Schema mode (legacy)

#### System prompt (`STAGE_2_SYSTEM` + `STAGE_2_MDF_BLOCK`)

Fixed across runs. Defines JSON field contract, record boundaries, `target_glosses`, examples, hyphen rejoining, etc.

#### User prompt (`stage_2_user`)

Built per page: `<dictionary_languages>`, intro text, `<transcription>`, optional `<extra_fields_discovery>`, optional `--stage-2-guides`.

---

## 5. Output schema

### 5.1 Canonical fields (`DictionaryEntry`)

| JSON field | TSV column(s) | Description |
| --- | --- | --- |
| `entry_type` | `Entry_Type` | `main`, `subentry`, or `sense` |
| `headword` | `Headword` | Lemma with diacritics; no POS embedded |
| `parent_lexeme` | `Parent_Lexeme` | Parent `\lx` for subentries/senses |
| `sense_number` | `Sense_Number` | Sense index when `entry_type` is `sense` |
| `homonym_number` | `Homonym_Number` | Homograph marker when printed |
| `pos` | `POS` | Part of speech as printed |
| `target_glosses` | `Gloss_<code>` | Per-target short glosses (e.g. `Gloss_en`, `Gloss_zh`) |
| `definition` | `Definition` | Longer `\de` text |
| `semantic_domain` | `Semantic_Domain` | Short label only when explicitly marked |
| `citation_form` | `Citation_Form` | Citation form when ≠ headword |
| `phonetic` | `Phonetic` | Pronunciation when marked |
| `cross_references` | `Cross_References` | Target lemmas (` \| `-joined in TSV) |
| `examples` | `Examples` | Usage citations (` \| `-joined) |
| `example_glosses` | `Example_Glosses` | Parallel translations |
| `extra_fields` | dynamic | Discovery mode only (§5.3) |

**Gloss composition:** Near-synonyms for one target → `; ` within that `target_glosses[code]`. Minor sub-meanings of the same sense in prose → `definition` with ` | `. Distinct senses the source treats as separate → separate rows (`entry_type` `sense` or new `main`), not merged into one gloss string.

**Trilingual:** Do not merge English and other targets into one string. Use separate keys per `dictionary_languages.yaml`. Column-trilingual dictionaries align gloss lines to the configured `column_id` per target.

### 5.2 MDF serialisation (current vs planned)

| Today | Planned |
| --- | --- |
| JSON array + review TSV with MDF-oriented columns | `json_to_mdf()` → Toolbox `.txt` with `\lx`, `\ge`, `\de`, … |
| Multiple JSON rows per visual block (`main` + `sense` / `subentry`) | Exporter groups rows into one Toolbox record |
| `run_config.json` flag `stage2_output_format: "mdf"` | Same + exporter version / marker map |

See `docs/stage_2_outline.md` for the full JSON → MDF table and grouping notes.

### 5.3 Extra-field discovery (`--discover-extra-fields`)

**Default: off.** When disabled, the prompt requires `extra_fields: {}` on every entry.

When enabled, the model may add **only** keys from the **built-in allowlist** (hard-coded in `src/dictextractor/llm/prompts.py` — no per-dictionary file yet). Full table with TSV column names and MDF hints: **`docs/mdf_field_reference.md` § Default `extra_fields` allowlist**.

| JSON key | TSV column |
| --- | --- |
| `etymology` | `Etymology` |
| `plural_form` | `Plural_Form` |
| `gender` | `Gender` |
| `noun_class` | `Noun_Class` |
| `tone_class` | `Tone_Class` |
| `register` | `Register` |
| `dialect` | `Dialect` |
| `usage_note` | `Usage_Note` |
| `inflection` | `Inflection` |
| `literal_meaning` | `Literal_Meaning` |
| `variant_form` | `Variant_Form` |
| `antonym` | `Antonym` |

Rules:

- Only for content the dictionary **visibly marks** with a dedicated convention.  
- Do **not** put IPA, pronunciation, or “see also” prose in `extra_fields` — use `phonetic` and `cross_references`.  
- Do **not** duplicate `target_glosses`, `definition`, `pos`, or `semantic_domain`.

Discovered keys become additional TSV columns via `json_to_tsv` (only columns for keys present on that page).

### 5.4 On-disk artifacts (per page)

#### Direct MDF (default)

Under `{entry}/outputs/stage-2/<stage2-experiment>/<stem>/`:

| File | Purpose |
| --- | --- |
| `<stem>.mdf.txt` | Toolbox MDF output |
| `<stem>_stage2_raw.txt` | Raw LLM response (Pass 2) |
| `<stem>_stage2_input.json` | Sanitised messages (images as placeholders) |
| `<stem>_usage.json` | Token/cost: `field_discovery`, `stage2`, optional `stage1` |
| `<stem>_gold_compare.json` | Optional dev compare when gold MDF exists under `stage-2-gold/` |

Per experiment (once):

| File | Purpose |
| --- | --- |
| `{entry}/outputs/stage-2/<experiment>/field_cheatsheet.json` | Pass 1 marker cheat sheet (cached per experiment; refresh with `--overwrite`) |

Gold MDF for evaluation: `{entry}/outputs/stage-2-gold/<stem>/<stem>.mdf.txt`.

#### Schema mode (legacy)

Under `{entry}/outputs/stage-2/<stage2-experiment>/<stem>/`:

| File | Purpose |
| --- | --- |
| `<stem>.json` | Raw structured entries array |
| `<stem>.tsv` | Canonical + `Gloss_*` + optional extra columns |
| `<stem>_stage2_raw.json` | API raw response |
| `<stem>_stage2_input.json` | Sanitised messages |
| `<stem>_usage.json` | Token/cost summary |

Experiment-level `run_config.json` records model, reasoning, `discover_extra_fields`, `stage2_output_format`, `dictionary_languages` (YAML snapshot), intro paths, `stage1_source`, per-page Stage-1 TSV paths, stage-2 guides, and git SHA. On resume, the existing manifest is preserved unless `--overwrite` is passed.

**Important:** With `--stage 1` only, nothing is written under `stage-2/` (usage for Stage 1 alone is stored next to Stage-1 outputs).

---

## 6. Experiment slots and lineage

Stage 1 and Stage 2 use **independent experiment namespaces**:

```text
outputs/
  stage-1/<stage1-experiment>/<stem>/<stem>_stage1_flat.txt   (or *_stage1.tsv)
  stage-1-gold/<stem>/<stem>_stage1_GOLD_flat.txt             (eval / Pass 2 input)
  stage-2-gold/<stem>/<stem>.mdf.txt                          (MDF gold)
  stage-2/<stage2-experiment>/field_cheatsheet.json            (Pass 1 cache)
  stage-2/<stage2-experiment>/<stem>/<stem>.mdf.txt           (direct_mdf pred)
  stage-2/<stage2-experiment>/<stem>/<stem>.json|.tsv         (schema pred)
```

| Flag | Effect |
| --- | --- |
| `--experiment-name` | Stage-1 output slot **and** the Stage-1 TSVs Stage 2 reads |
| `--stage2-experiment-name` | Stage-2 output slot only (defaults to `--experiment-name`) |

**Typical sweep:** Fix Stage 1 once (`gemini3flash_alpha_ocr`), then run multiple Stage-2 configs:

```bash
bash examples/stage-2/run_stage2_extraction.sh
# intro ablation: RUN_INTRO=0 or RUN_NOINTRO=0 to run one arm only
# or:
uv run dictextractor-extract \
  --strategy two_stage --stage 2 \
  --stage2-mode direct_mdf \
  --samples-dir assets/dictionaries/samples \
  --languages Chukchi-Russian \
  --model gemini/gemini-3.1-pro-preview \
  --stage1-input flat \
  --stage2-experiment-name gemini31pro_high_mdf_intro_notoolbox \
  --stage2-reasoning high
```

Optional: `DISCOVER_EXTRA=1` or `--discover-extra-fields` on **schema mode** runs.

Resume behaviour (`direct_mdf`): skip a page if `<stem>.mdf.txt` exists unless `--overwrite`. Stage-2-only skips pages with no Stage-1 transcript at the expected path (`--stage1-input` resolves gold under `stage-1-gold/` in batch mode).

---

## 7. Parsing procedure (logical steps)

### Direct MDF (default)

**Pass 1 (once per dictionary):**

1. Ingest intro + sample page images and transcript.  
2. Select MDF markers from `MDF_MARKER_REFERENCE` that appear on the sample page.  
3. Write structure rules (homographs, senses, subentries, gloss-line conventions).  
4. Cache as `outputs/stage-2/<experiment>/field_cheatsheet.json`.

**Pass 2 (per page):**

1. Load cached cheat sheet for this experiment (or run Pass 1 if missing; `--overwrite` forces refresh).  
2. Copy vernacular and gloss **characters verbatim** from the Stage-1 transcript.  
3. Use page image + intro for entry boundaries and marker roles.  
4. Emit blank-line-delimited Toolbox MDF (`\marker value` lines).  
5. Apply structural normalisations only (strip `<b>`/`<i>`, rejoin hyphens, normalise `\sn`/`\hm` digits).

### Schema mode (legacy)

For each page, the model is instructed to:

1. **Ingest conventions** from introduction text/images (if provided).  
2. **Apply language roles** from `<dictionary_languages>`.  
3. **Scan the transcript** for reading order and markup cues.  
4. **Segment** into entry blocks; classify with `entry_type`.  
5. **Fill** canonical JSON fields; emit `EntriesResponse`.  

Post-processing: `save_to_json` → `json_to_tsv`.

---

## 8. Evaluation methodology

Stage 2 quality is measured on two tracks:

| Track | Format | CLI | Doc |
| --- | --- | --- | --- |
| **MDF (primary)** | `.mdf.txt` | `dictextractor-eval-stage2-mdf` | [`stage_2_evaluation_metrics.md`](stage_2_evaluation_metrics.md) |
| **TSV (legacy)** | entry TSV | `dictextractor-evaluate` | §8.1–8.3 below |

The MDF track evaluates record detection, marker assignment, and read order on blank-line-delimited Toolbox output. Use it for **`direct_mdf`** experiments and gold under `outputs/stage-2-gold/`.

### 8.1 MDF evaluation (recommended)

```bash
bash examples/evaluation/run_stage2_eval_mdf.sh
```

Default thresholds: record **0.6**, line **0.7**. Marker substitutions: `assets/evaluation/mdf_marker_sub_list.yaml`.

Metrics: **Record Accuracy**, **MDF Fields F1**, **ReadOrderEdit** (OmniDocBench-style gold record indices). Full definitions: **`docs/stage_2_evaluation_metrics.md`**.

### 8.2 Legacy TSV matching

`DictionaryEvaluator` loads extracted and gold TSVs, then greedily pairs rows:

1. Headword similarity must exceed **0.7** (normalised `SequenceMatcher`).  
2. Combined score = **50%** headword + **15%** POS + **35%** translation/gloss column.  
3. Pairs above `--threshold` (default **0.85**) count as matches.

Gold/eval TSVs in older Label Studio exports may use legacy headers (`Headword_Phrase`, `Translation_RU`, `Grammar_Notes`); the evaluator normalises via those column names. New MDF-shaped exports use `Headword`, `Gloss_*` or `Definition` — adapters may be needed for strict comparison until evaluators are updated.

### 8.3 Legacy TSV metrics (`EvaluationMetrics`)

| Metric | Meaning |
| --- | --- |
| **Precision / Recall / F1** | Entry detection vs gold row count |
| **Exact / partial matches** | Above-threshold pairs |
| **Missing / extra entries** | Unmatched gold / predicted rows |
| **Headword / grammatical / definition accuracy** | Mean field similarity over matched pairs |

Optional **character-level error analysis** (`DetailedErrorAnalyzer`) breaks down edit patterns on matched headwords and glosses.

### 8.4 Legacy TSV CLI

```bash
uv run dictextractor-evaluate \
  -e path/to/page_42.tsv \
  -g path/to/gold_entries.tsv \
  -o results/ \
  -t 0.85
```

Corpus-level benchmarking (many pages, many languages) should aggregate per-page F1 with the same threshold and document which `stage2-experiment` and `stage1_source` produced the predictions.

### 8.5 What Stage 2 evaluation does *not* measure

- Character-level OCR quality (Stage 1 / `eval-flat`)  
- Typography tag preservation (Stage 1 markup metrics)  
- Layout reconstruction from non-VLM OCR backends  

Those belong on the **transcription track**. MDF validity beyond `\marker value` line syntax is covered indirectly by **MDF Fields F1**, not a separate Toolbox linter.

---

## 9. Design rationale

| Choice | Rationale |
| --- | --- |
| **Two-pass direct MDF** | Pass 1 locks marker vocabulary per dictionary; Pass 2 focuses on transcription-faithful digitization. |
| **Transcript + image** | Transcript is authoritative for characters; image resolves boundaries and field roles. |
| **`field_cheatsheet.json`** | Per-experiment marker map (under `outputs/stage-2/<experiment>/`) without hand-authoring prompts for every language. |
| **`dictionary_languages.yaml`** | Source/target roles and layout for Pass 1 discovery and schema-mode export. |
| **Separate experiment slots** | Stage-1 ablations fixed while sweeping Stage-2 model, reasoning, or guides. |
| **Intro only in Stage 2** | Alphabet priming is recognition; abbreviation keys are parsing. |
| **Schema mode retained** | Legacy JSON/TSV path and `dictextractor-evaluate` TSV matching. |

---

## 10. Limitations and planned extensions

**Current limitations:**

- Column-trilingual pages are harder from flat transcripts (no `column_id` in the file).  
- Schema mode still has no automatic `json_to_mdf()` exporter — JSON/TSV only there.  
- Pass 1 marker vocabulary is prompt-guided, not code-validated against `MDF_MARKER_REFERENCE`.  
- High reasoning effort may contaminate string fields on some models — validate before large sweeps.  
- Default `--discover-extra-fields` is off — rare dictionary-specific markers omitted unless schema mode + flag.

**Planned:**

- `json_to_mdf()` for schema mode — group `main` / `subentry` / `sense` rows into Toolbox records.  
- Richer Pass 1 validation and per-dictionary marker policy files.  
- Fixed Stage-2 model + frozen transcript for fair comparison across Stage-1 backends.

When implementation changes, update this document and `docs/stage_2_outline.md` in the same PR.

---

## 11. Related documentation

| Document | Topic |
| --- | --- |
| `docs/stage_2_outline.md` | JSON → MDF mapping, direct MDF outputs, TSV columns |
| `docs/stage_2_evaluation_metrics.md` | Record Accuracy, MDF Fields F1, ReadOrderEdit |
| `docs/evaluation_metrics.md` | Overview of Stage 1 + Stage 2 eval tracks |
| `docs/mdf_field_reference.md` | Full MDF marker list + Pass 1 reference |
| `docs/stage_1_evaluation_metrics.md` | Transcription fidelity (Stage 1) |
| `README.md` | CLI reference and quick start |
| `PLAN.md` | Benchmark tracks, ablations, dataset layout |
| `examples/stage-2/run_stage2_extraction.sh` | Direct MDF batch command |
| `examples/evaluation/run_stage2_eval_mdf.sh` | MDF evaluation batch |
| `src/dictextractor/llm/field_discovery.py` | Pass 1 discovery |
| `src/dictextractor/llm/stage2_direct_mdf.py` | Pass 2 direct MDF |
| `src/dictextractor/llm/mdf_marker_reference.py` | Curated marker vocabulary for Pass 1 |
| `assets/evaluation/mdf_marker_sub_list.yaml` | Eval-time marker substitution groups |
