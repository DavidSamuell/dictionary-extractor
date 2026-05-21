# Stage 2 Methodology: Structured Dictionary Parsing

Stage 2 turns a **faithful page transcription** (Stage 1) into a **list of structured dictionary entries** with MDF-aligned typed fields. It is the lexicographic-parsing layer of the two-stage pipeline: Stage 1 answers “what characters appear, in what order?”; Stage 2 answers “which spans are entries, and what is each headword, POS, gloss per target language, example, etc.?”

Evaluation for Stage 2 is **entry-level** (precision, recall, F1 against gold entry TSVs). That is intentionally separate from Stage 1 transcription metrics (`docs/stage1_evaluation_metrics.md`), so OCR fidelity and structuring quality can be reported on independent tracks.

Field-to-Toolbox mapping: `docs/stage_2_outline.md`.

---

## 1. Role in the pipeline

```text
Page image ──┬──► Stage 1 (transcription) ──► column TSV  ──┐
             │                                              ├──► Stage 2 (structuring) ──► entry JSON + TSV
Introduction ┴──► (text + optional images) ────────────────┘
                      dictionary_languages.yaml ─────────────► (per entry folder, batch mode)
```

| Stage | Task type | Typical reasoning | Structured output schema |
| --- | --- | --- | --- |
| **1** | Faithful copy of visible text | `low` | `TranscriptionResponse` → column TSV |
| **2** | Interpret layout + map to MDF-shaped entries | `low`–`high` | `EntriesResponse` → `DictionaryEntry` list |

**Separation of concerns:** Stage 1 must not “fix” or normalize dictionary content; Stage 2 may use linguistic judgment to split subentries/senses, join hyphenated line breaks, and strip markup tags from field values.

Implementation: `src/dictextractor/extraction/llm_two_stage.py` (`TwoStageLLMExtraction._stage2_structure`). Prompts: `src/dictextractor/llm/prompts.py` (`STAGE_2_SYSTEM`, `STAGE_2_MDF_BLOCK`, `stage_2_user`).

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

The rendered `<dictionary_languages>` block is appended to the Stage 2 **user** prompt so the model knows which gloss keys to fill and how to read multi-column pages.

### 2.5 User-defined guidelines (optional)

`--stage-2-guides <path>` appends a `.txt` / `.md` / `.docx` file verbatim under `USER DEFINED GUIDELINES` at the end of the user prompt. Use for per-language parsing rules without editing `prompts.py`.

### 2.6 What Stage 2 does *not* use

- `alphabet.txt` and `mathpix/` OCR hints (Stage 1 only)  
- Stage 1 raw/input JSON (debug artifacts only)

---

## 3. Model call

| Parameter | CLI flag | Default | Notes |
| --- | --- | --- | --- |
| Structure model | `--model` / `--structure-model` | (required) | `--structure-model` overrides `--model` for Stage 2 only |
| Reasoning effort | `--stage2-reasoning` | `low` | `low` \| `medium` \| `high` — example batch script uses `high` |
| Extra fields | `--discover-extra-fields` | off | Populates `extra_fields` from frozen allowlist only |
| Stage | `--stage 2` or `both` | — | `both` runs Stage 1 then Stage 2 on the same page |

**API:** `llm.complete_structured` with `response_schema=EntriesResponse`, enforcing valid JSON matching the Pydantic schema (no heuristic JSON repair).

**Reasoning budget:** Structuring requires layout understanding, abbreviation decoding, and subentry/sense splitting. Higher reasoning can help on dense pages but has been observed to **leak chain-of-thought into JSON string fields** on some models. Default CLI is `low`; `examples/stage-2/run_stage2_extraction.sh` sets `--stage2-reasoning high` — validate outputs before large sweeps.

Stage 1 and Stage 2 can use **different models** (e.g. Flash for transcription, Pro for structuring) via `--model` + `--structure-model`.

---

## 4. Prompt design

### 4.1 System prompt (`STAGE_2_SYSTEM` + `STAGE_2_MDF_BLOCK`)

Fixed across runs. It defines:

1. **Inputs** — column TSV (or flat text if supported later), page image, optional intro images  
2. **MDF export contract** — explicit `entry_type` (`main` / `subentry` / `sense`), field hygiene, no reasoning in JSON values  
3. **Record boundaries** — decision tree for senses vs run-on subentries vs new headwords  
4. **Target glosses** — use `target_glosses` map; leave legacy `gloss` empty  
5. **Examples** — `examples` and `example_glosses` as parallel lists  
6. **Phonetic / cross-refs** — `phonetic` and `cross_references`, not `extra_fields`  
7. **Hyphen rejoining** — Stage 1 line-break hyphens joined in field values  
8. **Phonetic fidelity** — preserve special characters (ŋ, ə, ь, …)

Schema field descriptions in `DictionaryEntry` match this contract.

### 4.2 User prompt (`stage_2_user`)

Built per page from dynamic blocks (order):

1. `<dictionary_languages>` — from `DictionaryLanguagesConfig.format_prompt_block()`, when batch config is loaded  
2. `<dictionary_introduction>` — intro text, if any  
3. `<transcription>` — full Stage-1 TSV body  
4. `<extra_fields_discovery>` — only when `--discover-extra-fields` is set (see §5.3)  
5. Closing instructions — parse entries; first image = page, further images = intro pages  
6. `USER DEFINED GUIDELINES` — optional file from `--stage-2-guides`

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

Under `{entry}/outputs/stage-2/<stage2-experiment>/<stem>/`:

| File | Purpose |
| --- | --- |
| `<stem>.json` | Raw structured entries array |
| `<stem>.tsv` | Canonical + `Gloss_*` + optional extra columns |
| `<stem>_stage2_raw.json` | API raw response |
| `<stem>_stage2_input.json` | Sanitised messages (images replaced with placeholders) |
| `<stem>_usage.json` | Token/cost summary for Stage 1+2 when both ran on the page |

Experiment-level `run_config.json` records model, reasoning, `discover_extra_fields`, `stage2_output_format`, `dictionary_languages` (YAML snapshot), intro paths, `stage1_source`, per-page Stage-1 TSV paths, stage-2 guides, and git SHA. On resume, the existing manifest is preserved unless `--overwrite` is passed.

**Important:** With `--stage 1` only, nothing is written under `stage-2/` (usage for Stage 1 alone is stored next to Stage-1 outputs).

---

## 6. Experiment slots and lineage

Stage 1 and Stage 2 use **independent experiment namespaces**:

```text
outputs/
  stage-1/<stage1-experiment>/<stem>/<stem>_stage1.tsv
  stage-2/<stage2-experiment>/<stem>/<stem>.json
  stage-2/<stage2-experiment>/<stem>/<stem>.tsv
```

| Flag | Effect |
| --- | --- |
| `--experiment-name` | Stage-1 output slot **and** the Stage-1 TSVs Stage 2 reads |
| `--stage2-experiment-name` | Stage-2 output slot only (defaults to `--experiment-name`) |

**Typical sweep:** Fix Stage 1 once (`gemini3flash_alpha_ocr`), then run multiple Stage-2 configs:

```bash
export UV_CACHE_DIR="${HOME}/.cache/uv"   # if project quota is tight

bash examples/stage-2/run_stage2_extraction.sh
# or:
uv run dictextractor-extract \
  --strategy two_stage --stage 2 \
  --samples-dir assets/dictionaries/samples \
  --languages Chepang-English \
  --model gemini/gemini-3.1-pro-preview \
  --experiment-name gemini3flash_alpha_ocr \
  --stage2-experiment-name pro_highreasoning_mdf \
  --stage2-reasoning high
```

Optional: `DISCOVER_EXTRA=1` or `--discover-extra-fields` on the same command.

Resume behaviour: if `<stem>.tsv` already exists under the stage-2 experiment slot, the page is skipped unless `--overwrite` is set. Stage-2-only skips pages with no Stage-1 TSV at the expected path.

---

## 7. Parsing procedure (logical steps)

For each page, the model is instructed to:

1. **Ingest conventions** from introduction text/images (if provided).  
2. **Apply language roles** from `<dictionary_languages>` (layout, gloss keys, column alignment).  
3. **Scan the TSV** for reading order (`column_id`, `line_number`) and markup cues (`<b>`, `<i>`).  
4. **Ignore** `header` / `footer` rows.  
5. **Segment** the body into entry blocks using bold transitions and visual layout on the image.  
6. **Classify** each row with `entry_type` and link subentries/senses via `parent_lexeme` / `sense_number`.  
7. **Fill** only fields evidenced in the source; strip markup from values.  
8. **Populate** `target_glosses` per configured language codes; leave `gloss` empty.  
9. **Rejoin** hyphenated line-break artifacts from Stage 1 inside field strings.  
10. **Emit** JSON matching `EntriesResponse` (no commentary in field values).

Post-processing in the CLI: `save_to_json` → `json_to_tsv` for the final TSV.

---

## 8. Evaluation methodology

Stage 2 quality is measured with **entry-level matching** (`src/dictextractor/evaluation/stage2/`).

### 8.1 Matching

`DictionaryEvaluator` loads extracted and gold TSVs, then greedily pairs rows:

1. Headword similarity must exceed **0.7** (normalised `SequenceMatcher`).  
2. Combined score = **50%** headword + **15%** POS + **35%** translation/gloss column.  
3. Pairs above `--threshold` (default **0.85**) count as matches.

Gold/eval TSVs in older Label Studio exports may use legacy headers (`Headword_Phrase`, `Translation_RU`, `Grammar_Notes`); the evaluator normalises via those column names. New MDF-shaped exports use `Headword`, `Gloss_*` or `Definition` — adapters may be needed for strict comparison until evaluators are updated.

### 8.2 Metrics (`EvaluationMetrics`)

| Metric | Meaning |
| --- | --- |
| **Precision / Recall / F1** | Entry detection vs gold row count |
| **Exact / partial matches** | Above-threshold pairs |
| **Missing / extra entries** | Unmatched gold / predicted rows |
| **Headword / grammatical / definition accuracy** | Mean field similarity over matched pairs |

Optional **character-level error analysis** (`DetailedErrorAnalyzer`) breaks down edit patterns on matched headwords and glosses.

### 8.3 CLI

```bash
uv run dictextractor-evaluate \
  -e path/to/page_42.tsv \
  -g path/to/gold_entries.tsv \
  -o results/ \
  -t 0.85
```

Corpus-level benchmarking (many pages, many languages) should aggregate per-page F1 with the same threshold and document which `stage2-experiment` and `stage1_source` produced the predictions.

### 8.4 What Stage 2 evaluation does *not* measure

- Character-level OCR quality (Stage 1 / `eval-s1` / `eval-flat`)  
- Typography tag preservation (Stage 1 markup metrics)  
- Layout reconstruction from non-VLM OCR backends  
- Toolbox MDF file validity (until `json_to_mdf()` exists)

Those belong on the **transcription track** or a future **export track**.

---

## 9. Design rationale

| Choice | Rationale |
| --- | --- |
| **Transcript + image** | TSV gives stable reading order; image resolves ambiguous characters and column boundaries. |
| **MDF-shaped schema** | One interchange model for Toolbox export and cross-dictionary comparison. |
| **`dictionary_languages.yaml`** | Per-dictionary gloss keys and trilingual layout without per-row ISO tags. |
| **`target_glosses` map** | Supports bilingual and multi-target MDF markers (`\ge`, `\gn`, …). |
| **Structured output API** | Eliminates brittle JSON parsing; schema enforces list-shaped examples and typed hierarchy. |
| **Separate experiment slots** | Stage-1 ablations can be held fixed while sweeping Stage-2 model, reasoning, intro, or discovery mode. |
| **Intro only in Stage 2** | Alphabet priming is a recognition concern; abbreviation keys are a parsing concern. |
| **Conservative `semantic_domain`** | Reduces hallucinated domain labels; empty string when uncertain. |
| **Optional `extra_fields`** | Keeps canonical columns stable; discovery is opt-in with a frozen allowlist. |

---

## 10. Limitations and planned extensions

**Current limitations:**

- Column-trilingual pages are harder from flat transcripts (no `column_id` in the file).  
- No automatic `json_to_mdf()` / Toolbox file writer — JSON/TSV only.  
- No automatic consumption of third-party OCR transcripts in the Stage 2 prompt (planned: transcript-only ablations in `PLAN.md`).  
- Entry evaluator column names still reflect some legacy gold formats; MDF TSVs may need adapter columns for F1 runs.  
- High reasoning effort may contaminate string fields on some models — validate before large sweeps.  
- Default `--discover-extra-fields` is off — rare dictionary-specific markers are omitted unless enabled.

**Planned:**

- `json_to_mdf()` grouping `main` / `subentry` / `sense` rows into Toolbox records.  
- Feed a canonical flat `page_transcript.txt` into Stage 2 without requiring column TSV.  
- Stage-2 evaluator alignment with `Gloss_*` and `Entry_Type` columns.  
- Fixed Stage-2 model + frozen transcript for fair comparison across Stage-1 backends.

When implementation changes, update this document and `docs/stage_2_outline.md` in the same PR.

---

## 11. Related documentation

| Document | Topic |
| --- | --- |
| `docs/stage_2_outline.md` | JSON → MDF mapping, TSV columns, hierarchy examples |
| `docs/mdf_field_reference.md` | Full MDF marker list (Appendix A) + allowlist mapping |
| `docs/stage1_evaluation_metrics.md` | Transcription fidelity (Stage 1) |
| `docs/architecture.md` | Module map and batch layout |
| `PLAN.md` | Benchmark tracks, ablations, dataset layout |
| `examples/stage-2/run_stage2_extraction.sh` | Example Stage-2 batch command |
| `scripts/generate_dictionary_languages_yaml.py` | Regenerate per-sample language YAML |
| `src/dictextractor/llm/prompts.py` | Authoritative prompt text |
| `src/dictextractor/schemas/entry.py` | `DictionaryEntry` schema |
| `src/dictextractor/schemas/dictionary_languages.py` | Language config schema |
