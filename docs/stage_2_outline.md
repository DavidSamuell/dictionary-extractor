# Stage 2 output — MDF / Toolbox mapping

Stage 2 has two output paths:

| Mode | CLI | Primary output |
| --- | --- | --- |
| **`direct_mdf`** (default) | `--stage2-mode direct_mdf` | `{stem}.mdf.txt` (Toolbox MDF text) |
| **`schema`** (legacy) | `--stage2-mode schema` | `{stem}.json` + `{stem}.tsv` |

**Direct MDF (primary):** Pass 1 discovers markers → `outputs/stage-2/<experiment>/field_cheatsheet.json`; Pass 2 writes blank-line-delimited `\marker value` records. Gold for evaluation: `outputs/stage-2-gold/<stem>/<stem>.mdf.txt`. See `docs/stage_2_methodology.md`.

**Schema mode:** per-page JSON aligned with `DictionaryEntry`, exported to review TSV. A dedicated `json_to_mdf()` exporter is **planned** for this path; direct MDF bypasses JSON entirely.

## Design goals

| Goal | Mechanism |
| --- | --- |
| One schema across all sample dictionaries | Fixed `DictionaryEntry` canonical fields |
| Bilingual vs trilingual glosses | `target_glosses` keyed by language code; roles from `dictionary_languages.yaml` |
| Run-ons, senses, homonyms | `entry_type` + `parent_lexeme` + `sense_number` / `homonym_number` |
| Rare per-dictionary markers | Optional `extra_fields` (`--discover-extra-fields`; 12-key built-in allowlist) |

## Per-dictionary language config

Each sample folder may contain `dictionary_languages.yaml` (generated from folder name + `assets/dictionaries/full dictionaries/dictionary_metadata.csv`):

```bash
uv run python scripts/generate_dictionary_languages_yaml.py --overwrite
```

| `layout` | When | Stage 2 behaviour |
| --- | --- | --- |
| `bilingual` | One target language | Single `target_glosses` key (e.g. `en`) → MDF `\ge` |
| `inline_trilingual` | Multiple targets in one entry block (e.g. Na–English–Chinese–French) | Split glosses into separate MDF markers per target (`\ge`, `\gn`, `\gr`, …) |
| `column_trilingual` | Targets in separate columns (e.g. Circassian–English–Turkish) | Align glosses to `column_id`; headword from source column |

Target language **roles** (English vs national vs regional) come from `dictionary_languages.yaml` and inform Pass 1 discovery. In **direct MDF**, actual `\marker` codes are assigned from **`field_cheatsheet.json`**, not from the YAML directly.

Implementation: `src/dictextractor/schemas/dictionary_languages.py`, `src/dictextractor/utils/dictionary_languages.py`.

## JSON field → MDF marker

| JSON field | MDF | Notes |
| --- | --- | --- |
| `entry_type` | record role | `main` → new `\lx`; `subentry` → `\se`; `sense` → `\sn` |
| `headword` | `\lx` | Lemma only; no POS in this field |
| `parent_lexeme` | groups `\se`/`\sn` | Required when `entry_type` is `subentry` or `sense` |
| `sense_number` | `\sn` | e.g. `1`, `2`, `I` |
| `homonym_number` | `\hm` | When dictionary marks homographs |
| `pos` | `\ps` | As printed (`n.`, `сущ.`, …) |
| `target_glosses` | `\ge`, `\gn`, `\gf`, … | Map: language code → short gloss; marker per `dictionary_languages.yaml` |
| `gloss` | (legacy) | Always leave `""`; use `target_glosses` only |
| `definition` | `\de` | Longer definitional text |
| `semantic_domain` | `\sd` | Short label only |
| `citation_form` | `\lc` | When printed form ≠ `headword` |
| `phonetic` | `\ph` | Pronunciation if marked |
| `cross_references` | `\cf` | List of target lemmas |
| `examples` | `\xv` | One list element per example |
| `example_glosses` | `\xe` | Parallel translations (same length as `examples` when bilingual) |
| `extra_fields` | custom | Discovery only; **12 built-in keys** — `docs/mdf_field_reference.md` § Default allowlist |

## Hierarchy examples

### Main entry (bilingual)

```json
{
  "entry_type": "main",
  "headword": "ac-úkwʌn",
  "parent_lexeme": "",
  "sense_number": "",
  "target_glosses": {"ru": "кремень"},
  "gloss": "",
  "definition": "букв. жирный камень",
  "pos": "сущ."
}
```

Toolbox (conceptual):

```text
\lx ac-úkwʌn
\ps сущ.
\ge кремень
\de букв. жирный камень
```

### Inline trilingual

```json
{
  "entry_type": "main",
  "headword": "example_lemma",
  "target_glosses": {
    "en": "English gloss",
    "zh": "中文释义",
    "fr": "glossaire français"
  },
  "gloss": "",
  "definition": ""
}
```

```text
\lx example_lemma
\ge English gloss
\gn 中文释义
\gr glossaire français
```

### Subentry

```json
{
  "entry_type": "subentry",
  "headword": "ac-ékwəŋ",
  "parent_lexeme": "ac-úkwʌn",
  "target_glosses": {"ru": "дробить"},
  "gloss": "",
  "pos": "гл."
}
```

```text
\lx ac-úkwʌn
...
\se ac-ékwəŋ
\ps гл.
\ge дробить
```

### Numbered sense

```json
{
  "entry_type": "sense",
  "headword": "arkyčety",
  "parent_lexeme": "arkyčety",
  "sense_number": "1",
  "target_glosses": {"en": "наклонно, покато"},
  "gloss": ""
}
```

```text
\lx arkyčety
\sn 1
\ge наклонно, покато
```

## TSV columns

`json_to_tsv()` (`src/dictextractor/utils/io.py`) writes:

1. **Canonical columns** — `Entry_Type`, `Headword`, `Parent_Lexeme`, `Sense_Number`, `Homonym_Number`, `POS`, `Definition`, `Semantic_Domain`, `Citation_Form`, `Phonetic`, `Cross_References`, `Examples`, `Example_Glosses` (lists joined with ` | `).
2. **Per-target gloss columns** — `Gloss_<code>` for each key seen in `target_glosses` on the page (e.g. `Gloss_en`, `Gloss_zh`, `Gloss_ru`). Legacy rows with only `gloss` and no map use `Gloss_legacy` when no `target_glosses` keys exist.
3. **Discovery columns** — one column per `extra_fields` key (snake_case → `Title_Case` header), when present.

There is no single `Gloss` column in the canonical set; glosses live in `Gloss_*` columns.

## Export to Toolbox (planned)

A future `json_to_mdf()` step must:

1. **Group rows** by lemma block — merge `main`, then attach `subentry` / `sense` rows sharing `parent_lexeme` (or same headword for senses).
2. **Emit markers** — map `target_glosses[code]` through `dictionary_languages.yaml` `mdf_marker` per target.
3. **Map `extra_fields`** — assign custom MDF markers per project or dictionary.

Until that exporter exists, JSON + TSV are sufficient for human review and for building the exporter.

## Manifest

Stage-2 `run_config.json` includes (when batch mode loads config):

- `"stage2_output_format"` — `"mdf"` for `direct_mdf`, `"schema"` for legacy JSON/TSV
- `"dictionary_languages"` — snapshot of the loaded YAML
- `"stage1_source"` — which Stage-1 experiment supplied transcripts

See `docs/stage_2_methodology.md` for the full pipeline and CLI flags.  
For the complete MDF marker inventory, see `docs/mdf_field_reference.md`.
