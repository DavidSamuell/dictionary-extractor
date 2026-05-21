# Stage 2 output — MDF / Toolbox mapping

Stage 2 produces per-page JSON (`<stem>.json`) aligned with SIL Multi-Dictionary Formatter (MDF) field markers for downstream `json_to_mdf()` export (planned).

## JSON field → MDF marker

| JSON field | MDF | Notes |
|------------|-----|--------|
| `entry_type` | record role | `main` → new `\lx`; `subentry` → `\se`; `sense` → `\sn` |
| `headword` | `\lx` | Lemma only; no POS in this field |
| `parent_lexeme` | groups `\se`/`\sn` | Required when `entry_type` is `subentry` or `sense` |
| `sense_number` | `\sn` | e.g. `1`, `2`, `I` |
| `homonym_number` | `\hm` | When dictionary marks homographs |
| `pos` | `\ps` | As printed (`n.`, `сущ.`, …) |
| `gloss` | `\ge` | Short target-language gloss |
| `definition` | `\de` | Longer definitional text |
| `meaning_description` | (legacy) | Leave empty; TSV falls back if `definition` empty |
| `semantic_domain` | `\sd` | Short label only |
| `citation_form` | `\lc` | When printed form ≠ `headword` |
| `phonetic` | `\ph` | Pronunciation if marked |
| `cross_references` | `\cf` | List of target lemmas |
| `examples` | `\xv` | One list element per example |
| `example_glosses` | `\xe` | Parallel translations |
| `extra_fields` | custom | Non-standard markers only (allowlist in prompt) |

## Hierarchy examples

### Main entry

```json
{
  "entry_type": "main",
  "headword": "ac-úkwʌn",
  "parent_lexeme": "",
  "sense_number": "",
  "gloss": "кремень",
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

### Subentry

```json
{
  "entry_type": "subentry",
  "headword": "ac-ékwəŋ",
  "parent_lexeme": "ac-úkwʌn",
  "gloss": "дробить",
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
  "gloss": "наклонно, покато"
}
```

```text
\lx arkyčety
\sn 1
\ge наклонно, покато
```

## TSV columns

`json_to_tsv()` writes the canonical columns listed in `utils/io.py` (`CANONICAL_TSV_FIELDS`) for spreadsheet review before MDF export.

## Manifest flag

Stage-2 `run_config.json` includes `"stage2_output_format": "mdf"` to record that prompts expect this schema.
