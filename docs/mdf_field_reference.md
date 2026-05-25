# MDF / SIL Toolbox field reference

Reference for **Multi-Dictionary Formatter (MDF)** data-field markers used in SIL Toolbox / Shoebox lexicons. MDF defines on the order of **100 markers**; most projects use **20–30** regularly ([SIL MDF overview](https://software.sil.org/shoebox/mdf/)).

This document supports dictionary-extractor Stage 2 design: which markers map to canonical `DictionaryEntry` fields, which belong in optional `extra_fields` discovery, and how a **user-extended allowlist** could work later.

**Primary source:** Coward & Grimes, *Making dictionaries: a guide to lexicography and MDF* (2000), **Appendix A** — alphabetized list of field markers ([PDF](https://s3.amazonaws.com/downloads.sil.org/legacy/shoebox/MDF_2000.pdf)).  
**See also:** [LingTranSoft MDF code reference](https://lingtran.net/MDF-Code-reference), `docs/stage_2_outline.md`.

---

## How MDF relates to dictionary-extractor

| Layer | What it is |
| --- | --- |
| **Pass 1 marker reference** | Curated subset in `src/dictextractor/llm/mdf_marker_reference.py` — vocabulary for field discovery (not a hard allowlist). |
| **`field_cheatsheet.json`** | Pass 1 output: markers + rules cached under `outputs/stage-2/<experiment>/` for **direct MDF Pass 2**. |
| **Canonical `DictionaryEntry` fields** | Stable JSON/TSV columns (`schema` mode): `\lx`, `\ge`, `\de`, `\ps`, … |
| **`target_glosses`** | Per-target gloss keys in schema mode; MDF markers from `dictionary_languages.yaml` at export time. |
| **`extra_fields` (discovery)** | Optional snake_case keys when `--discover-extra-fields` is on (**schema mode**). **12 built-in keys only** — see § Default allowlist below. |
| **Eval substitution** | `assets/evaluation/mdf_marker_sub_list.yaml` — equivalent markers for scoring (e.g. `gn`↔`dn`). |
| **Planned `json_to_mdf()`** | Groups schema-mode JSON rows into Toolbox text (direct MDF already emits MDF). |

MDF is **not** a single fixed schema: projects may define **custom markers**. Appendix A is the standard MDF set; FieldWorks and other tools may accept additional SFM markers.

---

## Alphabetized MDF markers (Appendix A)

Markers are written `\xx` in Toolbox files. “English label” is what MDF may print before the field value in formatted output (varies by template).

| Marker | Function | English label (typical) |
| --- | --- | --- |
| `\1d` | first person dual inflection | (inflection label) |
| `\1e` | first person plural exclusive | (inflection label) |
| `\1i` | first person plural inclusive | (inflection label) |
| `\1p` | first person plural | (inflection label) |
| `\1s` | first person singular | (inflection label) |
| `\2d` | second person dual inflection | (inflection label) |
| `\2p` | second person plural | (inflection label) |
| `\2s` | second person singular | (inflection label) |
| `\3d` | third person dual inflection | (inflection label) |
| `\3p` | third person plural | (inflection label) |
| `\3s` | third person singular | (inflection label) |
| `\4d` | non-human/non-animate dual | (inflection label) |
| `\4p` | non-human/non-animate plural | (inflection label) |
| `\4s` | non-human/non-animate singular | (inflection label) |
| `\an` | antonym | Ant: |
| `\bb` | bibliographical reference for further reading | Read: |
| `\bw` | borrowed word (loan) | From: |
| `\ce` | cross-reference gloss (English) | (see entry) |
| `\cf` | cross-reference | See: |
| `\cn` | cross-reference gloss (national language) | Lihatlah: |
| `\cr` | cross-reference gloss (regional language) | (regional) |
| `\de` | definition/explication (English) | (often none) |
| `\dn` | definition/explication (national language) | (national) |
| `\dr` | definition/explication (regional language) | (regional) |
| `\dt` | date (entry last worked on) | (date) |
| `\dv` | definition/explication (vernacular) | (vernacular def.) |
| `\ec` | etymology comment | Etym: |
| `\ee` | encyclopedic information (English) | (encyclopedic) |
| `\eg` | etymology gloss | (etymology gloss) |
| `\en` | encyclopedic info. (national language) | (national) |
| `\er` | encyclopedic info. (regional language) | (regional) |
| `\es` | etymology source | (source) |
| `\et` | etymology (proto form) | Etym: / Asal: |
| `\ev` | encyclopedic info. (vernacular) | (vernacular) |
| `\ge` | gloss (English) | — |
| `\gn` | gloss (national language) | — |
| `\gr` | gloss (regional language) | — |
| `\gv` | gloss (vernacular) | — |
| `\hm` | homonym / homophone / homograph number | (subscript) |
| `\is` | index of semantics | Semantics: |
| `\lc` | citation form (lexical citation) | — |
| `\le` | gloss of `\lf` (English) | — |
| `\lf` | lexical functions | (various, see §7) |
| `\ln` | gloss of `\lf` (national language) | — |
| `\lr` | gloss of `\lf` (regional language) | — |
| `\lt` | literally | Lit: ‘…’ |
| `\lx` | lexeme (headword / lemma) | — |
| `\mn` | main entry form | See main entry: |
| `\mr` | morphology | Morph: |
| `\na` | notes (anthropology) | [Anth: …] |
| `\nd` | notes (discourse) | [Disc: …] |
| `\ng` | notes (grammar) | [Gram: …] |
| `\np` | notes (phonology) | [Phon: …] |
| `\nq` | notes (questions for investigation) | [Ques: …] |
| `\ns` | notes (sociolinguistics) | [Socio: …] |
| `\nt` | notes (general) | [Note: …] |
| `\oe` | only / restrictions (English) | Restrict: |
| `\on` | only / restrictions (national language) | (national) |
| `\or` | only / restrictions (regional language) | (regional) |
| `\ov` | only / restrictions (vernacular) | VerRestrict: |
| `\pc` | picture or graphic link | (…) |
| `\pd` | paradigm | Prdm: |
| `\ph` | phonetic form (pronunciation) | [phonetic brackets] |
| `\pl` | plural form | Pl: |
| `\pn` | part of speech (national language) | (national POS) |
| `\ps` | part of speech | (POS) |
| `\rd` | reduplication form(s) | Redup: |
| `\re` | reversal (English) | re: |
| `\rf` | reference to written source (text or notebook) | Ref: |
| `\rn` | reversal (national language) | rn: |
| `\rr` | reversal (regional language) | rr: |
| `\sc` | scientific name | (Latin name) |
| `\sd` | semantic domain | SD: / Golongan: |
| `\se` | subentry | — |
| `\sg` | singular form | Sg: / Tunggal: |
| `\sn` | sense number | (sense index) |
| `\so` | source | [Source: …] / [Dari: …] |
| `\st` | status (editing or printing) | [Status: …] |
| `\sy` | synonym | Syn: / Searti: |
| `\tb` | table (chart) | — |
| `\th` | thesaurus | Thes: / Keluarga: |
| `\ue` | usage (English) | Usage: |
| `\un` | usage (national language) | Kegunaan: |
| `\ur` | usage (regional language) | (regional) |
| `\uv` | usage (vernacular) | VerUsage: |
| `\va` | variant forms | Variant: / Bentuk lain: |
| `\ve` | variant (English gloss or comment) | (…) |
| `\vn` | variant (national language) | (…) |
| `\vr` | variant (regional language) | (…) |
| `\we` | word-level gloss (English) | — |
| `\wn` | word-level gloss (national language) | — |
| `\wr` | word-level gloss (regional language) | — |
| `\xe` | example (English free translation) | — |
| `\xg` | example (gloss for interlinearizing) | — |
| `\xn` | example (national lang. free translation) | — |
| `\xr` | example (regional lang. free translation) | — |
| `\xv` | example (vernacular) | — |

**Note:** Person/number inflection markers `\1s` … `\4s` are listed in Appendix A but marked ***not supported by MDF*** for printing in the same way as core lexical fields; they appear in paradigms / interlinear workflows.

**Lexical functions (`\lf`):** MDF uses `\lf` with function codes (e.g. SynD, SynR, SynT, SynL) for dialectal/register/taboo/loan synonyms; glosses may use `\le`, `\ln`, `\lr`. See Appendix D in the same PDF.

**Language-specific gloss/definition families:** Many markers exist in **English / national / regional / vernacular** sets (`\ge`/`\gn`/`\gr`/`\gv`, `\de`/`\dn`/`\dr`/`\dv`, `\xe`/`\xn`/`\xr`, etc.). dictionary-extractor maps targets via `dictionary_languages.yaml` instead of hard-coding `\ge` only.

---

## Markers by role (quick navigation)

### Entry structure (required for export grouping)

| Marker | Role |
| --- | --- |
| `\lx` | Main lexeme — **obligatory** record boundary |
| `\hm` | Homonym number |
| `\lc` | Citation form (printed headword when different from `\lx`) |
| `\se` | Subentry under `\lx` |
| `\sn` | Sense number |
| `\mn` | “See main entry” pointer |

### Glosses and definitions

| Marker | Role |
| --- | --- |
| `\ge`, `\gn`, `\gr`, `\gv` | Short glosses (by language tier) |
| `\de`, `\dn`, `\dr`, `\dv` | Longer definitions |
| `\lt` | Literal meaning |
| `\we`, `\wn`, `\wr` | Word-level glosses |

### Grammar and paradigm

| Marker | Role |
| --- | --- |
| `\ps`, `\pn` | Part of speech |
| `\pl`, `\sg` | Plural / singular |
| `\pd` | Paradigm |
| `\rd` | Reduplication |
| `\mr` | Morphology block |
| `\1s`–`\4s`, etc. | Inflection slots (paradigm; limited MDF print support) |

### Examples

| Marker | Role |
| --- | --- |
| `\xv` | Vernacular example |
| `\xe`, `\xn`, `\xr` | Example translations |
| `\xg` | Interlinear gloss line |

### Semantics and relations

| Marker | Role |
| --- | --- |
| `\sd` | Semantic domain |
| `\is` | Semantic index |
| `\th` | Thesaurus |
| `\an` | Antonym |
| `\sy` | Synonym |
| `\lf` + `\le`/`\ln`/`\lr` | Lexical functions |
| `\cf`, `\ce`, `\cn`, `\cr` | Cross-references |

### Etymology and encyclopedic

| Marker | Role |
| --- | --- |
| `\et`, `\eg`, `\es`, `\ec`, `\ev` | Etymology (form, gloss, source, comment) |
| `\ee`, `\en`, `\er` | Encyclopedic notes |
| `\bw` | Borrowed word |

### Variants, usage, restrictions

| Marker | Role |
| --- | --- |
| `\va`, `\ve`, `\vn`, `\vr` | Variants |
| `\ue`, `\un`, `\ur`, `\uv` | Usage notes |
| `\oe`, `\on`, `\or`, `\ov` | Restrictions / “only” |

### Editorial / meta (often not dictionary body text)

| Marker | Role |
| --- | --- |
| `\na`, `\nd`, `\ng`, `\np`, `\nq`, `\ns`, `\nt` | Compiler notes by domain |
| `\dt` | Last-edited date |
| `\st` | Editorial status |
| `\so`, `\rf` | Source / reference |
| `\bb` | Bibliography pointer |
| `\pc` | Picture link |
| `\tb` | Table |
| `\re`, `\rn`, `\rr` | Reversal (finderlist; not main entry body) |
| `\sc` | Scientific name (Latin) |

---

## Mapping to dictionary-extractor (today)

| MDF marker(s) | `DictionaryEntry` / config | Notes |
| --- | --- | --- |
| `\lx` | `headword` (`entry_type=main`) | Lemma only |
| `\se` | `entry_type=subentry`, `parent_lexeme` | Run-on / derivative block |
| `\sn` | `entry_type=sense`, `sense_number`, `parent_lexeme` | Numbered sense |
| `\hm` | `homonym_number` | |
| `\lc` | `citation_form` | |
| `\ps`, `\pn` | `pos` | Prefer printed POS in source language |
| `\ge`, `\gn`, `\gf`, … | `target_glosses[code]` | Codes from `dictionary_languages.yaml` |
| `\de`, `\dn`, … | `definition` | Longer text; multi-language via targets if needed later |
| `\sd` | `semantic_domain` | Short label only |
| `\ph` | `phonetic` | |
| `\cf` (+ gloss variants) | `cross_references` | Target lemmas, not “see” prose |
| `\xv` | `examples[]` | |
| `\xe`, `\xn`, `\xr` | `example_glosses[]` | Parallel to `\xv` |
| Most other Appendix A markers | `extra_fields` (if discovery on) | Only when allowlisted and visibly marked |

## Default `extra_fields` allowlist (built-in, no user file)

This is the **only** set of `extra_fields` keys Stage 2 may use today. It is compiled into the repo — **not** loaded from per-dictionary YAML yet.

| Item | Value |
| --- | --- |
| **Enable** | CLI `--discover-extra-fields`, or `DISCOVER_EXTRA=1` in `examples/stage-2/run_stage2_extraction.sh` |
| **Default** | Off — every entry must have `extra_fields: {}` |
| **Source code** | `EXTRA_FIELDS_ALLOWLIST` in `src/dictextractor/llm/prompts.py` |
| **Prompt block** | `<extra_fields_discovery>` in `stage_2_user()` when discovery is on |
| **TSV columns** | One column per key **used on that page** — snake_case → Title_Case (e.g. `usage_note` → `Usage_Note`) |

### Built-in keys (12)

| # | JSON key (`extra_fields`) | TSV column | Typical MDF marker | Use when |
| --- | --- | --- | --- | --- |
| 1 | `etymology` | `Etymology` | `\et`, `\eg`, `\ec` | Etymology line or labeled proto form |
| 2 | `plural_form` | `Plural_Form` | `\pl` | Plural printed for the entry |
| 3 | `gender` | `Gender` | (project-specific) | Gender/class label — **not** MDF `\ng` (grammar notes) |
| 4 | `noun_class` | `Noun_Class` | (project-specific) | Noun class (e.g. Bantu classes) |
| 5 | `tone_class` | `Tone_Class` | (project-specific) | Tone pattern when marked |
| 6 | `register` | `Register` | `\ue`, `\lf` SynR | Register label (formal, colloq., …) |
| 7 | `dialect` | `Dialect` | `\lf` SynD, `\va` | Dialectal variant label |
| 8 | `usage_note` | `Usage_Note` | `\ue`, `\un`, `\uv` | Usage / grammar note beyond POS |
| 9 | `inflection` | `Inflection` | `\pd`, paradigm slots | Inflection table or paradigm hint |
| 10 | `literal_meaning` | `Literal_Meaning` | `\lt` | “Lit.” / literal gloss |
| 11 | `variant_form` | `Variant_Form` | `\va`, `\ve` | Orthographic or variant form |
| 12 | `antonym` | `Antonym` | `\an` | Antonym when explicitly marked |

**Not on this list** (use canonical fields instead): IPA/pronunciation → `phonetic`; see also → `cross_references`; glosses → `target_glosses`; long text → `definition`; POS → `pos`; domain → `semantic_domain`.

**Extraction rules** (same as prompt):

- Populate a key only when the dictionary **visibly marks** that information (abbreviation, label, fixed position).  
- Multiple values for one key on one entry → join with `"; "`.  
- If nothing applies → `extra_fields: {}` (no extra TSV columns on that page).

**Caution:** MDF marker names and allowlist key names do not always match 1:1 (e.g. `\ng` = grammar **notes** in MDF, not `gender`). Map at export time per project.

---

## User-extended allowlist (proposed — not implemented)

Goal: keep canonical fields fixed, but let each dictionary (or run) **add** MDF-aligned keys to Stage 2 discovery without editing Python.

Possible design (not implemented yet):

1. **File:** `{entry}/stage2_extra_fields.yaml` or global `config/stage2_extra_fields.yaml`
2. **Contents:** list of snake_case keys + optional `mdf_marker` + one-line extraction hint
3. **Merge order:** `DEFAULT_EXTRA_FIELDS_ALLOWLIST` + user keys (deduplicated)
4. **Prompt:** inject merged list into `<extra_fields_discovery>` only when `--discover-extra-fields`
5. **TSV:** unchanged — dynamic columns from keys present in JSON
6. **Export:** future `json_to_mdf()` uses `mdf_marker` from config

Example:

```yaml
# assets/dictionaries/samples/Some-Lang-English/stage2_extra_fields.yaml
extra_fields:
  - key: etymology
    mdf: et
    hint: "Line introduced by Etym: or Asal:"
  - key: scientific_name
    mdf: sc
    hint: "Latin binomial when marked as scientific name"
```

Until that exists, extend the allowlist by editing `EXTRA_FIELDS_ALLOWLIST` in `src/dictextractor/llm/prompts.py` and document new keys in the table above.

---

## References

| Resource | URL |
| --- | --- |
| SIL MDF overview | https://software.sil.org/shoebox/mdf/ |
| *Making dictionaries* (2000) PDF | https://s3.amazonaws.com/downloads.sil.org/legacy/shoebox/MDF_2000.pdf |
| LingTranSoft MDF reference | https://lingtran.net/MDF-Code-reference |
| dictionary-extractor Stage 2 mapping | `docs/stage_2_outline.md` |
| Stage 2 methodology | `docs/stage_2_methodology.md` |
