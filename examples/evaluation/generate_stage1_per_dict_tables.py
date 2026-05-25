#!/usr/bin/env python3
"""Generate Stage 1 per-dictionary LaTeX tables from flat eval summary CSV."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "evaluations/stage1_flat_eval/stage1_flat_eval_summary.csv"
OUT_PATH = Path(__file__).with_name("stage1_per_dict_tables_generated.tex")

DISPLAY_NAMES: dict[str, str] = {
    "Bengalese-English": "Bengali-English",
}

DICT_ORDER: list[str] = [
    "Assyrian-English",
    "Bengalese-English",
    "Canala-English",
    "Chepang-English",
    "Chukchi-Russian",
    "Circassian-English-Turkish",
    "Efik-English",
    "Evenki-Russian",
    "Georgian-Russian",
    "Gojri-English-Hindi",
    "Greek-English",
    "Gujarati-English",
    "Iñupiatun Eskimo-English",
    "Japanese-English",
    "Kashmiri-English",
    "Khmer-English",
    "Malay-English",
    "Na-English-Chinese-French",
    "Nahuatl-French",
    "Punjabi-English",
    "Reel-English",
    "Ritharngu-English",
    "Sanskrit-English",
    "Shilluk-English",
    "Syriac-English",
    "Telugu-English",
    "Thai-Russian",
    "Tiri-English",
    "Vernacular Syriac-Kurdish_Turkish-English",
    "Yiddish-English",
]

ROW_SPECS: list[tuple[str, str | None, str]] = [
    ("GLM-OCR", "alpha", "GLM-OCR-flat_alpha"),
    ("GLM-OCR", None, "GLM-OCR-flat_noalpha"),
    ("Mathpix", None, "Mathpix-OCR"),
    ("MinerU 2.5", None, "MinerU2.5-Pro"),
    ("PaddleOCR-VL", None, "PaddleOCR-VL-1.5"),
    ("Qwen3-VL-235B", "alpha", "qwen3vl235_flat_alpha"),
    ("Qwen3-VL-235B", None, "qwen3vl235_flat_noalpha"),
    ("Claude Opus 4.7", "alpha", "claudeopus47_flat_alpha"),
    ("Claude Opus 4.7", None, "claudeopus47_flat_noalpha"),
    ("GPT-5.5", "alpha", "gpt55_flat_alpha"),
    ("GPT-5.5", None, "gpt55_flat_noalpha"),
    ("Gemini 3 Flash", "alpha", "gemini3flash_flat_alpha"),
    ("Gemini 3 Flash", None, "gemini3flash_flat_noalpha"),
    ("Gemini 3.1 Pro", "alpha", "gemini31pro_flat_alpha"),
    ("Gemini 3.1 Pro", None, "gemini31pro_flat_noalpha"),
]

METRICS = ("TextEdit", "GCER", "WER", "typography_f1", "ReadOrderEdit")
METRIC_HEADERS = ("Edit", "GCER", "WER", "Mrk. F1", "RO")
LOWER_BETTER = (True, True, True, False, True)


def load_data() -> dict[tuple[str, str], dict[str, float]]:
    """Load main eval rows keyed by (language, experiment)."""
    data: dict[tuple[str, str], dict[str, float]] = {}
    with CSV_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["ocr-hint"].lower() == "true":
                continue
            exp_lower = row["experiment"].lower()
            if exp_lower.endswith("_ocr") or "ocrhint" in exp_lower:
                continue
            lang = row["language"]
            exp = row["experiment"]
            data[(lang, exp)] = {m: float(row[m]) for m in METRICS}
    return data


def fmt(v: float) -> str:
    return f"{v:.2f}"


NO_ALPHABET_OCR = {"Mathpix", "MinerU 2.5", "PaddleOCR-VL"}


def alph_cell(model: str, alph: str | None) -> str:
    if alph == "alpha" and model not in NO_ALPHABET_OCR:
        return r"\cmark"
    return ""


def render_row(
    model: str,
    alph: str | None,
    values: dict[str, float],
    bold: dict[str, bool],
) -> str:
    cells = [
        model,
        alph_cell(model, alph),
        *[f"\\textbf{{{fmt(values[m])}}}" if bold[m] else fmt(values[m]) for m in METRICS],
    ]
    return " & ".join(cells)


def best_flags(rows: list[dict[str, float]]) -> list[dict[str, bool]]:
    flags: list[dict[str, bool]] = [{m: False for m in METRICS} for _ in rows]
    for i, metric in enumerate(METRICS):
        vals = [r[metric] for r in rows]
        best = min(vals) if LOWER_BETTER[i] else max(vals)
        for j, v in enumerate(vals):
            if abs(v - best) < 1e-9:
                flags[j][metric] = True
    return flags


def render_dict_col(lang: str, data: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    display = DISPLAY_NAMES.get(lang, lang.replace("_", "-"))
    lines: list[str] = [
        rf"\multicolumn{{7}}{{>{{\columncolor{{green!12}}}}l}}{{\textbf{{{display}}}}}",
    ]

    groups: list[tuple[str, list[tuple[str, str | None, str]]]] = [
        ("OCR systems", ROW_SPECS[:5]),
        ("Vision Language Models", ROW_SPECS[5:7]),
        ("General-purpose LLMs", ROW_SPECS[7:]),
    ]

    all_values: list[dict[str, float]] = []
    row_meta: list[tuple[str, str | None]] = []

    for _label, specs in groups:
        lines.append(rf"\multicolumn{{7}}{{>{{\columncolor{{gray!12}}}}l}}{{\emph{{{_label}}}}}")
        group_values: list[dict[str, float]] = []
        group_meta: list[tuple[str, str | None]] = []
        for model, alph, exp in specs:
            key = (lang, exp)
            if key not in data:
                raise KeyError(f"Missing {exp} for {lang}")
            group_values.append(data[key])
            group_meta.append((model, alph))
        flags = best_flags(group_values)
        for (model, alph), vals, bold in zip(group_meta, group_values, flags):
            lines.append(render_row(model, alph, vals, bold))
        all_values.extend(group_values)
        row_meta.extend(group_meta)

    return lines


def merge_columns(left: list[str], right: list[str]) -> list[str]:
    out: list[str] = []
    li = ri = 0
    while li < len(left) or ri < len(right):
        if left[li].startswith(r"\multicolumn"):
            out.append(f"{left[li]} & {right[ri]} \\\\")
            li += 1
            ri += 1
        else:
            out.append(f"{left[li]} & {right[ri]} \\\\")
            li += 1
            ri += 1
    return out


def render_table(pairs: list[tuple[str, str]], *, continued: bool) -> str:
    data = load_data()
    caption = (
        "Stage 1 dictionary-specific evaluation results grouped by dictionary (continued)."
        if continued
        else "Stage 1 dictionary-specific evaluation results grouped by dictionary."
    )
    parts = [
        r"\begin{table*}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\caption{{{caption}}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lcrrrrr@{\qquad}lcrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Alph.} & \textbf{Edit} & \textbf{GCER} & \textbf{WER} & \textbf{Mrk. F1} & \textbf{RO} & "
        r"\textbf{Model} & \textbf{Alph.} & \textbf{Edit} & \textbf{GCER} & \textbf{WER} & \textbf{Mrk. F1} & \textbf{RO} \\",
        r"\midrule",
        "",
    ]

    for i, (l_lang, r_lang) in enumerate(pairs):
        if i > 0:
            parts.extend([r"\addlinespace[0.35em]", r"\midrule", r"\addlinespace[0.35em]", ""])
        merged = merge_columns(render_dict_col(l_lang, data), render_dict_col(r_lang, data))
        parts.extend(merged)

    parts.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table*}",
    ])
    return "\n".join(parts)


def main() -> None:
    chunks: list[tuple[str, str]] = []
    for i in range(0, len(DICT_ORDER), 2):
        left = DICT_ORDER[i]
        right = DICT_ORDER[i + 1] if i + 1 < len(DICT_ORDER) else DICT_ORDER[i]
        chunks.append((left, right))

    pages: list[list[tuple[str, str]]] = []
    for i in range(0, len(chunks), 3):
        pages.append(chunks[i : i + 3])

    tex_parts: list[str] = []
    for i, page in enumerate(pages):
        if i > 0:
            tex_parts.extend(["", r"\clearpage", ""])
        flat: list[tuple[str, str]] = []
        for pair in page:
            flat.append(pair)
        tex_parts.append(render_table(flat, continued=i > 0))

    OUT_PATH.write_text("\n".join(tex_parts) + "\n")
    print(f"Wrote {OUT_PATH} ({len(DICT_ORDER)} dictionaries, no OCR column)")


if __name__ == "__main__":
    main()
