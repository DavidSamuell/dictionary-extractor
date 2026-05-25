#!/usr/bin/env python3
"""Generate Stage 2 gold field-map cheat sheet LaTeX table (single-column layout)."""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAIN_CSV = REPO / "evaluations/stage2_mdf_eval/stage2_mdf_eval_summary.csv"
GOLD_CSV = REPO / "evaluations/stage_2_gold_cheat_sheet/stage2_mdf_eval_summary.csv"
OUT_TEX = Path(__file__).with_name("stage2_gold_cheat_sheet_table.tex")

LANG_DISPLAY = {
    "Evenki-Russian": "Evenki",
    "Iñupiatun Eskimo-English": "Iñupiatun",
    "Kashmiri-English": "Kashmiri",
    "Na-English-Chinese-French": "Na (Mosuo)",
    "Tiri-English": "Tiri",
    "Nahuatl-French": "Nahuatl",
}

# Per-language best config from tab:stage2-mdf-aggregate (excludes perfect-score Iñupiatun).
GOLD_CHEAT_CONFIGS: dict[str, str] = {
    "Evenki-Russian/page_1": "claudeopus47_high_mdf_nointro_toolbox",
    "Kashmiri-English/page_14": "gemini31pro_high_mdf_intro_notoolbox",
    "Na-English-Chinese-French/page_19": "gemini31pro_high_mdf_nointro_toolbox",
    "Tiri-English/page_15": "gemini31pro_high_mdf_nointro_toolbox",
    "Nahuatl-French/page_74": "gpt55_high_mdf_intro_toolbox",
}

MODEL_DISPLAY = {
    "claudeopus47": "Claude Opus 4.7",
    "gemini31pro": "Gemini 3.1 Pro",
    "gpt55": "GPT-5.5",
}

LANG_ORDER = [
    "Evenki-Russian",
    "Kashmiri-English",
    "Na-English-Chinese-French",
    "Tiri-English",
    "Nahuatl-French",
]


def fmt(v: float) -> str:
    return f"{v:.2f}"


def bold(v: float) -> str:
    return f"\\textbf{{{fmt(v)}}}"


def parse_flags(experiment: str) -> tuple[bool, bool]:
    base = experiment.removesuffix("_goldcheat").split("high_mdf_", 1)[1]
    intro = base.startswith("intro_")
    mdf = base.endswith("_toolbox")
    return intro, mdf


def checkmark(enabled: bool) -> str:
    return r"\cmark" if enabled else ""


def load_rows() -> list[dict[str, object]]:
    main = {
        (r["experiment"], r["page_id"]): r
        for r in csv.DictReader(MAIN_CSV.open(newline=""))
        if r["page_id"] != "__aggregate__" and not r["experiment"].endswith("_goldcheat")
    }
    gold_by_page = {
        r["page_id"]: r
        for r in csv.DictReader(GOLD_CSV.open(newline=""))
        if r["page_id"] != "__aggregate__"
    }
    rows: list[dict[str, object]] = []
    for page_id, base_exp in GOLD_CHEAT_CONFIGS.items():
        gold_exp = f"{base_exp}_goldcheat"
        key = (base_exp, page_id)
        if key not in main:
            raise KeyError(f"Missing main-sweep row for {key}")
        if page_id not in gold_by_page:
            raise KeyError(f"Missing gold-cheat row for {page_id}")
        lang = page_id.split("/", 1)[0]
        intro, mdf = parse_flags(gold_exp)
        model_key = base_exp.split("_", 1)[0]
        rows.append(
            {
                "lang": lang,
                "model": MODEL_DISPLAY[model_key],
                "intro": intro,
                "mdf": mdf,
                "inf_f1": float(main[key]["MDF_Fields_F1"]),
                "gold_f1": float(gold_by_page[page_id]["MDF_Fields_F1"]),
            }
        )
    order = {lang: i for i, lang in enumerate(LANG_ORDER)}
    rows.sort(key=lambda x: order[x["lang"]])
    return rows


def f1_cells(inf_f1: float, gold_f1: float) -> tuple[str, str]:
    if abs(inf_f1 - gold_f1) < 1e-9:
        cell = fmt(inf_f1)
        return cell, cell
    if inf_f1 > gold_f1:
        return bold(inf_f1), fmt(gold_f1)
    return fmt(inf_f1), bold(gold_f1)


def render_row(row: dict[str, object]) -> str:
    lang = LANG_DISPLAY[str(row["lang"])]
    inf_cell, gold_cell = f1_cells(float(row["inf_f1"]), float(row["gold_f1"]))
    return (
        f"{lang} & {row['model']} & {checkmark(bool(row['intro']))} & "
        f"{checkmark(bool(row['mdf']))} & {inf_cell} & {gold_cell}"
    )


def main() -> None:
    rows = load_rows()
    inf_f1 = sum(float(r["inf_f1"]) for r in rows) / len(rows)
    gold_f1 = sum(float(r["gold_f1"]) for r in rows) / len(rows)

    body = [render_row(r) + r" \\" for r in rows]
    body.append(
        f"\\midrule\n\\textbf{{Macro avg.}} &  &  &  & {fmt(inf_f1)} & {fmt(gold_f1)} \\\\"
    )

    tex = "\n".join(
        [
            "% Requires: booktabs, pifont, adjustbox",
            r"% \newcommand{\cmark}{\ding{51}}",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Stage~2 gold field-map upper bound on dictionaries where the model doesn't generate a perfect MDF file. "
            r"Each row uses the per-language best model and ablation setting from Table~\ref{tab:stage2-mdf-aggregate}; "
            r"Pass~1 field maps are inferred or replaced with human-validated gold cheat sheets.}",
            r"\label{tab:stage2-gold-cheat-sheet}",
            r"\begin{adjustbox}{width=\columnwidth,center}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l l cc rr}",
            r"\toprule",
            r"\textbf{Dictionary} & \textbf{Model} & \textbf{Intro} & \textbf{MDF} & "
            r"\textbf{Inf. F1} & \textbf{Gold F1} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
            "",
        ]
    )
    OUT_TEX.write_text(tex)
    print(f"Wrote {OUT_TEX} ({len(rows)} dictionaries, single-column layout)")


if __name__ == "__main__":
    main()
