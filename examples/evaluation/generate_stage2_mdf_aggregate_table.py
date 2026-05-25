#!/usr/bin/env python3
"""Generate Stage 2 MDF aggregate-by-model LaTeX table."""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "evaluations/stage2_mdf_eval/stage2_mdf_eval_summary.csv"
OUT_TEX = Path(__file__).with_name("stage2_mdf_aggregate_table.tex")

SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        r"\multicolumn{6}{l}{\cellcolor{gray!10}\textit{Vision Language Models}} \\",
        [
            ("Qwen3-VL-235B", "qwen3vl235_high_mdf_nointro_notoolbox"),
            ("Qwen3-VL-235B", "qwen3vl235_high_mdf_intro_notoolbox"),
            ("Qwen3-VL-235B", "qwen3vl235_high_mdf_nointro_toolbox"),
            ("Qwen3-VL-235B", "qwen3vl235_high_mdf_intro_toolbox"),
        ],
    ),
    (
        r"\multicolumn{6}{l}{\cellcolor{gray!10}\textit{General-purpose LLMs}} \\",
        [
            ("Claude Opus 4.7", "claudeopus47_high_mdf_nointro_notoolbox"),
            ("Claude Opus 4.7", "claudeopus47_high_mdf_intro_notoolbox"),
            ("Claude Opus 4.7", "claudeopus47_high_mdf_nointro_toolbox"),
            ("Claude Opus 4.7", "claudeopus47_high_mdf_intro_toolbox"),
            ("GPT-5.5", "gpt55_high_mdf_nointro_notoolbox"),
            ("GPT-5.5", "gpt55_high_mdf_intro_notoolbox"),
            ("GPT-5.5", "gpt55_high_mdf_nointro_toolbox"),
            ("GPT-5.5", "gpt55_high_mdf_intro_toolbox"),
            ("Gemini 3.1 Pro", "gemini31pro_high_mdf_nointro_notoolbox"),
            ("Gemini 3.1 Pro", "gemini31pro_high_mdf_intro_notoolbox"),
            ("Gemini 3.1 Pro", "gemini31pro_high_mdf_nointro_toolbox"),
            ("Gemini 3.1 Pro", "gemini31pro_high_mdf_intro_toolbox"),
        ],
    ),
]


def fmt(v: float) -> str:
    return f"{v:.2f}"


def cell(v: float, bold: bool) -> str:
    text = fmt(v)
    return f"\\textbf{{{text}}}" if bold else text


def checkmark(enabled: bool) -> str:
    return r"\cmark" if enabled else ""


def parse_flags(experiment: str) -> tuple[bool, bool]:
    base = experiment.split("high_mdf_", 1)[1]
    return base.startswith("intro_"), base.endswith("_toolbox")


def main() -> None:
    aggregates = {
        r["experiment"]: r
        for r in csv.DictReader(CSV_PATH.open(newline=""))
        if r["page_id"] == "__aggregate__" and not r["experiment"].endswith("_goldcheat")
    }

    experiments = [exp for _, rows in SECTIONS for _, exp in rows]
    rec_values = [float(aggregates[exp]["Record_Accuracy"]) for exp in experiments]
    f1_values = [float(aggregates[exp]["MDF_Fields_F1"]) for exp in experiments]
    roe_values = [float(aggregates[exp]["ReadOrderEdit"]) for exp in experiments]

    # Compare on displayed scale so bolding matches formatted cells.
    best_rec = max(fmt(v) for v in rec_values)
    best_f1 = max(fmt(v) for v in f1_values)
    best_roe = min(fmt(v) for v in roe_values)

    body: list[str] = []
    first = True
    for section_line, rows in SECTIONS:
        if not first:
            body.append(r"\midrule")
        first = False
        body.append(section_line)
        for model, experiment in rows:
            row = aggregates[experiment]
            intro, mdf = parse_flags(experiment)
            rec = float(row["Record_Accuracy"])
            f1 = float(row["MDF_Fields_F1"])
            roe = float(row["ReadOrderEdit"])
            body.append(
                f"{model} & {checkmark(intro)} & {checkmark(mdf)} & "
                f"{cell(rec, fmt(rec) == best_rec)} & "
                f"{cell(f1, fmt(f1) == best_f1)} & "
                f"{cell(roe, fmt(roe) == best_roe)} \\\\"
            )

    tex = "\n".join(
        [
            "% Requires: booktabs, pifont, colortbl",
            r"% \newcommand{\cmark}{\ding{51}}",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{2pt}",
            r"\caption{Stage 2 MDF evaluation results aggregated by model. Check marks indicate that the corresponding condition was enabled.}",
            r"\label{tab:stage2-mdf-aggregate}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\textbf{Model} & \textbf{Intro} & \textbf{MDF} & \textbf{Rec. Acc.} & \textbf{MDF F1} & \textbf{ROE} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    OUT_TEX.write_text(tex)
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
