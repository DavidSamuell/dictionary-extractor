#!/usr/bin/env bash
# Stage-2 gold field cheat sheet: per-language best model+config from the main
# MDF sweep, with Pass 1 loaded from outputs/stage-2-gold/field_cheatsheet.json.
#
# Best configs are read from evaluations/stage2_mdf_eval/stage2_mdf_eval_summary.csv
# (highest MDF_Fields_F1 per language page; __aggregate__ rows excluded).
#
# Outputs: {lang}/outputs/stage-2/<best_experiment>_goldcheat/
#
# Usage:
#   bash examples/stage-2/run_stage2_gold_cheatsheet.sh
#   bash examples/stage-2/run_stage2_gold_cheatsheet.sh --overwrite

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"
EVAL_CSV="${EVAL_CSV:-evaluations/stage2_mdf_eval/stage2_mdf_eval_summary.csv}"
TOOLBOX_PDF="Pages from ToolboxReferenceManual.pdf"
EXTRACT_EXTRA_ARGS=("$@")

LANGUAGES=(
    "Iñupiatun Eskimo-English"
    Kashmiri-English
    Tiri-English
    Evenki-Russian
    Nahuatl-French
    Na-English-Chinese-French
)

run_stage2() {
    if ! uv run dictextractor-extract "$@"; then
        echo "WARNING: stage-2 gold-cheat run failed or was skipped; continuing." >&2
    fi
}

stage2_model_for_experiment() {
    local exp="$1"
    case "${exp}" in
        gemini31pro_*) echo "gemini/gemini-3.1-pro-preview" ;;
        gpt55_*) echo "openrouter/openai/gpt-5.5" ;;
        claudeopus47_*) echo "openrouter/anthropic/claude-opus-4.7" ;;
        qwen3vl235_*) echo "openrouter/qwen/qwen3-vl-235b-a22b-instruct" ;;
        *)
            echo "ERROR: unknown stage-2 experiment prefix: ${exp}" >&2
            return 1
            ;;
    esac
}

stage2_flags_for_experiment() {
    local exp="$1"
    local -n _out="$2"
    _out=()
    if [[ "${exp}" == *"_nointro_"* ]]; then
        _out+=(--no-intro)
    fi
    if [[ "${exp}" == *_toolbox ]]; then
        _out+=(--toolbox-pdf "${TOOLBOX_PDF}")
    fi
}

echo "============================================================"
echo " Stage-2 gold field cheat sheet (per-language best config)"
echo "  Eval CSV: ${EVAL_CSV}"
echo "  Languages: ${#LANGUAGES[@]}"
echo "============================================================"

if [[ ! -f "${EVAL_CSV}" ]]; then
    echo "ERROR: eval summary missing: ${EVAL_CSV}" >&2
    exit 1
fi

missing_gold=()
for lang in "${LANGUAGES[@]}"; do
    gold="${SAMPLES_DIR}/${lang}/outputs/stage-2-gold/field_cheatsheet.json"
    if [[ ! -f "${gold}" ]]; then
        missing_gold+=("${lang} → ${gold}")
    fi
done
if [[ ${#missing_gold[@]} -gt 0 ]]; then
    echo "ERROR: missing gold field_cheatsheet.json:" >&2
    printf '  %s\n' "${missing_gold[@]}" >&2
    exit 1
fi

declare -a BEST_ROWS=()
while IFS=$'\t' read -r lang base_exp f1; do
    BEST_ROWS+=("${lang}"$'\t'"${base_exp}"$'\t'"${f1}")
done < <(
    uv run python3 - "${EVAL_CSV}" "${LANGUAGES[@]}" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
target_langs = sys.argv[2:]
best: dict[str, tuple[str, float]] = {}

with open(csv_path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        page_id = row["page_id"]
        if page_id == "__aggregate__":
            continue
        lang = page_id.split("/", 1)[0]
        if lang not in target_langs:
            continue
        f1 = float(row["MDF_Fields_F1"])
        exp = row["experiment"]
        if lang not in best or f1 > best[lang][1]:
            best[lang] = (exp, f1)

for lang in target_langs:
    if lang not in best:
        print(f"ERROR: no eval row for {lang}", file=sys.stderr)
        sys.exit(1)
    exp, f1 = best[lang]
    print(f"{lang}\t{exp}\t{f1:.6f}")
PY
)

for row in "${BEST_ROWS[@]}"; do
    IFS=$'\t' read -r lang base_exp f1 <<<"${row}"
    goldcheat_exp="${base_exp}_goldcheat"
    model="$(stage2_model_for_experiment "${base_exp}")"
    stage2_ablation_flags=()
    stage2_flags_for_experiment "${base_exp}" stage2_ablation_flags

    echo ""
    echo "------------------------------------------------------------"
    echo " ${lang}"
    echo "  Best sweep: ${base_exp} (MDF_Fields_F1=${f1})"
    echo "  Output slot: ${goldcheat_exp}"
    echo "  Model: ${model}"
    echo "  Flags: ${stage2_ablation_flags[*]:-<none>}"
    echo "------------------------------------------------------------"

    run_stage2 \
        --strategy two_stage --stage 2 \
        --stage2-mode direct_mdf \
        --model "${model}" \
        --stage2-reasoning high \
        --samples-dir "${SAMPLES_DIR}" \
        --languages "${lang}" \
        --one-page-per-entry \
        --stage1-input flat \
        --field-cheatsheet-gold \
        --stage2-experiment-name "${goldcheat_exp}" \
        "${stage2_ablation_flags[@]}" \
        "${EXTRACT_EXTRA_ARGS[@]}"
done

echo ""
echo "Done: gold cheat sheet Stage-2 across ${#LANGUAGES[@]} languages (per-language best configs)."
