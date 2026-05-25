#!/usr/bin/env bash
# Stage-1 flat transcription: per-language best model+config from the flat eval
# sweep WITH OCR hints added, written under outputs/stage-1-ocr/<base>_ocrhint/.
#
# Best configs from stage1_flat_eval_summary.csv (lowest TextEdit per language).
# OCR-hint LLM runs live in stage1_flat_eval_ocr_hint_summary.csv (split from main).
#
# OCR hints are always enabled for this arm (Mathpix convert + --ocr-hint), regardless
# of whether the baseline sweep row had ocr-hint=false.
#
# Targets all 30 languages from run_stage1_extraction_flat.sh (commented + active).
#
# Usage:
#   bash examples/stage-1/stage-1-ocr/run_stage1_per_lang_best_flat_alpha_ocr.sh
#   bash examples/stage-1/stage-1-ocr/run_stage1_per_lang_best_flat_alpha_ocr.sh --overwrite
#   SKIP_MATHPIX_CONVERT=1 bash examples/stage-1/stage-1-ocr/run_stage1_per_lang_best_flat_alpha_ocr.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"
EVAL_CSV="${EVAL_CSV:-evaluations/stage1_flat_eval/stage1_flat_eval_summary.csv}"
STAGE1_OUTPUT_SUBDIR="${STAGE1_OUTPUT_SUBDIR:-stage-1-ocr}"
SKIP_MATHPIX_CONVERT="${SKIP_MATHPIX_CONVERT:-0}"
EXTRACT_EXTRA_ARGS=("$@")

# Full language list from run_stage1_extraction_flat.sh (lines 22–55).
LANGUAGES=(
    Canala-English
    Chepang-English
    Efik-English
    Na-English-Chinese-French
    Reel-English
    Ritharngu-English
    Shilluk-English
    Evenki-Russian
    Chukchi-Russian
    Circassian-English-Turkish
    Nahuatl-French
    Khmer-English
    Malay-English
    Kashmiri-English
    Greek-English
    Telugu-English
    "Iñupiatun Eskimo-English"
    "Vernacular Syriac-Kurdish_Turkish-English"
    Syriac-English
    Tiri-English
    Thai-Russian
    Assyrian-English
    Yiddish-English
    Georgian-Russian
    Japanese-English
    Punjabi-English
    Gujarati-English
    Gojri-English-Hindi
    Bengalese-English
    Sanskrit-English
)

GEMINI_FLASH_MODEL="gemini/gemini-3-flash-preview"
GEMINI_PRO_MODEL="gemini/gemini-3.1-pro-preview"
GPT55_MODEL="openrouter/openai/gpt-5.5"
CLAUDE_OPUS47_MODEL="openrouter/anthropic/claude-opus-4.7"
QWEN3_VL_MODEL="openrouter/qwen/qwen3-vl-235b-a22b-instruct"
OPENROUTER_STAGE1_REASONING="${OPENROUTER_STAGE1_REASONING:-low}"

run_stage1() {
    if ! uv run dictextractor-extract "$@"; then
        echo "WARNING: stage-1 run failed or was skipped; continuing." >&2
    fi
}

stage1_model_for_experiment() {
    local exp="$1"
    case "${exp}" in
        gemini31pro_*) echo "${GEMINI_PRO_MODEL}" ;;
        gemini3flash_*) echo "${GEMINI_FLASH_MODEL}" ;;
        gpt55_*) echo "${GPT55_MODEL}" ;;
        claudeopus47_*) echo "${CLAUDE_OPUS47_MODEL}" ;;
        qwen3vl235_*) echo "${QWEN3_VL_MODEL}" ;;
        *)
            echo "ERROR: unknown stage-1 experiment prefix: ${exp}" >&2
            return 1
            ;;
    esac
}

stage1_reasoning_for_experiment() {
    local exp="$1"
    case "${exp}" in
        gemini31pro_*|gemini3flash_*) echo "low" ;;
        *) echo "${OPENROUTER_STAGE1_REASONING}" ;;
    esac
}

ocrhint_experiment_name() {
    local base="$1"
    if [[ "${base}" == *_ocrhint ]]; then
        echo "${base}"
    else
        echo "${base}_ocrhint"
    fi
}

run_mathpix_convert_for_lang() {
    local lang="$1"
    local -a mathpix_convert_extra=()
    local arg

    for arg in "${EXTRACT_EXTRA_ARGS[@]}"; do
        if [[ "${arg}" == "--overwrite" ]]; then
            mathpix_convert_extra+=(--overwrite-files --force)
        fi
    done

    echo ""
    echo " Mathpix convert: ${lang}"
    if ! uv run dictextractor-mathpix-convert \
        --samples-dir "${SAMPLES_DIR}" \
        --languages "${lang}" \
        "${mathpix_convert_extra[@]}"; then
        echo "WARNING: Mathpix convert failed for ${lang}; extraction may skip OCR hints." >&2
    fi
}

validate_flat_outputs() {
    local lang="$1"
    local experiment="$2"
    local exp_dir="${SAMPLES_DIR}/${lang}/outputs/${STAGE1_OUTPUT_SUBDIR}/${experiment}"
    local snippets_dir="${SAMPLES_DIR}/${lang}/snippets"
    local missing=0

    if [[ ! -d "${snippets_dir}" ]]; then
        echo "WARNING: ${lang}: snippets dir missing" >&2
        return 1
    fi

    for snippet in "${snippets_dir}"/*; do
        [[ -f "${snippet}" ]] || continue
        case "${snippet##*.}" in
            png|jpg|jpeg|webp|gif|pdf) ;;
            *) continue ;;
        esac
        local stem
        stem="$(basename "${snippet}" | sed 's/\.[^.]*$//')"
        local flat_path="${exp_dir}/${stem}/${stem}_stage1_flat.txt"
        if [[ ! -s "${flat_path}" ]]; then
            echo "WARNING: ${lang}/${stem}: missing or empty ${flat_path}" >&2
            missing=$((missing + 1))
        fi
    done

    if [[ "${missing}" -gt 0 ]]; then
        echo "WARNING: ${lang}/${experiment}: ${missing} page(s) missing flat output." >&2
        return 1
    fi
    echo "VALIDATION OK: ${lang}/${experiment} — flat outputs present."
}

echo "============================================================"
echo " Stage-1 per-language best config + OCR hint → ${STAGE1_OUTPUT_SUBDIR}/"
echo "  Eval CSV: ${EVAL_CSV}"
echo "  Languages: ${#LANGUAGES[@]}"
echo "============================================================"

if [[ ! -f "${EVAL_CSV}" ]]; then
    echo "ERROR: eval summary missing: ${EVAL_CSV}" >&2
    exit 1
fi

declare -a BEST_ROWS=()
while IFS=$'\t' read -r lang base_exp textedit alphabet ocr; do
    BEST_ROWS+=("${lang}"$'\t'"${base_exp}"$'\t'"${textedit}"$'\t'"${alphabet}"$'\t'"${ocr}")
done < <(
    uv run python3 - "${EVAL_CSV}" "${LANGUAGES[@]}" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
target_langs = sys.argv[2:]
best: dict[str, dict[str, str]] = {}

with open(csv_path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        lang = row["language"]
        if lang not in target_langs:
            continue
        textedit = float(row["TextEdit"])
        if lang not in best or textedit < float(best[lang]["TextEdit"]):
            best[lang] = row

for lang in target_langs:
    if lang not in best:
        print(f"ERROR: no eval row for {lang}", file=sys.stderr)
        sys.exit(1)
    row = best[lang]
    print(
        f"{lang}\t{row['experiment']}\t{float(row['TextEdit']):.6f}\t"
        f"{row['alphabet']}\t{row['ocr-hint']}"
    )
PY
)

for row in "${BEST_ROWS[@]}"; do
    IFS=$'\t' read -r lang base_exp textedit alphabet ocr <<<"${row}"
    experiment="$(ocrhint_experiment_name "${base_exp}")"
    model="$(stage1_model_for_experiment "${base_exp}")"
    reasoning="$(stage1_reasoning_for_experiment "${base_exp}")"

    local_flags=(
        --strategy two_stage
        --stage 1
        --stage1-mode flat
        --model "${model}"
        --stage1-reasoning "${reasoning}"
        --samples-dir "${SAMPLES_DIR}"
        --languages "${lang}"
        --experiment-name "${experiment}"
        --stage1-output-subdir "${STAGE1_OUTPUT_SUBDIR}"
    )
    if [[ "${alphabet}" == "false" ]]; then
        local_flags+=(--no-alphabet)
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo " ${lang}"
    echo "  Best sweep (excl flat_alpha_ocr): ${base_exp} (TextEdit=${textedit})"
    echo "  Output slot: ${STAGE1_OUTPUT_SUBDIR}/${experiment}"
    echo "  Model: ${model} | Reasoning: ${reasoning}"
    echo "  Alphabet: ${alphabet} | OCR hint: forced on (baseline had ocr-hint=${ocr})"
    echo "------------------------------------------------------------"

    if [[ "${SKIP_MATHPIX_CONVERT}" != "1" ]]; then
        run_mathpix_convert_for_lang "${lang}"
    fi

    run_stage1 "${local_flags[@]}" "${EXTRACT_EXTRA_ARGS[@]}"

    uv run python3 scripts/validate_stage1_flat_experiment.py \
        --samples-dir "${SAMPLES_DIR}" \
        --experiment-name "${experiment}" \
        --stage1-output-subdir "${STAGE1_OUTPUT_SUBDIR}" \
        --languages "${lang}" \
        || echo "WARNING: OCR validation failed for ${lang}/${experiment}." >&2
done

echo ""
echo "Done: per-language best Stage-1 configs + OCR hint across ${#LANGUAGES[@]} languages → ${STAGE1_OUTPUT_SUBDIR}/."
