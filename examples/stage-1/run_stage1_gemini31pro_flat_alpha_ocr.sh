#!/usr/bin/env bash
# Stage-1 flat transcription: Gemini 3.1 Pro + alphabet + Mathpix OCR hint.
#
# Runs across all languages listed in run_stage1_extraction_flat.sh (commented and
# active entries). OCR hints are read from {lang}/mathpix/{stem}.md (Mathpix
# Convert markdown). Alphabet comes from {lang}/alphabet.txt when present.
#
# Outputs:
#   {lang}/outputs/stage-1/gemini31pro_flat_alpha_ocr/{stem}/{stem}_stage1_flat.txt
#   plus {stem}_stage1_raw.json and {stem}_stage1_input.json per page.
#
# Prerequisites:
#   GEMINI_API_KEY — direct Gemini API
#   MATHPIX_* credentials — for dictextractor-mathpix-convert (skipped if hints exist)
#
# Evaluate preds: add gemini31pro_flat_alpha_ocr to run_stage1_eval_flat.sh, then:
#   bash examples/evaluation/run_stage1_eval_flat.sh
#
# Usage:
#   bash examples/stage-1/run_stage1_gemini31pro_flat_alpha_ocr.sh
#   bash examples/stage-1/run_stage1_gemini31pro_flat_alpha_ocr.sh --overwrite
#   SKIP_MATHPIX_CONVERT=1 bash examples/stage-1/run_stage1_gemini31pro_flat_alpha_ocr.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gemini31pro_flat_alpha_ocr}"
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

GEMINI_PRO_MODEL="gemini/gemini-3.1-pro-preview"

run_mathpix_convert() {
    local lang
    local -a mathpix_convert_extra=()
    local arg

    for arg in "${EXTRACT_EXTRA_ARGS[@]}"; do
        if [[ "${arg}" == "--overwrite" ]]; then
            mathpix_convert_extra+=(--overwrite-files --force)
        fi
    done

    echo ""
    echo "============================================================"
    echo " Mathpix Convert: snippets → mathpix/{stem}.md (+ .lines.json)"
    echo "  (${#LANGUAGES[@]} languages; OCR hints for Gemini Stage 1)"
    echo "============================================================"

    for lang in "${LANGUAGES[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo " Mathpix convert: ${lang}"
        echo "------------------------------------------------------------"
        if ! uv run dictextractor-mathpix-convert \
            --samples-dir "${SAMPLES_DIR}" \
            --languages "${lang}" \
            "${mathpix_convert_extra[@]}"; then
            echo "WARNING: Mathpix convert failed for ${lang}; Gemini may skip if hints missing." >&2
        fi
    done
}

run_gemini31pro_flat_alpha_ocr() {
    echo ""
    echo "============================================================"
    echo " Gemini 3.1 Pro flat: ${EXPERIMENT_NAME}"
    echo "  Alphabet: on | OCR hint: on (mathpix/*.md) | Reasoning: low"
    echo "  Languages: ${#LANGUAGES[@]}"
    echo "============================================================"

    if ! uv run dictextractor-extract \
        --strategy two_stage \
        --stage 1 \
        --stage1-mode flat \
        --model "${GEMINI_PRO_MODEL}" \
        --stage1-reasoning low \
        --samples-dir "${SAMPLES_DIR}" \
        --languages "${LANGUAGES[@]}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        "${EXTRACT_EXTRA_ARGS[@]}"; then
        echo "WARNING: ${EXPERIMENT_NAME} failed or was skipped." >&2
        return 1
    fi
}

validate_outputs() {
    echo ""
    echo "============================================================"
    echo " Validating ${EXPERIMENT_NAME} outputs"
    echo "============================================================"
    uv run python3 scripts/validate_stage1_flat_experiment.py \
        --samples-dir "${SAMPLES_DIR}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        --languages "${LANGUAGES[@]}"
}

if [[ "${SKIP_MATHPIX_CONVERT}" != "1" ]]; then
    run_mathpix_convert
else
    echo "SKIP_MATHPIX_CONVERT=1 — assuming mathpix/*.md OCR hints already exist."
fi

set +e
run_gemini31pro_flat_alpha_ocr
extract_status=$?
set -e

if [[ "${extract_status}" -ne 0 ]]; then
    echo "WARNING: extraction exited with status ${extract_status}; validating partial outputs." >&2
fi

validate_outputs

echo ""
echo "Done: ${EXPERIMENT_NAME} across ${#LANGUAGES[@]} languages."
