#!/usr/bin/env bash
# Evaluate Stage-1 per-language best + OCR hint runs under outputs/stage-1-ocr/.
#
# Produced by: examples/stage-1/stage-1-ocr/run_stage1_per_lang_best_flat_alpha_ocr.sh
# Only evaluates *_ocrhint experiments (excludes stale flat_alpha_ocr copies).
#
# Outputs:
#   evaluations/stage_1_ocr_hint/stage1_flat_eval_summary.csv
#   evaluations/stage_1_ocr_hint/stage1_flat_eval_detailed.csv
#   evaluations/stage_1_ocr_hint/<experiment>/stage1_flat_evaluation_report.*
#
# Usage:
#   bash examples/evaluation/run_stage1_eval_ocr_hint.sh
#   bash examples/evaluation/run_stage1_eval_ocr_hint.sh --overwrite --workers 14

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/stage_1_ocr_hint}"
STAGE1_OUTPUT_SUBDIR="${STAGE1_OUTPUT_SUBDIR:-stage-1-ocr}"

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

echo "=== Stage 1 eval-flat (stage-1-ocr / per-language best + OCR hint) ==="
echo "Languages (${#LANGUAGES[@]}): ${LANGUAGES[*]}"
echo "Predictions: outputs/${STAGE1_OUTPUT_SUBDIR}/"
echo "Output: ${OUTPUT_DIR}/"
echo ""

uv run dictextractor-eval-flat \
  --samples-dir "${SAMPLES_DIR}" \
  --stage1-output-subdir "${STAGE1_OUTPUT_SUBDIR}" \
  --experiment-name-contains ocrhint \
  --languages "${LANGUAGES[@]}" \
  -o "${OUTPUT_DIR}" \
  "$@"
