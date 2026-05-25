#!/usr/bin/env bash
#
# eval-flat: flat transcription vs *_stage1_GOLD_flat.txt (spec v2).
#
# Run:  bash examples/evaluation/run_stage1_eval_flat.sh
#
# Keep in sync with examples/stage-1/run_stage1_extraction_flat.sh (commented +
# uncommented entries listed here as active eval targets).
#
# Outputs under evaluations/stage1_flat_eval/:
#   stage1_flat_eval_detailed.csv       — all non-OCR-hint experiments
#   stage1_flat_eval_summary.csv        — same (per experiment × language)
#   stage1_flat_eval_ocr_hint_detailed.csv
#   stage1_flat_eval_ocr_hint_summary.csv — LLM runs with ocr_hint.used=true
#   gemini31pro_flat_alpha_ocr_summary.csv — subset of OCR-hint sidecar
#   stage1_flat_eval_cache.json
#   <experiment>/stage1_flat_evaluation_report.*
#
# OCR-hint LLM experiments (e.g. gemini31pro_flat_alpha_ocr) are split out of the
# main summary CSVs automatically via run_config.json → ocr_hint.used.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Keep in sync with examples/stage-1/run_stage1_extraction_flat.sh LANGUAGES block.
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

echo "=== Stage 1 eval-flat (LLM flat ablations + VLM OCR; OCR-hint split) ==="
echo "Languages (${#LANGUAGES[@]}): ${LANGUAGES[*]}"

uv run dictextractor-eval-flat \
  --samples-dir assets/dictionaries/samples \
  --all-experiments \
  --languages "${LANGUAGES[@]}" \
  -o evaluations/stage1_flat_eval \
  "$@"
