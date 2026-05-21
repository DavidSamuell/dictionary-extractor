#!/usr/bin/env bash
# Stage-2 structuring (MDF-aligned schema) for the same language subset as
# examples/stage-1/run_stage1_vlm_ocr.sh and run_stage1_extraction.sh.
#
# Reads stage-1 column TSVs from:
#   {lang}/outputs/stage-1/<STAGE1_EXPERIMENT>/<stem>/<stem>_stage1.tsv
# Writes stage-2 JSON/TSV to:
#   {lang}/outputs/stage-2/<STAGE2_EXPERIMENT>/<stem>/<stem>.json|.tsv
#
# Introduction: ON by default in batch mode when {lang}/introduction/ exists
# (images, PDFs, or text). No --intro flag needed; extract.py sets it per entry.
#
# Prerequisites: stage-1 TSVs must already exist for STAGE1_EXPERIMENT (e.g. from
# run_stage1_extraction.sh). VLM OCR slots (MinerU2.5-Pro, etc.) use different
# artifacts unless converted to *_stage1.tsv.
#
# Usage:
#   bash examples/stage-2/run_stage2_extraction.sh
#   bash examples/stage-2/run_stage2_extraction.sh --overwrite

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"
STAGE1_EXPERIMENT="${STAGE1_EXPERIMENT:-gemini3flash_alpha_ocr}"
STAGE2_EXPERIMENT="${STAGE2_EXPERIMENT:-pro_highreasoning_mdf}"
MODEL="${MODEL:-gemini/gemini-3.1-pro-preview}"

# Keep in sync with examples/stage-1/run_stage1_vlm_ocr.sh LANGUAGES.
LANGUAGES=(
    # Assyrian-English
    # Canala-English
    # Chepang-English
    # Chung-English
    # Efik-English
    # Na-English-Chinese
    # Reel-English
    # Ritharngu-English
    # Shilluk-English
    # Circassian-English-Turkish
    Evenki-Russian
    Chukchi-Russian
    Yiddish-English
    Nahuatl-French
)

EXTRACT_EXTRA_ARGS=("$@")

echo "Stage 2: ${MODEL} | reasoning=high"
echo "  samples:     ${SAMPLES_DIR}"
echo "  stage-1 src: ${STAGE1_EXPERIMENT}"
echo "  stage-2 out: ${STAGE2_EXPERIMENT}"
echo "  languages:   ${#LANGUAGES[@]}"
echo "  intro:       auto per language if introduction/ exists"
echo ""

uv run dictextractor-extract \
    --strategy two_stage \
    --stage 2 \
    --samples-dir "${SAMPLES_DIR}" \
    --languages "${LANGUAGES[@]}" \
    --model "${MODEL}" \
    --stage2-reasoning high \
    --experiment-name "${STAGE1_EXPERIMENT}" \
    --stage2-experiment-name "${STAGE2_EXPERIMENT}" \
    "${EXTRACT_EXTRA_ARGS[@]}"
