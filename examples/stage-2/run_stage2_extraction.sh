#!/usr/bin/env bash
# Stage-2 structuring (MDF-aligned schema) for the same language subset as
# examples/stage-1/run_stage1_extraction.sh and run_stage1_vlm_ocr.sh.
#
# Inputs (column TSV and/or flat text — see STAGE1_INPUT):
#   {lang}/outputs/stage-1/<STAGE1_EXPERIMENT>/<stem>/<stem>_stage1.tsv
#   {lang}/outputs/stage-1/<STAGE1_EXPERIMENT>/<stem>/<stem>_stage1_flat.txt
#   {lang}/dictionary_languages.yaml  (auto-loaded; drives target_glosses keys)
#
# Outputs:
#   {lang}/outputs/stage-2/<STAGE2_EXPERIMENT>/<stem>/<stem>.json|.tsv
#   TSV includes Gloss_<code> columns from target_glosses (e.g. Gloss_en, Gloss_zh).
#
# Per-entry language config (layout, source/target codes, MDF markers, column_id
# for trilingual columns) is loaded from dictionary_languages.yaml and injected
# into the Stage-2 prompt. Regenerate all sample YAMLs from metadata:
#   uv run python scripts/generate_dictionary_languages_yaml.py --overwrite
#
# Introduction: ON by default in batch mode when {lang}/introduction/ exists.
# No --intro flag needed; extract.py sets it per entry.
#
# Prerequisites:
#   uv sync
#   Stage-1 outputs for STAGE1_EXPERIMENT (column and/or flat per STAGE1_INPUT).
#   VLM OCR flat exports work with STAGE1_INPUT=auto or flat.
#
# Usage:
#   bash examples/stage-2/run_stage2_extraction.sh
#   bash examples/stage-2/run_stage2_extraction.sh --overwrite
#   STAGE1_INPUT=flat STAGE1_EXPERIMENT=gemini3flash_flat_alpha_noocr bash examples/stage-2/run_stage2_extraction.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Avoid project-tree uv cache when GPFS quota is tight (see project README).
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"
STAGE1_EXPERIMENT="${STAGE1_EXPERIMENT:-gemini3flash_flat_alpha_noocr}"
STAGE1_INPUT="${STAGE1_INPUT:-flat}"
STAGE2_EXPERIMENT="${STAGE2_EXPERIMENT:-pro_highreasoning_mdf_stage-1-flat}"
MODEL="${MODEL:-gemini/gemini-3.1-pro-preview}"

# Keep in sync with examples/stage-1/run_stage1_extraction.sh LANGUAGES.
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
    Evenki-Russian
    Chukchi-Russian
    Circassian-English-Turkish
    Yiddish-English
    Nahuatl-French
)


EXTRACT_EXTRA_ARGS=("$@")

echo "Stage 2: ${MODEL} | reasoning=high | MDF nested schema"
echo "  samples:              ${SAMPLES_DIR}"
echo "  stage-1 source slot:  ${STAGE1_EXPERIMENT}"
echo "  stage-1 input:        ${STAGE1_INPUT} (auto=TSV then flat)"
echo "  stage-2 output slot:  ${STAGE2_EXPERIMENT}"
echo "  languages:            ${#LANGUAGES[@]}"
echo "  dictionary_languages: ${SAMPLES_DIR}/<lang>/dictionary_languages.yaml (auto)"
echo "  intro:                auto per language if introduction/ exists"
echo "  uv cache:             ${UV_CACHE_DIR}"
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
    --stage1-input "${STAGE1_INPUT}" \
    --discover-extra-fields \
    --overwrite \
    "${EXTRACT_EXTRA_ARGS[@]}"
