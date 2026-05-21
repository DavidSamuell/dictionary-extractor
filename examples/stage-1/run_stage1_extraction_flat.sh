#!/usr/bin/env bash
# Stage-1 flat transcription (eval-flat spec v2). Outputs land in
#   {lang}/outputs/stage-1/<experiment-name>/<stem>/<stem>_stage1_flat.txt
# plus <stem>_stage1_raw.json and <stem>_stage1_input.json per page.
#
# Same ablation matrix as run_stage1_extraction.sh, with --stage1-mode flat.
# Flat mode writes *_stage1_flat.txt. Stage 2 can consume them via --stage1-input auto|flat.
#
# Evaluate preds: examples/evaluation/run_stage1_eval_flat.sh
#
# Extra args ("$@") go to dictextractor-extract only, e.g. --overwrite

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"
EXTRACT_EXTRA_ARGS=("$@")

LANGUAGES=(
    Assyrian-English
    Canala-English
    Chepang-English
    Chung-English
    Efik-English
    Na-English-Chinese
    Reel-English
    Ritharngu-English
    Shilluk-English
    Evenki-Russian
    Chukchi-Russian
    Circassian-English-Turkish
    Yiddish-English
    Nahuatl-French
)

# Baseline: alphabet + OCR hint + gemini flash (flat)
uv run dictextractor-extract \
    --strategy two_stage \
    --stage 1 \
    --stage1-mode flat \
    --model gemini/gemini-3-flash-preview \
    --samples-dir "${SAMPLES_DIR}" \
    --languages "${LANGUAGES[@]}" \
    --experiment-name gemini3flash_flat_alpha_ocr \
    "${EXTRACT_EXTRA_ARGS[@]}"

# Ablations (same toggles as column runs)
uv run dictextractor-extract \
    --strategy two_stage \
    --stage 1 \
    --stage1-mode flat \
    --model gemini/gemini-3-flash-preview \
    --samples-dir "${SAMPLES_DIR}" \
    --languages "${LANGUAGES[@]}" \
    --no-alphabet \
    --experiment-name gemini3flash_flat_noalpha_ocr \
    "${EXTRACT_EXTRA_ARGS[@]}"

uv run dictextractor-extract \
    --strategy two_stage \
    --stage 1 \
    --stage1-mode flat \
    --model gemini/gemini-3-flash-preview \
    --samples-dir "${SAMPLES_DIR}" \
    --languages "${LANGUAGES[@]}" \
    --no-ocr-hint \
    --experiment-name gemini3flash_flat_alpha_noocr \
    "${EXTRACT_EXTRA_ARGS[@]}"

uv run dictextractor-extract \
    --strategy two_stage \
    --stage 1 \
    --stage1-mode flat \
    --model gemini/gemini-3-flash-preview \
    --samples-dir "${SAMPLES_DIR}" \
    --languages "${LANGUAGES[@]}" \
    --no-alphabet \
    --no-ocr-hint \
    --experiment-name gemini3flash_flat_bare \
    "${EXTRACT_EXTRA_ARGS[@]}"

# Different model (uncomment):
# uv run dictextractor-extract \
#     --strategy two_stage --stage 1 --stage1-mode flat \
#     --model gemini/gemini-3.1-pro-preview \
#     --samples-dir "${SAMPLES_DIR}" \
#     --languages "${LANGUAGES[@]}" \
#     --experiment-name gemini31pro_flat_alpha_ocr \
#     "${EXTRACT_EXTRA_ARGS[@]}"

# Single language + guides (uncomment):
# uv run dictextractor-extract \
#     --strategy two_stage --stage 1 --stage1-mode flat \
#     --model gemini/gemini-3-flash-preview \
#     --samples-dir "${SAMPLES_DIR}" \
#     --languages Sisaali-English \
#     --stage-1-guides assets/dictionaries/samples-2/Sisaali-English/guides.md \
#     --experiment-name gemini3flash_flat_alpha_ocr_guides \
#     --overwrite
