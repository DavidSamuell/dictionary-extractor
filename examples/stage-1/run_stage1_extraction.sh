#!/bin/bash

# Stage-1 transcription only.  Outputs land in
#   {entry}/outputs/stage-1/<experiment-name>/<stem>/<stem>_stage1.tsv
# alongside a single run_config.json per experiment.
#
# Pass --experiment-name to keep ablation runs separate. Combine with
# --no-alphabet and/or --no-ocr-hint to control which Stage-1 inputs are
# in scope. Re-runs without --overwrite resume per-experiment.

# Subset of language subfolders to process across every experiment below.
# Edit this list (or comment out the `--languages "${LANGUAGES[@]}"` line on
# any invocation) to run against the full samples root.
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
    # Evenki-Russian
    # Chukchi-Russian
    Circassian-English-Turkish
    Yiddish-English
    Nahuatl-French
)

# # Baseline: alphabet + OCR hint + gemini flash
uv run dictextractor-extract \
    --strategy two_stage \
    --model gemini/gemini-3-flash-preview \
    --samples-dir assets/dictionaries/samples \
    --stage 1 \
    --languages "${LANGUAGES[@]}" \
    --experiment-name gemini3flash_alpha_ocr

# Sweep example — ablations against the baseline (uncomment what you need):
uv run dictextractor-extract \
    --strategy two_stage --stage 1 \
    --model gemini/gemini-3-flash-preview \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --no-alphabet \
    --experiment-name gemini3flash_noalpha_ocr

uv run dictextractor-extract \
    --strategy two_stage --stage 1 \
    --model gemini/gemini-3-flash-preview \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --no-ocr-hint \
    --experiment-name gemini3flash_alpha_noocr

uv run dictextractor-extract \
    --strategy two_stage --stage 1 \
    --model gemini/gemini-3-flash-preview \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --no-alphabet --no-ocr-hint \
    --experiment-name gemini3flash_bare
#
# # Different model with the full input bundle:
# uv run dictextractor-extract \
#     --strategy two_stage --stage 1 \
#     --model gemini/gemini-3.1-pro-preview \
#     --samples-dir assets/dictionaries/samples \
#     --languages "${LANGUAGES[@]}" \
#     --experiment-name gemini31pro_alpha_ocr

# Single language with custom guidelines:
# uv run dictextractor-extract \
#     --strategy two_stage --stage 1 \
#     --model gemini/gemini-3-flash-preview \
#     --samples-dir assets/dictionaries/samples \
#     --languages Sisaali-English \
#     --stage-1-guides assets/dictionaries/samples-2/Sisaali-English/guides.md \
#     --experiment-name gemini3flash_alpha_ocr_guides \
#     --overwrite
