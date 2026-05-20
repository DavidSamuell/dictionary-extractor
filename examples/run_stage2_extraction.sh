#!/bin/bash

# Stage-2 structuring. Reads stage-1 TSVs from the slot named by
# --experiment-name and writes structured entries into
#   {entry}/outputs/stage-2/<stage2-experiment>/<stem>/<stem>.tsv
# alongside a single run_config.json per stage-2 experiment.
#
# Defaults: --stage2-experiment-name inherits from --experiment-name.

# Simple case: stage-2 slot mirrors the stage-1 slot.
uv run dictextractor-extract \
    --samples-dir assets/dictionaries/samples-2 \
    --strategy two_stage \
    --stage 2 \
    --overwrite \
    --languages Chepang-English \
    --model gemini/gemini-3.1-pro-preview \
    --stage2-reasoning high \
    --experiment-name gemini3flash_alpha_ocr \
    --discover-extra-fields

# Stage-2 sweep against a fixed stage-1 baseline — stage 1 stays in
# gemini3flash_alpha_ocr; stage 2 lands in its own slot so multiple
# structure-model / reasoning configurations don't overwrite each other:
# uv run dictextractor-extract \
#     --samples-dir assets/dictionaries/samples-2 \
#     --strategy two_stage --stage 2 \
#     --languages Chepang-English \
#     --model gemini/gemini-3.1-pro-preview \
#     --stage2-reasoning high \
#     --experiment-name gemini3flash_alpha_ocr \
#     --stage2-experiment-name pro_highreasoning \
#     --discover-extra-fields
