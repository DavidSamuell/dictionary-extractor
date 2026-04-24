#!/bin/bash

# Process every language subfolder under samples-2/ in one go.
# Outputs land in {entry}/outputs/stage-1/.
# uv run dictextractor-extract \
#     --strategy two_stage \
#     --model gemini/gemini-3-flash-preview \
#     --samples-dir assets/dictionaries/samples-2 \
#     --stage 1

# Process only specific language subfolders, with cv2 preprocessing on:
uv run dictextractor-extract \
    --strategy two_stage \
    --model gemini/gemini-3-flash-preview \
    --samples-dir assets/dictionaries/samples-2 \
    --stage 1 \
    --overwrite
