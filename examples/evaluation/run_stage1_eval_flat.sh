#!/usr/bin/env bash
#
# eval-flat: flat transcription vs *_stage1_GOLD_flat.txt (spec v2).
#
# Run:  bash examples/evaluation/run_stage1_eval_flat.sh
#
# Uncomment ONE uv run block below (do not combine --include-vlm-ocr with
# --experiment-name for OCR only — that whitelists to OCR and skips Gemini).
#
# Outputs under evaluations/stage1_flat_eval/:
#   stage1_flat_eval_detailed.csv
#   stage1_flat_eval_summary.csv
#   stage1_flat_eval_cache.json
#   <experiment>/stage1_flat_evaluation_report.*

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== Stage 1 eval-flat (Gemini flat ablations + VLM OCR) ==="

# --- Active: all gemini3flash_flat_* + MinerU / Paddle / GLM ---
uv run dictextractor-eval-flat \
  --samples-dir assets/dictionaries/samples \
  --include-vlm-ocr \
  -o evaluations/stage1_flat_eval \
  --overwrite

# --- OCR only (recomputes VLM; CSVs still merge cached Gemini from cache) ---
# uv run dictextractor-eval-flat \
#   --samples-dir assets/dictionaries/samples \
#   --experiment-name MinerU2.5-Pro \
#   --experiment-name PaddleOCR-VL-1.5 \
#   --experiment-name GLM-OCR \
#   -o evaluations/stage1_flat_eval \
#   --overwrite

# --- Explicit experiment list (same set as --include-vlm-ocr; no --include-vlm-ocr) ---
# uv run dictextractor-eval-flat \
#   --samples-dir assets/dictionaries/samples \
#   --experiment-name gemini3flash_flat_alpha_ocr \
#   --experiment-name gemini3flash_flat_noalpha_ocr \
#   --experiment-name gemini3flash_flat_alpha_noocr \
#   --experiment-name gemini3flash_flat_bare \
#   --experiment-name MinerU2.5-Pro \
#   --experiment-name PaddleOCR-VL-1.5 \
#   --experiment-name GLM-OCR \
#   -o evaluations/stage1_flat_eval \
#   --overwrite
