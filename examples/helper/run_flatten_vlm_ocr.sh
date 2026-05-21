#!/usr/bin/env bash
# Regenerate *_stage1_flat.txt from VLM OCR artifacts (MinerU / Paddle / GLM).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

uv run python scripts/ocr_to_stage1_flat.py \
  --experiment MinerU2.5-Pro \
  --experiment PaddleOCR-VL-1.5 \
  --experiment GLM-OCR
