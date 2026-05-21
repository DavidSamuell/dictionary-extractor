#!/usr/bin/env bash
# Run each OCR model on models/assets; outputs under models/outputs/run_all_<timestamp>.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
OUT="${PROJECT_ROOT}/models/outputs/run_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"

OVIS_EXTRA=(--ovis-no-thinking)

for model in mineru paddleocr glm-ocr ovis; do
  echo "========== ${model} =========="
  "${PYTHON}" models/test_model_inference.py \
    --models "${model}" \
    --pages 0 \
    -o "${OUT}" \
    ${OVIS_EXTRA[@]+"${OVIS_EXTRA[@]}"} \
    || echo "FAILED: ${model}"
done

echo "Outputs: ${OUT}"
ls -la "${OUT}"
