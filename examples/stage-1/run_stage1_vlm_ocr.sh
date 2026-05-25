#!/usr/bin/env bash
# Stage-1 specialized VLM OCR on the LANGUAGES subset below (same list as
# examples/stage-1/run_stage1_extraction.sh). Edit LANGUAGES to change scope.
#
# Prerequisites:
#   uv sync
#   bash examples/helper/install_models_venv.sh
#
# Each model runs in its own venv (MinerU vs GLM-OCR need incompatible
# transformers). Re-runs skip pages that already have model output on disk.
# Pass --overwrite to force reprocessing. Resume skips only when artifacts have
# real content (MinerU: non-empty content.json or output.md; Paddle: non-empty
# parsing_res_list; GLM: non-empty output.txt / result.json). Empty runs re-run.
#
# Outputs per language:
#   {lang}/outputs/stage-1/MinerU2.5-Pro/page_N/...
#   {lang}/outputs/stage-1/PaddleOCR-VL-1.5/page_N/...
#   {lang}/outputs/stage-1/GLM-OCR/page_N/...

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export PATH="${HOME}/.local/bin:${PATH}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"
INSTALL_SCRIPT="examples/helper/install_models_venv.sh"
VLM_BACKEND="${VLM_BACKEND:-vllm}"

# Subset of language subfolders (keep in sync with run_stage1_extraction.sh).
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
LANG_ARGS=(--languages "${LANGUAGES[@]}")

# Optional extra args forwarded to dictextractor.cli.extract (e.g. --overwrite).
EXTRACT_EXTRA_ARGS=("$@")

venv_for_model() {
  case "$1" in
    mineru2.5-pro)
      if [[ "${VLM_BACKEND}" == "vllm" ]]; then
        echo "${PROJECT_ROOT}/.venv-mineru-vllm"
      else
        echo "${PROJECT_ROOT}/.venv-mineru"
      fi
      ;;
    paddleocr-vl-1.5) echo "${PROJECT_ROOT}/.venv-paddleocr" ;;
    glm-ocr) echo "${PROJECT_ROOT}/.venv-glmocr" ;;
    *)
      echo "Unknown --vlm-model: $1" >&2
      return 1
      ;;
  esac
}

require_model_venv() {
  local key="$1"
  local venv_dir
  venv_dir="$(venv_for_model "${key}")"
  local python="${venv_dir}/bin/python"
  if [[ ! -x "${python}" ]]; then
    echo "Missing ${venv_dir} — run: bash ${INSTALL_SCRIPT}" >&2
    exit 1
  fi
  echo "${python}"
}

run_vlm() {
  local key="$1"
  local experiment="$2"
  local python
  python="$(require_model_venv "${key}")"
  local -a vlm_extra=()
  if [[ "${key}" == "mineru2.5-pro" && "${VLM_BACKEND}" == "vllm" ]]; then
    vlm_extra+=(--vlm-backend vllm)
  fi
  if [[ "${key}" == "paddleocr-vl-1.5" && -n "${PADDLE_VL_REC_SERVER_URL:-}" ]]; then
    vlm_extra+=(
      --no-paddle-auto-vllm-server
      --paddle-vl-rec-backend vllm-server
      --paddle-vl-rec-server-url "${PADDLE_VL_REC_SERVER_URL}"
    )
  fi

  echo ""
  echo "============================================================"
  echo " VLM OCR: ${experiment} (--vlm-model ${key}, backend ${VLM_BACKEND})"
  echo " Python:  ${python}"
  echo "============================================================"
  "${python}" -m dictextractor.cli.extract \
    --strategy vlm_ocr \
    --vlm-model "${key}" \
    --samples-dir "${SAMPLES_DIR}" \
    --stage 1 \
    --experiment-name "${experiment}" \
    "${LANG_ARGS[@]}" \
    "${vlm_extra[@]}" \
    "${EXTRACT_EXTRA_ARGS[@]}" \
    || echo "FAILED: ${experiment}"
}

# run_vlm mineru2.5-pro MinerU2.5-Pro
run_vlm paddleocr-vl-1.5 PaddleOCR-VL-1.5
run_vlm glm-ocr GLM-OCR

echo ""
echo "Done. Check outputs under each language:"
echo "  ${SAMPLES_DIR}/<Lang-Pair>/outputs/stage-1/{MinerU2.5-Pro,PaddleOCR-VL-1.5,GLM-OCR}/"
