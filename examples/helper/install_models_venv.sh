#!/usr/bin/env bash
# Create per-model virtualenvs for stage-1 VLM OCR (CUDA 12.4 / A100).
#
# MinerU and GLM-OCR need incompatible transformers versions, so each backend
# gets its own env. Venvs live on /data/scratch (symlinked into the project root)
# because project quota is tight once HF model weights are cached.
#
# Usage:
#   uv sync
#   bash examples/helper/install_models_venv.sh
#   bash examples/helper/install_models_venv.sh mineru glmocr paddle   # subset
#   MIGRATE_VENVS=1 bash examples/helper/install_models_venv.sh       # move existing venvs to scratch
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"

TORCH_INDEX="https://download.pytorch.org/whl/cu124"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"
SCRATCH_UV_CACHE="/data/scratch/projects/punim0478/${USER}/uv-cache"
SCRATCH_VENV_ROOT="/data/scratch/projects/punim0478/${USER}/dictionary-extractor-venvs"

# Project quota is tight (~90GB+ with HF weights). Keep uv cache on scratch too.
if [[ -z "${UV_CACHE_DIR:-}" || "${UV_CACHE_DIR}" == /data/projects/punim0478/* ]]; then
  UV_CACHE_DIR="${SCRATCH_UV_CACHE}"
fi
mkdir -p "${UV_CACHE_DIR}" "${SCRATCH_VENV_ROOT}"
export UV_CACHE_DIR
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
echo "UV cache: ${UV_CACHE_DIR}"
echo "Model venvs: ${SCRATCH_VENV_ROOT} (symlinked as ${PROJECT_ROOT}/.venv-*)"

prepare_model_venv() {
  local name="$1"
  local scratch_venv="${SCRATCH_VENV_ROOT}/${name}"
  local project_link="${PROJECT_ROOT}/${name}"

  if [[ -d "${project_link}" && ! -L "${project_link}" ]]; then
    if [[ "${MIGRATE_VENVS:-0}" == "1" ]]; then
      echo "Migrating ${name} to scratch..."
      rm -rf "${scratch_venv}"
      mv "${project_link}" "${scratch_venv}"
      ln -sfn "${scratch_venv}" "${project_link}"
      echo "${scratch_venv}"
      return
    fi
    echo "Using existing ${name} on project disk (export MIGRATE_VENVS=1 to move to scratch)." >&2
    echo "${project_link}"
    return
  fi

  if [[ ! -d "${scratch_venv}" ]]; then
    uv venv "${scratch_venv}"
  fi
  ln -sfn "${scratch_venv}" "${project_link}"
  echo "${scratch_venv}"
}

install_editable_project() {
  local venv_python="$1"
  echo "Installing dictextractor (editable) into $(dirname "$(dirname "${venv_python}")")..."
  uv pip install --python "${venv_python}" -e .
}

install_mineru_venv() {
  local venv_dir venv_python
  venv_dir="$(prepare_model_venv ".venv-mineru")"
  venv_python="${venv_dir}/bin/python"

  echo ""
  echo "=== .venv-mineru (MinerU2.5-Pro, transformers 4.x) ==="
  uv pip install --python "${venv_python}" -U pip
  uv pip install --python "${venv_python}" torch==2.6.0 torchvision==0.21.0 \
    --index-url "${TORCH_INDEX}"
  uv pip install --python "${venv_python}" \
    "mineru-vl-utils[transformers]" accelerate pillow pyyaml pymupdf
  install_editable_project "${venv_python}"
  "${venv_python}" -c "import torch; from mineru_vl_utils import MinerUClient; print('mineru ok', torch.__version__, 'cuda', torch.cuda.is_available())"
}

install_glmocr_venv() {
  local venv_dir venv_python
  venv_dir="$(prepare_model_venv ".venv-glmocr")"
  venv_python="${venv_dir}/bin/python"

  echo ""
  echo "=== .venv-glmocr (GLM-OCR, transformers >=5.9) ==="
  uv pip install --python "${venv_python}" -U pip
  uv pip install --python "${venv_python}" torch==2.6.0 torchvision==0.21.0 \
    --index-url "${TORCH_INDEX}"
  uv pip install --python "${venv_python}" \
    "transformers>=5.9.0" accelerate pillow pyyaml pymupdf
  install_editable_project "${venv_python}"
  "${venv_python}" -c "import torch; import transformers; print('glmocr ok', torch.__version__, 'transformers', transformers.__version__)"
}

install_paddle_venv() {
  local venv_dir venv_python
  venv_dir="$(prepare_model_venv ".venv-paddleocr")"
  venv_python="${venv_dir}/bin/python"

  echo ""
  echo "=== .venv-paddleocr (PaddleOCR-VL-1.5) ==="
  uv pip install --python "${venv_python}" -U pip
  uv pip install --python "${venv_python}" paddlepaddle-gpu==3.2.1 \
    --index-url "${PADDLE_INDEX}"
  uv pip install --python "${venv_python}" -U "paddleocr[doc-parser]" pymupdf pyyaml
  install_editable_project "${venv_python}"
  "${venv_python}" -c "import paddle; print('paddleocr ok', paddle.__version__)"
}

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(mineru glmocr paddle)
fi

for target in "${TARGETS[@]}"; do
  case "${target}" in
    mineru) install_mineru_venv ;;
    glmocr) install_glmocr_venv ;;
    paddle) install_paddle_venv ;;
    *)
      echo "Unknown target: ${target} (choose mineru, glmocr, paddle)" >&2
      exit 1
      ;;
  esac
done

echo ""
echo "Done. Run stage-1 VLM OCR with:"
echo "  bash examples/stage-1/run_stage1_vlm_ocr.sh"
echo "Do not install vllm in these envs (upgrades torch and breaks cu124)."
