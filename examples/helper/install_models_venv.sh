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

install_mineru_vllm_venv() {
  local venv_dir venv_python
  venv_dir="$(prepare_model_venv ".venv-mineru-vllm")"
  venv_python="${venv_dir}/bin/python"

  echo ""
  echo "=== .venv-mineru-vllm (MinerU2.5-Pro, vLLM in-process) ==="
  echo "Note: vLLM pulls its own torch build (typically newer cu128 wheels)."
  uv pip install --python "${venv_python}" -U pip
  uv pip install --python "${venv_python}" \
    "mineru-vl-utils[vllm]" accelerate pillow pyyaml pymupdf
  install_editable_project "${venv_python}"
  "${venv_python}" -c "import torch; from vllm import LLM; from mineru_vl_utils import MinerUClient; print('mineru-vllm ok', torch.__version__, 'cuda', torch.cuda.is_available())"
}

install_glmocr_vllm_venv() {
  local venv_dir venv_python
  venv_dir="$(prepare_model_venv ".venv-glmocr-vllm")"
  venv_python="${venv_dir}/bin/python"

  echo ""
  echo "=== .venv-glmocr-vllm (GLM-OCR vLLM server + client) ==="
  uv pip install --python "${venv_python}" -U pip
  uv pip install --python "${venv_python}" \
    "transformers>=5.9.0" "vllm>=0.19.0" httpx accelerate pillow pyyaml pymupdf
  install_editable_project "${venv_python}"
  "${venv_python}" -c "import vllm; import httpx; print('glmocr-vllm ok', vllm.__version__)"
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

install_paddle_vllm_server_venv() {
  local venv_dir venv_python
  venv_dir="$(prepare_model_venv ".venv-paddle-vllm-server")"
  venv_python="${venv_dir}/bin/python"

  echo ""
  echo "=== .venv-paddle-vllm-server (Paddle GenAI vLLM server for PaddleOCR-VL) ==="
  echo "Tip: module load CUDA/12.2.0 && export CUDA_HOME=\$EBROOTCUDA before install if flash-attn build fails."
  uv pip install --python "${venv_python}" -U pip
  uv pip install --python "${venv_python}" "paddleocr[doc-parser]" pymupdf pyyaml
  uv pip install --python "${venv_python}" torch==2.8.0
  if ! uv pip install --python "${venv_python}" flash-attn==2.8.3 --no-build-isolation; then
    echo "flash-attn build failed — set CUDA_HOME and retry this target." >&2
    exit 1
  fi
  uv pip install --python "${venv_python}" \
    einops "transformers<5.0.0" uvloop "vllm==0.10.2" xformers
  install_editable_project "${venv_python}"
  "${venv_python}" -c "
from paddlex.utils.deps import is_genai_engine_plugin_available
assert is_genai_engine_plugin_available('vllm-server'), 'genai-vllm-server plugin missing'
print('paddle-vllm-server ok')
"
}

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(mineru glmocr paddle)
fi

for target in "${TARGETS[@]}"; do
  case "${target}" in
    mineru) install_mineru_venv ;;
    mineru-vllm) install_mineru_vllm_venv ;;
    glmocr) install_glmocr_venv ;;
    glmocr-vllm) install_glmocr_vllm_venv ;;
    paddle) install_paddle_venv ;;
    paddle-vllm-server) install_paddle_vllm_server_venv ;;
    *)
      echo "Unknown target: ${target} (choose mineru, mineru-vllm, glmocr, glmocr-vllm, paddle, paddle-vllm-server)" >&2
      exit 1
      ;;
  esac
done

echo ""
echo "Done. Run stage-1 VLM OCR with:"
echo "  VLM_BACKEND=vllm bash examples/stage-1/run_stage1_extraction_flat.sh"
echo "  bash examples/stage-1/run_stage1_vlm_ocr.sh"
echo ""
echo "MinerU vLLM: bash examples/helper/install_models_venv.sh mineru-vllm"
echo "GLM-OCR vLLM: bash examples/helper/install_models_venv.sh glmocr-vllm"
echo "Paddle vLLM server (auto-started during paddleocr-vl-1.5 runs):"
echo "  bash examples/helper/install_models_venv.sh paddle-vllm-server"
echo "Or point to an external server: export PADDLE_VL_REC_SERVER_URL=http://127.0.0.1:8765/v1"
echo "Do not install vllm into .venv-mineru (upgrades torch and breaks cu124)."
