#!/usr/bin/env bash
# Stage-1 flat transcription (eval-flat spec v2). Outputs land in
#   {lang}/outputs/stage-1/<experiment-name>/<stem>/<stem>_stage1_flat.txt
# plus <stem>_stage1_raw.json and <stem>_stage1_input.json per page.
#
# Stage-1 flat ablation: alphabet on vs off (--no-alphabet). OCR hint is always off
# (--no-ocr-hint): Stage 1 is the transcription pass; Mathpix at S1 would double up
# with the Stage 1 output that Stage 2 already consumes.
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
    # Canala-English
    # Chepang-English
    # Efik-English
    # Na-English-Chinese-French
    # Reel-English
    # Ritharngu-English
    # Shilluk-English
    # Evenki-Russian
    # Chukchi-Russian
    # Circassian-English-Turkish
    # Nahuatl-French
    # Khmer-English
    # Malay-English
    # Kashmiri-English
    # Greek-English
    # Telugu-English
    # "Iñupiatun Eskimo-English"
    # "Vernacular Syriac-Kurdish_Turkish-English"
    # "Syriac-English"
    # "Tiri-English"
    # Thai-Russian


    Assyrian-English
    Yiddish-English
    Georgian-Russian
    Japanese-English
    Punjabi-English
    Gujarati-English
    Gojri-English-Hindi
    Bengalese-English
    Sanskrit-English
)

### GEMINI (direct API — GEMINI_API_KEY) ###
GEMINI_FLASH_MODEL="gemini/gemini-3-flash-preview"
GEMINI_PRO_MODEL="gemini/gemini-3.1-pro-preview"

run_llm_flat() {
    local model="$1"
    local reasoning="$2"
    shift 2
    if ! uv run dictextractor-extract \
        --strategy two_stage \
        --stage 1 \
        --stage1-mode flat \
        --model "${model}" \
        --stage1-reasoning "${reasoning}" \
        --no-ocr-hint \
        --samples-dir "${SAMPLES_DIR}" \
        --languages "${LANGUAGES[@]}" \
        "$@" \
        "${EXTRACT_EXTRA_ARGS[@]}"; then
        echo "WARNING: experiment failed or was skipped; continuing to next experiment." >&2
    fi
}

run_gemini_flat() {
    local model="$1"
    shift
    run_llm_flat "${model}" low "$@"
}

# --- Flash (alphabet ablation; OCR hint off) ---
run_gemini_flat "${GEMINI_FLASH_MODEL}" \
    --experiment-name gemini3flash_flat_alpha
run_gemini_flat "${GEMINI_FLASH_MODEL}" \
    --no-alphabet \
    --experiment-name gemini3flash_flat_noalpha

# --- Pro (thinking_level=low; alphabet ablation) ---
run_gemini_flat "${GEMINI_PRO_MODEL}" \
    --experiment-name gemini31pro_flat_alpha
run_gemini_flat "${GEMINI_PRO_MODEL}" \
    --no-alphabet \
    --experiment-name gemini31pro_flat_noalpha

### OPENROUTER (OPEN_ROUTER_API_KEY via litellm) ###

export OPENROUTER_MAX_TOKENS="${OPENROUTER_MAX_TOKENS:-32768}"
GPT55_MODEL="openrouter/openai/gpt-5.5"
CLAUDE_OPUS47_MODEL="openrouter/anthropic/claude-opus-4.7"
OPENROUTER_STAGE1_REASONING="${OPENROUTER_STAGE1_REASONING:-low}"

run_openrouter_flat() {
    local model="$1"
    shift
    run_llm_flat "${model}" "${OPENROUTER_STAGE1_REASONING}" "$@"
}

# --- GPT-5.5 (alphabet ablation) ---
run_openrouter_flat "${GPT55_MODEL}" \
    --experiment-name gpt55_flat_alpha
run_openrouter_flat "${GPT55_MODEL}" \
    --no-alphabet \
    --experiment-name gpt55_flat_noalpha

# --- Claude Opus 4.7 (alphabet ablation) ---
run_openrouter_flat "${CLAUDE_OPUS47_MODEL}" \
    --experiment-name claudeopus47_flat_alpha
run_openrouter_flat "${CLAUDE_OPUS47_MODEL}" \
    --no-alphabet \
    --experiment-name claudeopus47_flat_noalpha


### QWEN3-VL (OpenRouter via litellm — OPEN_ROUTER_API_KEY) ###

QWEN3_VL_MODEL="openrouter/qwen/qwen3-vl-235b-a22b-instruct"
run_qwen3vl_flat() {
    if ! uv run dictextractor-extract \
        --strategy two_stage \
        --stage 1 \
        --stage1-mode flat \
        --model "${QWEN3_VL_MODEL}" \
        --no-ocr-hint \
        --samples-dir "${SAMPLES_DIR}" \
        --languages "${LANGUAGES[@]}" \
        "$@" \
        "${EXTRACT_EXTRA_ARGS[@]}"; then
        echo "WARNING: experiment failed or was skipped; continuing to next experiment." >&2
    fi
}

run_qwen3vl_flat --experiment-name qwen3vl235_flat_alpha
run_qwen3vl_flat --no-alphabet --experiment-name qwen3vl235_flat_noalpha

### Mathpix-OCR experiment (API convert → markdown + flat via markdown_to_flat) ###
run_mathpix() {
    local lang
    local -a mathpix_convert_extra=()
    local arg
    for arg in "${EXTRACT_EXTRA_ARGS[@]}"; do
        if [[ "${arg}" == "--overwrite" ]]; then
            mathpix_convert_extra+=(--overwrite-files --force)
        fi
    done

    echo ""
    echo "============================================================"
    echo " Mathpix OCR: Mathpix-OCR (--strategy mathpix_ocr)"
    echo "  convert: snippets/*.{pdf,png,...} → mathpix/{stem}.md + {stem}.lines.json"
    echo "  extract: mathpix/*.md → stage-1/Mathpix-OCR/{stem}/output.md + *_stage1_flat.txt"
    echo "============================================================"

    for lang in "${LANGUAGES[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo " Mathpix-OCR: ${lang}"
        echo "------------------------------------------------------------"
        if ! uv run dictextractor-mathpix-convert \
            --samples-dir "${SAMPLES_DIR}" \
            --languages "${lang}" \
            "${mathpix_convert_extra[@]}"; then
            echo "WARNING: Mathpix convert failed for ${lang}; skipping extract." >&2
            continue
        fi
        if ! uv run dictextractor-extract \
            --strategy mathpix_ocr \
            --samples-dir "${SAMPLES_DIR}" \
            --stage 1 \
            --no-ocr-hint \
            --no-alphabet \
            --experiment-name Mathpix-OCR \
            --languages "${lang}" \
            "${EXTRACT_EXTRA_ARGS[@]}"; then
            echo "WARNING: Mathpix-OCR failed for ${lang}; continuing." >&2
        fi
    done
}

run_mathpix

### VLM OCR (separate venvs per model — outputs are flat via vlm_ocr adapter) ###

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export PATH="${HOME}/.local/bin:${PATH}"
INSTALL_SCRIPT="examples/helper/install_models_venv.sh"
# MinerU: vllm (default, faster) uses .venv-mineru-vllm; transformers uses .venv-mineru
VLM_BACKEND="${VLM_BACKEND:-vllm}"

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
    local venv_dir python
    venv_dir="$(venv_for_model "${key}")"
    python="${venv_dir}/bin/python"
    if [[ ! -x "${python}" ]]; then
        echo "Missing ${venv_dir} — run: bash ${INSTALL_SCRIPT}" >&2
        exit 1
    fi
    echo "${python}"
}

run_vlm() {
    local key="$1"
    shift
    local -a experiments=()
    local -a vlm_extra=()
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == --* ]]; then
            vlm_extra+=("${arg}")
        else
            experiments+=("${arg}")
        fi
    done
    if ((${#experiments[@]} == 0)); then
        echo "run_vlm: at least one experiment name required" >&2
        return 1
    fi

    if [[ "${key}" == "mineru2.5-pro" && "${VLM_BACKEND}" == "vllm" ]]; then
        vlm_extra=(--vlm-backend vllm "${vlm_extra[@]}")
    fi
    # Paddle: native in-process VLM by default (--no-paddle-auto-vllm-server).
    # Set PADDLE_VL_REC_SERVER_URL to use an external vLLM server instead.
    if [[ "${key}" == "paddleocr-vl-1.5" ]]; then
        vlm_extra=(--no-paddle-auto-vllm-server "${vlm_extra[@]}")
        if [[ -n "${PADDLE_VL_REC_SERVER_URL:-}" ]]; then
            vlm_extra=(
                --paddle-vl-rec-backend vllm-server
                --paddle-vl-rec-server-url "${PADDLE_VL_REC_SERVER_URL}"
                "${vlm_extra[@]}"
            )
        fi
    fi
    # GLM-OCR: in-process transformers backend (no auto vLLM server required).
    if [[ "${key}" == "glm-ocr" ]]; then
        vlm_extra=(
            --glm-backend "${GLM_BACKEND:-transformers}"
            --no-glm-auto-vllm-server
            "${vlm_extra[@]}"
        )
    fi

    local python
    python="$(require_model_venv "${key}")"

    echo ""
    echo "============================================================"
    echo " VLM OCR: ${experiments[*]} (--vlm-model ${key}, one model load)"
    echo " Python:  ${python}"
    echo "============================================================"
    local -a experiment_args=()
    for arg in "${experiments[@]}"; do
        experiment_args+=(--experiment-name "${arg}")
    done
    if ! "${python}" -m dictextractor.cli.extract \
        --strategy vlm_ocr \
        --vlm-model "${key}" \
        --samples-dir "${SAMPLES_DIR}" \
        --stage 1 \
        --no-ocr-hint \
        "${experiment_args[@]}" \
        --languages "${LANGUAGES[@]}" \
        "${vlm_extra[@]}" \
        "${EXTRACT_EXTRA_ARGS[@]}"; then
        echo "WARNING: VLM run ${key} (${experiments[*]}) failed or was skipped; continuing." >&2
    fi
}

# Document parsers (CPU/GPU depending on model; one process per vlm-model)
run_vlm mineru2.5-pro MinerU2.5-Pro
run_vlm paddleocr-vl-1.5 PaddleOCR-VL-1.5
# GLM-OCR: both alphabet ablations in one process; *_noalpha slots disable alphabet
run_vlm glm-ocr GLM-OCR-flat_alpha GLM-OCR-flat_noalpha
