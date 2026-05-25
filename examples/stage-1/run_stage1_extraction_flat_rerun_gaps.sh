#!/usr/bin/env bash
# Re-run Stage-1 flat experiments that were missing or empty after cleanup.
#
# Gaps filled by this script:
#   Syriac-English — claudeopus47_flat_alpha, claudeopus47_flat_noalpha
#     (Claude via OpenRouter rejects images >5 MB; see image.py compression)
#
# Usage:
#   bash examples/stage-1/run_stage1_extraction_flat_rerun_gaps.sh
#   bash examples/stage-1/run_stage1_extraction_flat_rerun_gaps.sh --overwrite

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"
EXTRACT_EXTRA_ARGS=("$@")

CLAUDE_OPUS47_MODEL="openrouter/anthropic/claude-opus-4.7"

export OPENROUTER_MAX_TOKENS="${OPENROUTER_MAX_TOKENS:-16384}"
OPENROUTER_STAGE1_REASONING="${OPENROUTER_STAGE1_REASONING:-low}"

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
        "$@" \
        "${EXTRACT_EXTRA_ARGS[@]}"; then
        echo "WARNING: experiment failed or was skipped; continuing." >&2
    fi
}

SYRIAC=(Syriac-English)

echo ""
echo "============================================================"
echo " Syriac-English — Claude Opus gaps (2 experiments)"
echo "============================================================"

run_llm_flat "${CLAUDE_OPUS47_MODEL}" "${OPENROUTER_STAGE1_REASONING}" \
    --languages "${SYRIAC[@]}" \
    --experiment-name claudeopus47_flat_alpha

run_llm_flat "${CLAUDE_OPUS47_MODEL}" "${OPENROUTER_STAGE1_REASONING}" \
    --languages "${SYRIAC[@]}" \
    --no-alphabet \
    --experiment-name claudeopus47_flat_noalpha

echo ""
echo "Done."
