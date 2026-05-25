#!/usr/bin/env bash
# Export submitted annotations from the supervisor's VM Label Studio instance
# into the experiment-agnostic gold tree:
#   assets/dictionaries/samples/<lang>/outputs/stage-1-gold/<page>/<page>_stage1_GOLD.tsv
#
# Circassian-English-Turkish is always skipped (local gold only); see
# EXCLUDED_LANGUAGES in scripts/export_label_studio_gold.py.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
fi

VM_HOST="216.158.235.114"
LS_URL="http://${VM_HOST}:8080"
SAMPLES_DIR="assets/dictionaries/samples"

export LABEL_STUDIO_TOKEN="${VM_LS_TOKEN:?Set VM_LS_TOKEN in .env}"
export LABEL_STUDIO_AUTH_SCHEME="${VM_LS_AUTH_SCHEME:-PAT}" # PAT, Bearer, Token, or auto

uv run python scripts/export_label_studio_gold.py \
    --samples-dir "${SAMPLES_DIR}" \
    --ls-url "${LS_URL}" \
    "$@"

uv run python scripts/flatten_stage1_gold.py "$@"
