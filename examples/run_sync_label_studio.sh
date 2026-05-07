#!/usr/bin/env bash
# Export submitted annotations from the supervisor's VM Label Studio instance
# back into *_stage1_GOLD.tsv files under assets/dictionaries/samples-2/.

set -euo pipefail

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

VM_HOST="216.158.235.114"
LS_URL="http://${VM_HOST}:8080"
SAMPLES_DIR="assets/dictionaries/samples-2"

export LABEL_STUDIO_TOKEN="${VM_LS_TOKEN:?Set VM_LS_TOKEN in .env}"
export LABEL_STUDIO_AUTH_SCHEME="${VM_LS_AUTH_SCHEME:-PAT}" # PAT, Bearer, Token, or auto

uv run python scripts/export_label_studio_gold.py \
    --samples-dir "${SAMPLES_DIR}" \
    --ls-url "${LS_URL}"