#!/usr/bin/env bash
# Start local Label Studio, then create/update projects.

set -euo pipefail

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

export LABEL_STUDIO_TOKEN="${LS_ACCESS_TOKEN:?Set LS_ACCESS_TOKEN in .env}"
export LABEL_STUDIO_AUTH_SCHEME="Bearer"

LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(pwd)/.label-studio-renders" \
uv run label-studio --port 8080 &

# Wait for Label Studio to finish starting up
sleep 5

uv run python label-studio/setup.py \
    --samples-dir assets/dictionaries/samples-2 \
    --render-dir .label-studio-renders \
    --overwrite \
    --connect-local-storage
