#!/usr/bin/env bash
# Populate the remote Label Studio instance on the supervisor's VM.
#
# Prerequisites:
#   - Label Studio is already running on the VM (repo cloned, label-studio started)
#   - VM_REPO_ROOT is the path where the repo was cloned on the VM
#
# Workflow:
#   1. setup.py renders PNGs locally into .label-studio-renders/ and creates
#      projects on the remote Label Studio via API.
#   2. Commit and push .label-studio-renders/ so the VM can git pull to get images.
#
# Usage:
#   bash examples/run_label_studio_vm.sh

set -euo pipefail

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# ── Configuration ────────────────────────────────────────────────────────────
VM_HOST="216.158.235.114" 
VM_REPO_ROOT="/var/www/app/dictionary-extractor"   # where the repo is cloned on the VM

LS_URL="http://${VM_HOST}:8080"
export LABEL_STUDIO_TOKEN="${VM_LS_TOKEN:?Set VM_LS_TOKEN in .env}"
export LABEL_STUDIO_AUTH_SCHEME="${VM_LS_AUTH_SCHEME:-PAT}" # PAT, Bearer, Token, or auto

LOCAL_RENDER_DIR=".label-studio-renders"
SAMPLES_DIR="assets/dictionaries/samples-2"
# ─────────────────────────────────────────────────────────────────────────────

echo "==> Rendering PNGs and creating Label Studio projects on ${LS_URL} ..."
uv run python label-studio/setup.py \
    --samples-dir  "${SAMPLES_DIR}" \
    --ls-url       "${LS_URL}" \
    --render-dir   "${LOCAL_RENDER_DIR}" \
    --storage-root "${VM_REPO_ROOT}/${LOCAL_RENDER_DIR}" \
    --connect-local-storage \
    # --languages Sanskrit-English \
    # --overwrite

echo ""
echo "==> Projects created. Now commit and push the rendered images:"
echo "    git add .label-studio-renders"
echo "    git commit -m 'chore: add rendered page images for Label Studio'"
echo "    git push"
echo ""
echo "    Then on the VM: git pull"
echo "==> Done. Open ${LS_URL} to verify projects."
