#!/usr/bin/env bash
# One-time (or repeat) Shilluk gold pull from Label Studio, including pages with
# no human post-edit (100% OCR accepted). Uses import-time task prefill for those.
#
# After export, regenerate flat gold:
#   bash examples/evaluation/run_flatten_gold_label_s1.sh --languages Shilluk-English

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_sync_label_studio.sh" \
    --languages Shilluk-English \
    --include-prefill \
    "$@"
