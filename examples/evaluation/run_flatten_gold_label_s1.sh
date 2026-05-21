#!/usr/bin/env bash
# Generate flatten gold (spec v2) for stage-1 eval-flat.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
uv run python scripts/flatten_stage1_gold.py "$@"
