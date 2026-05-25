#!/usr/bin/env bash
# Evaluate Stage-2 gold field-cheat-sheet runs (*_goldcheat under outputs/stage-2/).
#
# Produced by: examples/stage-2/run_stage2_gold_cheatsheet.sh
#
# Outputs:
#   evaluations/stage_2_gold_cheat_sheet/stage2_mdf_eval_summary.csv
#   evaluations/stage_2_gold_cheat_sheet/<experiment>/stage2_mdf_evaluation_report.*
#
# Usage:
#   bash examples/evaluation/run_stage2_eval_gold_cheat_sheet.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluations/stage_2_gold_cheat_sheet}"
MARKER_SUB_LIST="${MARKER_SUB_LIST:-assets/evaluation/mdf_marker_sub_list.yaml}"

LANGUAGES=(
    "Iñupiatun Eskimo-English"
    Kashmiri-English
    Tiri-English
    Evenki-Russian
    Nahuatl-French
    Na-English-Chinese-French
)

mapfile -t EXPERIMENTS < <(
    uv run python3 - "${SAMPLES_DIR}" <<'PY'
import sys
from pathlib import Path

samples = Path(sys.argv[1])
names = sorted(
    {
        p.parent.name
        for p in samples.glob("*/outputs/stage-2/*_goldcheat/run_config.json")
    }
)
if not names:
    raise SystemExit("No *_goldcheat stage-2 experiments found on disk.")
for name in names:
    print(name)
PY
)

echo "=== Stage 2 MDF eval (gold cheat sheet experiments) ==="
echo "Languages (${#LANGUAGES[@]}): ${LANGUAGES[*]}"
echo "Experiments (${#EXPERIMENTS[@]}): ${EXPERIMENTS[*]}"
echo "Output: ${OUTPUT_DIR}/"
echo ""

EXP_ARGS=()
for exp in "${EXPERIMENTS[@]}"; do
    EXP_ARGS+=(--experiment-name "${exp}")
done

uv run dictextractor-eval-stage2-mdf \
  --marker-sub-list "${MARKER_SUB_LIST}" \
  --samples-dir "${SAMPLES_DIR}" \
  --languages "${LANGUAGES[@]}" \
  "${EXP_ARGS[@]}" \
  -o "${OUTPUT_DIR}" \
  "$@"
