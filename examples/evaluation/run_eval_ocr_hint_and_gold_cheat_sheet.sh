#!/usr/bin/env bash
# Run Stage-1 (stage-1-ocr) and Stage-2 (*_goldcheat) evaluations for the
# additional per-language best experiments.
#
# Outputs:
#   evaluations/stage_1_ocr_hint/
#   evaluations/stage_2_gold_cheat_sheet/
#
# Usage:
#   bash examples/evaluation/run_eval_ocr_hint_and_gold_cheat_sheet.sh
#   bash examples/evaluation/run_eval_ocr_hint_and_gold_cheat_sheet.sh --overwrite --workers 14

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

EVAL_DIR="$(dirname "${BASH_SOURCE[0]}")"

echo "============================================================"
echo " Additional experiment evaluation (Stage 1 + Stage 2)"
echo "============================================================"

bash "${EVAL_DIR}/run_stage1_eval_ocr_hint.sh" "$@"
echo ""
bash "${EVAL_DIR}/run_stage2_eval_gold_cheat_sheet.sh" "$@"

echo ""
echo "Done. Summaries:"
echo "  evaluations/stage_1_ocr_hint/stage1_flat_eval_summary.csv"
echo "  evaluations/stage_2_gold_cheat_sheet/stage2_mdf_eval_summary.csv"
