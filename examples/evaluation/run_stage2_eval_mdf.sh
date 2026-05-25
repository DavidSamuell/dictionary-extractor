#!/bin/bash
# Evaluate Stage 2 MDF predictions against gold on benchmark pages.
#
# Defaults: record_threshold=0.6, line_threshold=0.7
#
# Discovers every experiment under outputs/stage-2/ that has a matching pred
# for a page with stage-2-gold labels (Chukchi, Evenki, Nahuatl, Na-English-Chinese-French).
#
# Outputs:
#   evaluations/stage2_mdf_eval/stage2_mdf_eval_summary.csv
#   evaluations/stage2_mdf_eval/<experiment>/stage2_mdf_evaluation_report.{txt,json}
#
# Restrict languages or experiments via CLI flags, e.g.:
#   bash examples/evaluation/run_stage2_eval_mdf.sh  # all experiments (default)
#   uv run dictextractor-eval-stage2-mdf ... --languages Chukchi-Russian --experiment-name gemini31pro_high_mdf_intro_notoolbox

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_DIR="assets/dictionaries/samples"
OUTPUT_DIR="evaluations/stage2_mdf_eval"

echo "=== Stage 2 MDF evaluation (all experiments) ==="
echo ""

uv run dictextractor-eval-stage2-mdf \
  --marker-sub-list assets/evaluation/mdf_marker_sub_list.yaml \
  --samples-dir "$SAMPLES_DIR" \
  --all-experiments \
  -o "$OUTPUT_DIR"
