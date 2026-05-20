#!/bin/bash
#
# Evaluate every stage-1 experiment present under a samples root.
#
# Outputs:
#   <OUTPUT_DIR>/stage1_eval_detailed.csv   — one row per (experiment, page)
#   <OUTPUT_DIR>/stage1_eval_summary.csv     — per (experiment, language)
#   <OUTPUT_DIR>/stage1_eval_cache.json      — incremental cache (mtime+size + alignment params)
#   <OUTPUT_DIR>/<experiment>/             — drill-down CSVs + JSON + text report
#
# Pass --languages to recompute only those language folders; other pages load from
# the cache. Use --overwrite to force recomputation for the current --languages
# (and --experiment-name) slice. Omit --languages to evaluate every language but
# still reuse cache where files are unchanged.
#
# Requires the new layout (gold at outputs/stage-1-gold/, predictions at
# outputs/stage-1/<experiment>/).  Run scripts/migrate_stage1_layout.sh once
# if you are upgrading from the pre-experiment layout.
#

set -euo pipefail

SAMPLES_DIR="assets/dictionaries/samples"
OUTPUT_DIR="evaluations/stage1_eval"

# Subset of language subfolders to evaluate. Comment out the `--languages`
# line below to evaluate every language with stage-1 gold.
LANGUAGES=(
    Canala-English
    Chepang-English
    Efik-English
    Na-English-Chinese
    Reel-English
    Ritharngu-English
    Shilluk-English
    Evenki-Russian
    Chukchi-Russian
)

echo "=== Running Stage 1 evaluation across all experiments ==="
echo ""

uv run dictextractor-eval-s1 \
    --samples-dir "$SAMPLES_DIR" \
    --languages "${LANGUAGES[@]}" \
    --all-experiments \
    --metrics minimal \
    -o "$OUTPUT_DIR"

# Slimmer comparison CSVs (TextEdit, GCER, WER, typography_f1, ReadOrderEdit):
#     --metrics minimal \

# Restrict to specific experiments instead:
# uv run dictextractor-eval-s1 \
#     --samples-dir "$SAMPLES_DIR" \
#     --languages "${LANGUAGES[@]}" \
#     --experiment-name gemini3flash_alpha_ocr \
#     --experiment-name gemini3flash_bare \
#     -o "$OUTPUT_DIR"
