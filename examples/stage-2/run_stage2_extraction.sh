#!/bin/bash
set -euo pipefail

# Stage-2 direct MDF extraction only.  Outputs land in
#   {entry}/outputs/stage-2/<experiment-name>/<stem>/<stem>.mdf.txt
# alongside a single run_config.json per experiment.
#
# Pass --stage2-experiment-name to keep ablation runs separate. Combine with
# --no-intro and/or --toolbox-pdf to control which Stage-2 inputs are in scope.
# Re-runs without --overwrite resume per-experiment (Pass 1 cheat sheet is cached
# under outputs/stage-2/<experiment>/field_cheatsheet.json). Pass --overwrite to
# re-run Pass 1 discovery and Pass 2 extraction for that experiment.
#
# Default: --one-page-per-entry — one snippet per language. Uses the lowest-numbered
# stage-2-gold page when labeled; otherwise the lowest page with stage-1 gold, else
# the lowest page number among snippets (numeric, not lexicographic).
#
# Stage-1 inputs (gold): outputs/stage-1-gold/<stem>/*_stage1_GOLD_flat.txt
# Stage-1 inputs (predictions): outputs/stage-1/<experiment>/<stem>/*_stage1_flat.txt
# Experiment naming: <model>_<reasoning>_mdf_<intro|nointro>_<toolbox|notoolbox>
#
# Prerequisites: uv sync; stage-1 gold present; flat gold via
#   uv run python scripts/flatten_stage1_gold.py
# Toolbox arms need Pages from ToolboxReferenceManual.pdf at repo root.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_DIR="${SAMPLES_DIR:-assets/dictionaries/samples}"

# Subset of language subfolders to process across every experiment below.
# Edit this list (or comment out the `--languages "${LANGUAGES[@]}"` line on
# any invocation) to run against the full samples root.
LANGUAGES=(
    Evenki-Russian
    Chukchi-Russian
    Nahuatl-French
    Na-English-Chinese-French
    Kashmiri-English
    Tiri-English
    Greek-English
    Efik-English
    Circassian-English-Turkish
    "Iñupiatun Eskimo-English"
)

TOOLBOX_PDF="Pages from ToolboxReferenceManual.pdf"

run_stage2() {
    if ! uv run dictextractor-extract "$@"; then
        echo "WARNING: stage-2 experiment failed or was skipped; continuing." >&2
    fi
}

# # Baseline: Gemini 3.1 Pro, high reasoning, intro + no toolbox PDF
run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model gemini/gemini-3.1-pro-preview \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --stage2-experiment-name gemini31pro_high_mdf_intro_notoolbox

# Sweep example — intro × toolbox ablations against the baseline (uncomment what you need):
run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model gemini/gemini-3.1-pro-preview \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --toolbox-pdf "${TOOLBOX_PDF}" \
    --stage2-experiment-name gemini31pro_high_mdf_intro_toolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model gemini/gemini-3.1-pro-preview \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --no-intro \
    --stage2-experiment-name gemini31pro_high_mdf_nointro_notoolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model gemini/gemini-3.1-pro-preview \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --no-intro \
    --toolbox-pdf "${TOOLBOX_PDF}" \
    --stage2-experiment-name gemini31pro_high_mdf_nointro_toolbox

# # OpenRouter — GPT-5.5 (OPEN_ROUTER_API_KEY):
run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/openai/gpt-5.5 \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --stage2-experiment-name gpt55_high_mdf_intro_notoolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/openai/gpt-5.5 \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --toolbox-pdf "${TOOLBOX_PDF}" \
    --stage2-experiment-name gpt55_high_mdf_intro_toolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/openai/gpt-5.5 \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --no-intro \
    --stage2-experiment-name gpt55_high_mdf_nointro_notoolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/openai/gpt-5.5 \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --no-intro \
    --toolbox-pdf "${TOOLBOX_PDF}" \
    --stage2-experiment-name gpt55_high_mdf_nointro_toolbox

# # OpenRouter — Claude Opus 4.7:
run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/anthropic/claude-opus-4.7 \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --stage2-experiment-name claudeopus47_high_mdf_intro_notoolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/anthropic/claude-opus-4.7 \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --toolbox-pdf "${TOOLBOX_PDF}" \
    --stage2-experiment-name claudeopus47_high_mdf_intro_toolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/anthropic/claude-opus-4.7 \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --no-intro \
    --stage2-experiment-name claudeopus47_high_mdf_nointro_notoolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/anthropic/claude-opus-4.7 \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --no-intro \
    --toolbox-pdf "${TOOLBOX_PDF}" \
    --stage2-experiment-name claudeopus47_high_mdf_nointro_toolbox

# # OpenRouter — Qwen3-VL 235B:
run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/qwen/qwen3-vl-235b-a22b-instruct \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --stage2-experiment-name qwen3vl235_high_mdf_intro_notoolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/qwen/qwen3-vl-235b-a22b-instruct \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --toolbox-pdf "${TOOLBOX_PDF}" \
    --stage2-experiment-name qwen3vl235_high_mdf_intro_toolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/qwen/qwen3-vl-235b-a22b-instruct \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --no-intro \
    --stage2-experiment-name qwen3vl235_high_mdf_nointro_notoolbox

run_stage2 \
    --strategy two_stage --stage 2 \
    --stage2-mode direct_mdf \
    --model openrouter/qwen/qwen3-vl-235b-a22b-instruct \
    --stage2-reasoning high \
    --samples-dir assets/dictionaries/samples \
    --languages "${LANGUAGES[@]}" \
    --one-page-per-entry \
    --stage1-input flat \
    --no-intro \
    --toolbox-pdf "${TOOLBOX_PDF}" \
    --stage2-experiment-name qwen3vl235_high_mdf_nointro_toolbox
