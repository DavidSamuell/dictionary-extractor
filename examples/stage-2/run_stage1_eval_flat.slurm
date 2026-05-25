#!/bin/bash
#SBATCH --job-name=s1-eval-flat
#SBATCH --partition=cascade,sapphire
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/gpfs/projects/punim0478/davidsamuels/dictionary-extractor
#SBATCH --output=/data/gpfs/projects/punim0478/davidsamuels/dictionary-extractor/logs/stage1_flat_eval_%j.log
#SBATCH --error=/data/gpfs/projects/punim0478/davidsamuels/dictionary-extractor/logs/stage1_flat_eval_%j.log
#
# Stage-1 eval-flat (all experiments, 30 languages) with parallel page eval.
# CPU-only; ~32G RAM is plenty for 14 workers. Slurm picks cascade or sapphire
# whichever has idle nodes first.
# Submit from project root:
#   sbatch examples/stage-2/run_stage1_eval_flat.slurm
#
# Uses cache for unchanged pages; add --overwrite to the uv run below for full recompute.

set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

PROJECT_ROOT="/data/gpfs/projects/punim0478/davidsamuels/dictionary-extractor"
cd "${PROJECT_ROOT}"

WORKERS="${EVAL_WORKERS:-14}"

LANGUAGES=(
    Canala-English
    Chepang-English
    Efik-English
    Na-English-Chinese-French
    Reel-English
    Ritharngu-English
    Shilluk-English
    Evenki-Russian
    Chukchi-Russian
    Circassian-English-Turkish
    Nahuatl-French
    Khmer-English
    Malay-English
    Kashmiri-English
    Greek-English
    Telugu-English
    "Iñupiatun Eskimo-English"
    "Vernacular Syriac-Kurdish_Turkish-English"
    Syriac-English
    Tiri-English
    Thai-Russian
    Assyrian-English
    Yiddish-English
    Georgian-Russian
    Japanese-English
    Punjabi-English
    Gujarati-English
    Gojri-English-Hindi
    Bengalese-English
    Sanskrit-English
)

echo "=== Stage 1 eval-flat (SLURM) ==="
echo "Partition: ${SLURM_JOB_PARTITION:-?}  Job: ${SLURM_JOB_ID:-local}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-?}  Workers: ${WORKERS}"
echo "Languages (${#LANGUAGES[@]}): ${LANGUAGES[*]}"
echo "Started: $(date -Is)"

uv run dictextractor-eval-flat \
  --samples-dir assets/dictionaries/samples \
  --all-experiments \
  --languages "${LANGUAGES[@]}" \
  --workers "${WORKERS}" \
  -o evaluations/stage1_flat_eval

echo "Finished: $(date -Is)"
