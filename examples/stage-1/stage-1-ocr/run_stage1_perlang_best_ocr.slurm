#!/bin/bash
#SBATCH --job-name=s1-perlang-ocr
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --chdir=/data/gpfs/projects/punim0478/davidsamuels/dictionary-extractor
#SBATCH --output=/data/gpfs/projects/punim0478/davidsamuels/dictionary-extractor/logs/stage1_perlang_best_ocr_%j.log
#SBATCH --error=/data/gpfs/projects/punim0478/davidsamuels/dictionary-extractor/logs/stage1_perlang_best_ocr_%j.log
#
# Stage-1 per-language best model+config + OCR hint → outputs/stage-1-ocr/*_ocrhint/ (30 langs).
# API-bound (Gemini / OpenRouter); resumes without --overwrite.
#
# Submit:
#   sbatch examples/stage-1/stage-1-ocr/run_stage1_perlang_best_ocr.slurm
#
# Force full re-run:
#   sbatch examples/stage-1/stage-1-ocr/run_stage1_perlang_best_ocr.slurm --overwrite

set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

PROJECT_ROOT="/data/gpfs/projects/punim0478/davidsamuels/dictionary-extractor"
cd "${PROJECT_ROOT}"

echo "=== Stage-1 per-language best + OCR hint → stage-1-ocr (SLURM) ==="
echo "Partition: ${SLURM_JOB_PARTITION:-?}  Job: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-?}  Started: $(date -Is)"

bash examples/stage-1/stage-1-ocr/run_stage1_per_lang_best_flat_alpha_ocr.sh "$@"

echo "Finished: $(date -Is)"
