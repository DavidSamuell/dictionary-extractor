#!/bin/bash
#
# Evaluate Stage 1 OCR quality for Evenki-Russian and Chukchi-Russian samples.
#

set -euo pipefail

SAMPLES_DIR="assets/dictionaries/samples"
OUTPUT_DIR="assets/dictionaries/samples/stage1_eval"

echo "=== Running Stage 1 evaluation ==="
echo ""

uv run python -m dictextractor.cli.evaluate_stage1 \
    --samples-dir "$SAMPLES_DIR" \
    -o "$OUTPUT_DIR"
