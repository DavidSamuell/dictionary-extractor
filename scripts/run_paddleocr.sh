#!/bin/bash
# PaddleOCR extraction script for Chukchi-Russian dictionary
# Optimized for clean, non-overlapping boxes suitable for VLM input

# Option 1: Use best Cyrillic model (cyrillic_PP-OCRv3_mobile_rec - 94.28% accuracy)
# Clean box detection: avoid overlaps, allow multi-entry boxes
python src/paddle_ocr.py -i assets/test-dict-page.png -o paddle_outputs \
  --text-rec-model cyrillic_PP-OCRv3_mobile_rec \
  --det-db-thresh 0.3 \
  --det-db-box-thresh 0.6 \
  --det-db-unclip-ratio 1.8 \
  --rec-score-thresh 0.5 \
  --det-limit-side-len 960

# Option 2: Use Latin model for Chukchi headwords (if Cyrillic doesn't recognize them well)
# python src/paddle_ocr.py -i assets/test-dict-page.png -o paddle_outputs \
#   --text-rec-model latin_PP-OCRv5_mobile_rec