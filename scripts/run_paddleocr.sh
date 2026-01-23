#!/bin/bash
# PaddleOCR extraction script for Chukchi-Russian dictionary

# Option 1: Use best Cyrillic model (cyrillic_PP-OCRv3_mobile_rec - 94.28% accuracy)
python src/paddle_ocr.py -i assets/test-dict-page.png -o paddle_outputs \
  --det-db-thresh 0.4 \
  --det-db-box-thresh 0.5 \
  --det-db-unclip-ratio 1.3 \
  --det-limit-side-len 1280

# Option 2: Use Latin model for Chukchi headwords (if Cyrillic doesn't recognize them well)
# python src/paddle_ocr.py -i assets/test-dict-page.png -o paddle_outputs \
#   --text-rec-model latin_PP-OCRv5_mobile_rec