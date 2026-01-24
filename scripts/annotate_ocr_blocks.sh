#!/bin/bash
# Annotate image with bounding boxes from PaddleOCR VL JSON output

python src/annotate_ocr_blocks.py \
  -i paddle_outputs/preprocessed.png \
  -j paddle_ocr_vl_output/preprocessed_res.json \
  -o paddle_ocr_vl_output/annotated_image.png \
  --thickness 3 \
  --font-scale 0.8
