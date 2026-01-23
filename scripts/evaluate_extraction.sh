#!/bin/bash
# Evaluate dictionary extraction results against ground truth

python src/evaluate_extraction.py \
    -e outputs/gemini-2.5-pro-v2.0/extracted_dictionary.tsv \
    -g outputs/gold_label_dictionary.tsv \
    -o outputs/gemini-2.5-pro-v2.0/ \
    -t 0.85
