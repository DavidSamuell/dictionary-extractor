#!/bin/bash
#
# Dictionary extraction comparison experiments
# Tests 4 scenarios: Mathpix OCR (DOCX) vs PaddleOCR VL (crops), with and without preprocessing
#

set -e  # Exit on error

echo "========================================================================"
echo "DICTIONARY EXTRACTION EXPERIMENTS"
echo "========================================================================"
echo "Running 4 experiments:"
echo "  1. Mathpix OCR (DOCX) + Gemini 2.5 Pro (NO preprocessing)"
echo "  2. Mathpix OCR (DOCX) + Gemini 2.5 Pro (WITH all preprocessing)"
echo "  3. PaddleOCR VL (crops) + Gemini 2.5 Pro (NO preprocessing)"
echo "  4. PaddleOCR VL (crops) + Gemini 2.5 Pro (WITH all preprocessing)"
echo "========================================================================"
echo ""

# # Experiment 1: Mathpix OCR (DOCX) + Gemini 2.5 Pro (NO preprocessing)
# echo "-----------------------------------------------------------------------"
# echo "Experiment 1: Mathpix OCR (DOCX) + Gemini (NO preprocessing)"
# echo "-----------------------------------------------------------------------"
# python src/extract_dictionary.py \
#     -i preprocessing_outputs/preprocessed_baseline_grayscale.png \
#     -d assets/extracted-dict-page.docx \
#     -m gemini/gemini-2.5-pro \
#     -o outputs/mathpix+gemini-no-preprocess/extracted_dictionary.tsv \
#     --json

# echo ""
# echo "Running evaluation..."
# python src/evaluate_extraction.py \
#   -e outputs/mathpix+gemini-no-preprocess/extracted_dictionary.tsv \
#   -g assets/gold_label_dictionary.tsv \
#   -o outputs/mathpix+gemini-no-preprocess

# echo ""
# echo "✓ Completed: Mathpix OCR (NO preprocessing)"
# echo ""

# # Experiment 2: Mathpix OCR (DOCX) + Gemini 2.5 Pro (WITH all preprocessing)
# echo "-----------------------------------------------------------------------"
# echo "Experiment 2: Mathpix OCR (DOCX) + Gemini (WITH all preprocessing)"
# echo "-----------------------------------------------------------------------"
# python src/extract_dictionary.py \
#     -i preprocessing_outputs/preprocessed_test-dict-page.png \
#     -d assets/extracted-dict-page.docx \
#     -m gemini/gemini-2.5-pro \
#     -o outputs/mathpix+gemini-with-preprocess/extracted_dictionary.tsv \
#     --json

# echo ""
# echo "Running evaluation..."
# python src/evaluate_extraction.py \
#   -e outputs/mathpix+gemini-with-preprocess/extracted_dictionary.tsv \
#   -g assets/gold_label_dictionary.tsv \
#   -o outputs/mathpix+gemini-with-preprocess

# echo ""
# echo "✓ Completed: Mathpix OCR (WITH all preprocessing)"
# echo ""

# Experiment 3: PaddleOCR VL (crops) + Gemini 2.5 Pro (NO preprocessing)
echo "-----------------------------------------------------------------------"
echo "Experiment 3: PaddleOCR VL (crops) + Gemini (NO preprocessing)"
echo "-----------------------------------------------------------------------"
# Check if PaddleOCR VL output already exists
if [ -d "paddle_ocr_vl_output/baseline/preprocessed_baseline_grayscale_crops" ]; then
    echo "✓ PaddleOCR VL output already exists, skipping..."
else
    echo "Running PaddleOCR VL on baseline image..."
    python src/paddle_ocr_vl.py \
        -i preprocessing_outputs/preprocessed_baseline_grayscale.png \
        -o paddle_ocr_vl_output/baseline
fi

echo ""
echo "Extracting from crops..."
python src/extract_dictionary.py \
  --crops-dir paddle_ocr_vl_output/baseline/preprocessed_baseline_grayscale_crops \
  -m gemini/gemini-2.5-pro \
  -o outputs/paddleocr+gemini-no-preprocess/extracted_dictionary.tsv \
  --json

echo ""
echo "Running evaluation..."
python src/evaluate_extraction.py \
  -e outputs/paddleocr+gemini-no-preprocess/extracted_dictionary.tsv \
  -g assets/gold_label_dictionary.tsv \
  -o outputs/paddleocr+gemini-no-preprocess

echo ""
echo "✓ Completed: PaddleOCR VL (NO preprocessing)"
echo ""

# Experiment 4: PaddleOCR VL (crops) + Gemini 2.5 Pro (WITH all preprocessing)
echo "-----------------------------------------------------------------------"
echo "Experiment 4: PaddleOCR VL (crops) + Gemini (WITH all preprocessing)"
echo "-----------------------------------------------------------------------"
# Check if PaddleOCR VL output already exists
if [ -d "paddle_ocr_vl_output/preprocessed/preprocessed_test-dict-page_crops" ]; then
    echo "✓ PaddleOCR VL output already exists, skipping..."
else
    echo "Running PaddleOCR VL on preprocessed image..."
    python src/paddle_ocr_vl.py \
        -i preprocessing_outputs/preprocessed_test-dict-page.png \
        -o paddle_ocr_vl_output/preprocessed
fi

echo ""
echo "Extracting from crops..."
python src/extract_dictionary.py \
  --crops-dir paddle_ocr_vl_output/preprocessed/preprocessed_test-dict-page_crops \
  -m gemini/gemini-2.5-pro \
  -o outputs/paddleocr+gemini-with-preprocess/extracted_dictionary.tsv \
  --json

echo ""
echo "Running evaluation..."
python src/evaluate_extraction.py \
  -e outputs/paddleocr+gemini-with-preprocess/extracted_dictionary.tsv \
  -g assets/gold_label_dictionary.tsv \
  -o outputs/paddleocr+gemini-with-preprocess

echo ""
echo "✓ Completed: PaddleOCR VL (WITH all preprocessing)"
echo ""

# Generate comparison summary
echo "======================================================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "======================================================================="
echo "Generating comparison summary..."
echo ""

# Create comparison report
cat > outputs/experiment_comparison.txt << EOF
================================================================================
DICTIONARY EXTRACTION EXPERIMENT RESULTS
================================================================================
Generated: $(date)

Experiment Configurations:
--------------------------
1. Mathpix OCR (DOCX) + Gemini 2.5 Pro (NO preprocessing)
   - OCR: Mathpix (DOCX output)
   - Image: Grayscale only (baseline)
   
2. Mathpix OCR (DOCX) + Gemini 2.5 Pro (WITH all preprocessing)
   - OCR: Mathpix (DOCX output)
   - Image: Grayscale + Deskew + Denoise + Contrast + Sharpen

3. PaddleOCR VL (crops) + Gemini 2.5 Pro (NO preprocessing)
   - OCR: PaddleOCR VL (with layout detection)
   - Image: Grayscale only (baseline)
   
4. PaddleOCR VL (crops) + Gemini 2.5 Pro (WITH all preprocessing)
   - OCR: PaddleOCR VL (with layout detection)
   - Image: Grayscale + Deskew + Denoise + Contrast + Sharpen

Results Summary:
----------------
EOF

# Extract CER/WER from each experiment and append to summary
for exp_dir in \
    "mathpix+gemini-no-preprocess" \
    "mathpix+gemini-with-preprocess" \
    "paddleocr+gemini-no-preprocess" \
    "paddleocr+gemini-with-preprocess"
do
    if [ -f "outputs/${exp_dir}/character_error_report.txt" ]; then
        echo "" >> outputs/experiment_comparison.txt
        echo "${exp_dir}:" >> outputs/experiment_comparison.txt
        grep "Character Error Rate" "outputs/${exp_dir}/character_error_report.txt" >> outputs/experiment_comparison.txt || echo "  CER: N/A" >> outputs/experiment_comparison.txt
        grep "Word Error Rate" "outputs/${exp_dir}/character_error_report.txt" >> outputs/experiment_comparison.txt || echo "  WER: N/A" >> outputs/experiment_comparison.txt
    fi
done

echo "" >> outputs/experiment_comparison.txt
echo "Detailed reports available in:" >> outputs/experiment_comparison.txt
echo "  - outputs/mathpix+gemini-no-preprocess/" >> outputs/experiment_comparison.txt
echo "  - outputs/mathpix+gemini-with-preprocess/" >> outputs/experiment_comparison.txt
echo "  - outputs/paddleocr+gemini-no-preprocess/" >> outputs/experiment_comparison.txt
echo "  - outputs/paddleocr+gemini-with-preprocess/" >> outputs/experiment_comparison.txt
echo "================================================================================" >> outputs/experiment_comparison.txt

cat outputs/experiment_comparison.txt
echo ""
echo "Summary saved to: outputs/experiment_comparison.txt"
echo "======================================================================="
