#!/bin/bash
# Batch script to test CER impact of different preprocessing methods
# This script runs extraction and evaluation for each preprocessing variant

set -e  # Exit on error

# Configuration
MODEL="gemini/gemini-2.5-pro"
GROUND_TRUTH="assets/gold_label_dictionary.tsv"
DOCX_PATH="assets/extracted-dict-page.docx"

# Check if ground truth exists
if [ ! -f "$GROUND_TRUTH" ]; then
    echo "Error: Ground truth file not found at $GROUND_TRUTH"
    echo "Please create the ground truth file first."
    exit 1
fi

echo "======================================================================="
echo "PREPROCESSING CER COMPARISON TEST"
echo "======================================================================="
echo "Model: $MODEL"
echo "Ground Truth: $GROUND_TRUTH"
echo ""

# Array of preprocessing variants
declare -a variants=(
    "baseline_grayscale:Baseline (Grayscale Only)"
    "deskew_only:Grayscale + Deskew"
    "denoise_only:Grayscale + Denoise"
    "contrast_only:Grayscale + Contrast (CLAHE)"
    "sharpen_only:Grayscale + Sharpen"
    "test-dict-page:All Steps Combined"
)

# Loop through each variant
for variant_info in "${variants[@]}"; do
    # Split variant name and description
    IFS=':' read -r variant_name variant_desc <<< "$variant_info"
    
    IMAGE="preprocessing_outputs/preprocessed_${variant_name}.png"
    OUTPUT_DIR="outputs/preprocessing_comparison/${variant_name}"
    
    # Check if image exists
    if [ ! -f "$IMAGE" ]; then
        echo "Warning: Image not found: $IMAGE (skipping)"
        continue
    fi
    
    echo ""
    echo "-----------------------------------------------------------------------"
    echo "Testing: $variant_desc"
    echo "Image: $IMAGE"
    echo "Output: $OUTPUT_DIR"
    echo "-----------------------------------------------------------------------"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run extraction
    echo "Running extraction..."
    python src/extract_dictionary.py \
        -i "$IMAGE" \
        -d "$DOCX_PATH" \
        -m "$MODEL" \
        -o "$OUTPUT_DIR/extracted_dictionary.tsv" \
        --json
    
    if [ $? -ne 0 ]; then
        echo "Error: Extraction failed for $variant_desc"
        continue
    fi
    
    # Run evaluation
    echo ""
    echo "Running evaluation..."
    python src/evaluate_extraction.py \
        -e "$OUTPUT_DIR/extracted_dictionary.tsv" \
        -g "$GROUND_TRUTH" \
        -o "$OUTPUT_DIR" \
        -t 0.85
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed for $variant_desc"
        continue
    fi
    
    echo "✓ Completed: $variant_desc"
done

echo ""
echo "======================================================================="
echo "ALL TESTS COMPLETED"
echo "======================================================================="
echo ""
echo "Generating CER comparison summary..."

# Generate summary report
SUMMARY_FILE="outputs/preprocessing_comparison/cer_summary.txt"
mkdir -p "outputs/preprocessing_comparison"

echo "=======================================================================" > "$SUMMARY_FILE"
echo "PREPROCESSING CER COMPARISON SUMMARY" >> "$SUMMARY_FILE"
echo "=======================================================================" >> "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "Model: $MODEL" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Method                          | CER     | WER     | F1 Score" >> "$SUMMARY_FILE"
echo "--------------------------------|---------|---------|----------" >> "$SUMMARY_FILE"

for variant_info in "${variants[@]}"; do
    IFS=':' read -r variant_name variant_desc <<< "$variant_info"
    REPORT="outputs/preprocessing_comparison/${variant_name}/character_error_report.txt"
    EVAL_REPORT="outputs/preprocessing_comparison/${variant_name}/evaluation_report.txt"
    
    if [ -f "$REPORT" ]; then
        # Extract CER and WER from character error report
        CER=$(grep "Character Error Rate" "$REPORT" | head -1 | awk '{print $5}' | tr -d '()')
        WER=$(grep "Word Error Rate" "$REPORT" | head -1 | awk '{print $5}' | tr -d '()')
        
        # Extract F1 from evaluation report
        F1="N/A"
        if [ -f "$EVAL_REPORT" ]; then
            F1=$(grep "Overall F1" "$EVAL_REPORT" | awk '{print $3}')
        fi
        
        printf "%-31s | %-7s | %-7s | %s\n" "$variant_desc" "$CER" "$WER" "$F1" >> "$SUMMARY_FILE"
    else
        printf "%-31s | ERROR - Report not found\n" "$variant_desc" >> "$SUMMARY_FILE"
    fi
done

echo "" >> "$SUMMARY_FILE"
echo "Detailed reports available in: outputs/preprocessing_comparison/" >> "$SUMMARY_FILE"

# Display summary
cat "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
