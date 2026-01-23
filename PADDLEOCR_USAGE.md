# PaddleOCR Dictionary Entry Extraction - Usage Guide

## Installation

Install PaddleOCR and dependencies:

```bash
pip install paddleocr paddlepaddle scikit-learn
```

## Quick Start

Extract entries from a dictionary page with default Cyrillic OCR:

```bash
python src/paddle_ocr.py -i assets/test-dict-page.png -o paddle_outputs
```

## Command-Line Options

```
Required Arguments:
  -i, --image         Input dictionary page image
  -o, --output-dir    Output directory for results

Optional Arguments:
  --lang              OCR language (default: cyrillic)
                      Options: cyrillic, multilingual, en
  --rec-model         Specific recognition model name
                      Example: cyrillic_PP-OCRv5_mobile_rec
  --padding           Padding around entry crops in pixels (default: 20)
  --use-gpu           Use GPU acceleration for OCR
```

## Language Recommendations

### For Chukchi-Russian Dictionary:

**Option 1: Multilingual (Recommended)**
- Best for mixed Latin (Chukchi headwords) + Cyrillic (Russian definitions)
```bash
python src/paddle_ocr.py -i page.png -o outputs --lang multilingual
```

**Option 2: Cyrillic with specific model**
- Optimized for Cyrillic script
```bash
python src/paddle_ocr.py -i page.png -o outputs \
  --rec-model cyrillic_PP-OCRv5_mobile_rec
```

## Output Structure

The script generates the following outputs:

```
paddle_outputs/
├── preprocessed.png              # CLAHE-enhanced grayscale image
├── stage1_raw_ocr.png           # Visualization: All detected text boxes (green)
├── stage2_columns.png           # Visualization: Left (blue) vs Right (red) columns
├── stage3_entries.png           # Visualization: Segmented entries with IDs
├── ocr_results.json             # Raw OCR output (all detected lines)
├── entries.json                 # Segmented entry metadata
└── entry_crops/                 # Individual entry images
    ├── L001.png                # Left column, entry 1
    ├── L002.png                # Left column, entry 2
    ├── R001.png                # Right column, entry 1
    └── ...
```

## Visualizations Explained

### Stage 1: Raw OCR Detection
- Green boxes around all detected text
- Confidence scores shown above boxes
- Verifies OCR is detecting text correctly

### Stage 2: Column Segmentation
- Blue boxes = Left column
- Red boxes = Right column
- Verifies midline column split is working

### Stage 3: Entry Segmentation
- Different colors for different entries
- Entry IDs labeled (L001, L002, R001, etc.)
- Individual line boxes within each entry
- Verifies entry boundaries are correct

## Tuning Entry Segmentation

If entry boundaries are incorrect, you can tune the segmentation logic in the code:

```python
# In segment_entries() method:

# Adjust gap threshold (currently: median_gap + 2 * mad_gap)
is_new_entry = gap > median_gap + 3 * mad_gap  # More conservative

# Adjust indent threshold (currently: 25th percentile)
indent_threshold = np.percentile(indents, 30)  # More sensitive to indentation
```

## Integration with VLM

The entry crops and metadata can be fed to Qwen3-VL for field extraction:

```python
# Example integration (to be implemented)
from paddle_ocr import PaddleOCRExtractor

extractor = PaddleOCRExtractor(lang='multilingual')
entries = extractor.extract_entries('page.png', 'outputs')

# For each entry, send to Qwen3-VL
for entry in entries:
    # entry.crop_image - the cropped entry image
    # entry.lines - OCR text lines with confidence
    # entry.entry_id - unique identifier
    
    # Feed to VLM for structured extraction
    result = extract_with_qwen(entry.crop_image, entry.lines)
```

## Troubleshooting

### No text detected
- Check image quality and contrast
- Try different preprocessing (adjust CLAHE parameters)
- Verify correct language setting

### Incorrect column split
- Image might not be 2-column layout
- Adjust midline calculation in `segment_columns()`

### Incorrect entry boundaries
- Tune gap and indent thresholds in `segment_entries()`
- Check Stage 3 visualization to identify patterns
- Dictionary layout might need custom heuristics

## Next Steps

1. Run on test page and review Stage 3 visualization
2. Verify entry boundaries match actual dictionary entries
3. Adjust segmentation parameters if needed
4. Once segmentation is good, integrate with Qwen3-VL for field extraction
