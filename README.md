# Chukchi-Russian Dictionary Extractor

This project extracts dictionary entries from scanned images of Chukchi-Russian dictionaries using Large Language Models (LLMs).

## Methods to Try

1. MinerU
2. Mathpix
3. Gemini 2.5 Pro (currently implemented)
4. PaddleOCR + QwenLM

Reference: https://www.reddit.com/r/LocalLLaMA/comments/1q3qda4/what_is_the_best_opensource_vlm_model_for_ocr/

## Features

- **Image-based extraction**: Processes dictionary page images to extract structured entries
- **Dual input support**: Can use both image and OCR-extracted text (DOCX) for improved accuracy
- **TSV output format**: Generates tab-separated values with structured schema
- **Evaluation framework**: Compare extraction results against ground truth
- **Detailed metrics**: Calculates precision, recall, F1-score, and field-level accuracy

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Usage

### Extract Dictionary Entries

Run the extraction script to process dictionary images:

```bash
python extract_dictionary.py
```

This will:
- Load the test image (`test-dict-page.png`)
- Optionally use the extracted text (`extracted-dict-page.docx`) if available
- Extract entries using Gemini 2.5 Pro
- Save results to `extracted_dictionary.tsv`

### Customize Extraction

You can modify the extraction parameters in `extract_dictionary.py`:

```python
# Use a different model
dictionary_page = extract_dictionary_entries(
    image_path="your_image.png",
    docx_path="your_text.docx",  # Optional
    page_number=1,
    model="gemini/gemini-2.5-pro"  # or "gemini/gemini-2.5-flash"
)
```

### Evaluate Results

To evaluate extraction quality against ground truth:

```bash
python evaluate_extraction.py
```

This will:
- Compare `extracted_dictionary.tsv` with `ground_truth.tsv`
- Generate `evaluation_report.txt` with detailed metrics
- Save detailed results in JSON format

## Output Format

The TSV output follows this schema:

| Column | Description |
|--------|-------------|
| Headword | The Chukchi word being defined |
| Grammatical_Info | Grammatical category (e.g., сущ., гл.) |
| Russian_Definition | Russian translation/definition |
| Examples/Notes | Usage examples and additional notes |

## Creating Ground Truth

To create ground truth data for evaluation:

1. Manually review the dictionary image
2. Create a TSV file with the correct entries
3. Save as `ground_truth.tsv` in the project directory

## Evaluation Metrics

The evaluator calculates:
- **Exact matches**: Entries with perfect correspondence
- **Partial matches**: Entries with high similarity (>85% by default)
- **Field-level accuracy**: Accuracy for each column (headword, grammar, definition)
- **Precision/Recall/F1**: Standard classification metrics
- **Missing/Extra entries**: Entries not found or incorrectly added

## Project Structure

```
dictionary-extractor/
├── extract_dictionary.py     # Main extraction script
├── evaluate_extraction.py    # Evaluation script
├── requirements.txt          # Python dependencies
├── .env.example             # API key template
├── test-dict-page.png       # Sample dictionary page
├── extracted-dict-page.docx # OCR-extracted text (optional)
└── README.md                # This file
```

## Improving Extraction Quality

1. **Better OCR text**: Provide high-quality OCR output in DOCX format
2. **Tune prompts**: Modify the system prompt in `extract_dictionary.py`
3. **Adjust temperature**: Lower temperature (0.1) for consistent extraction
4. **Use larger models**: Try `gemini-2.5-pro` for better accuracy
5. **Multiple passes**: Run extraction multiple times and merge results

## Troubleshooting

- **API errors**: Check your API key in `.env` file
- **Low accuracy**: Ensure image quality is high and text is readable
- **Missing entries**: May need to adjust the extraction prompt
- **Encoding issues**: Ensure UTF-8 encoding for Cyrillic and special characters

## Future Improvements

- Batch processing for multiple pages
- Support for different dictionary formats
- ML-based post-processing for error correction
- Interactive UI for manual corrections
- Integration with other OCR engines