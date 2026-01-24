# Dictionary Extraction Approach and Results Report

## Overview
This report describes the methodology, experiments, and findings from automated dictionary extraction of Chukchi-Russian bilingual dictionary pages using various OCR and LLM-based approaches.

## Methodology

### 1. Initial Baseline Approach
**Tool Stack:** Mathpix OCR (DOCX output) + Google Gemini 2.5 Pro

This approach was selected as the baseline based on previous successful dictionary extraction work in earlier research. The workflow:
- Mathpix OCR extracts text from dictionary page images and outputs structured DOCX format
- DOCX text is passed to Gemini 2.5 Pro with structured prompts to extract dictionary entries
- Output is parsed into TSV format with fields: Headword_Phrase, Entry_Type, POS, Translation_RU, Literal_Meaning, Grammar_Notes

**Results:** 
- CER: 1.57%, WER: 7.77% (no preprocessing)
- CER: 1.43%, WER: 6.80% (with all preprocessing)

### 2. Image Preprocessing Experiments
To improve OCR quality, we tested various preprocessing techniques on the input images:
- **Grayscale conversion** (baseline)
- **Deskewing** (rotation correction)
- **Denoising** (bilateral filtering)
- **Contrast normalization** (CLAHE)
- **Sharpening** (Gaussian + weighted addition)

**Finding:** Applying all preprocessing steps together ("all") yielded the best results, reducing CER from 1.57% to 1.43%.

### 3. Alternative LLM: Qwen3-VL-30B
We experimented with the open-source vision-language model Qwen3-VL-30B (with extended thinking) as an alternative to Gemini 2.5 Pro.

**Results:** 
- CER: 12.84%, WER: 34.23%
- Significantly worse performance compared to Gemini
- **Conclusion:** Gemini 2.5 Pro remains the superior choice for this task

### 4. Alternative OCR: PaddleOCR Experiments

#### 4.1 PaddleOCR Cyrillic Models
Tested PaddlePaddle/cyrillic_PP-OCRv3_mobile_rec (highest accuracy for Cyrillic) and PP-OCRv5:
- **Results:** Poor performance with many misdetections and incorrect character recognition
- **Conclusion:** Traditional OCR models struggled with the complex mixed-script layout

#### 4.2 PaddleOCR-VL-0.9B (Vision-Language Model)
Switched to PaddleOCR-VL-0.9B, a vision-language model with layout detection capabilities.

**Advantages:**
- Successfully detected and separated 2 distinct text blocks (left and right columns)
- Each column processed independently, maintaining layout structure
- Better handling of mixed Cyrillic and Latin scripts

**Results:**
- CER: 9.41%, WER: 17.53% (no preprocessing)
- CER: 6.90%, WER: 23.57% (with all preprocessing)

**Limitations:**
- Still significantly worse than Mathpix OCR
- Tendency to omit text within parentheses in Russian translations, artificially increasing CER
- Higher word error rate despite improvements from preprocessing

### 5. Post-Processing: Homoglyph Normalization

#### Problem Identification
Character-level error analysis revealed systematic confusion between visually similar characters (homoglyphs) from different Unicode scripts:
- Latin `B` (U+0042) vs Cyrillic `В` (U+0412)
- Latin `p` (U+0070) vs Cyrillic `р` (U+0440)
- Latin schwa `ə` (U+0259) vs Cyrillic schwa `ә` (U+04D9)

These homoglyphs appeared as false positive errors in evaluation, inflating the CER metrics.

#### Solution
Implemented bidirectional homoglyph normalization in the evaluation script:
- All text (both ground truth and extracted) is normalized to Cyrillic equivalents before comparison
- Applied to both character-level and word-level error analysis
- Significantly reduced false positive substitution errors

#### Impact
This normalization revealed the true character-level errors and improved evaluation accuracy.

### 6. Additional Post-Processing Normalizations
- **Comma-to-semicolon normalization:** Gold label data used semicolons as delimiters, but some OCR outputs produced commas. All commas normalized to semicolons for consistent comparison.

## Final Results Summary

| Approach | CER | WER |
|----------|-----|-----|
| **Mathpix + Gemini (no preprocessing)** | **1.57%** | **7.77%** |
| **Mathpix + Gemini (with all preprocessing)** | **1.43%** | **6.80%** |
| PaddleOCR-VL + Gemini (no preprocessing) | 9.41% | 17.53% |
| PaddleOCR-VL + Gemini (with all preprocessing) | 6.90% | 23.57% |
| Qwen3-VL-30B | 12.84% | 34.23% |

**Winner:** Mathpix OCR + Gemini 2.5 Pro with full preprocessing pipeline

## Error Analysis and Limitations

### Key Character-Level Errors
1. **`ь` (soft sign) vs `ъ` (hard sign):** These two Cyrillic letters are visually very similar in the dictionary image, making them difficult to distinguish even for human annotators
2. **`n` (Latin) vs `п` (Cyrillic):** Another ambiguous pair in the source material
3. **Diacritics:** Occasional confusion with accented characters (e.g., `á` vs `a`, `ǽ` vs `æ`)
4. **Text in parentheses:** PaddleOCR-VL tends to omit or incompletely extract text within parentheses

### Gold Label Considerations
**Important Caveat:** The gold label reference (`assets/gold_label_dictionary.json`) was created by:
1. Starting with initial Mathpix + Gemini 2.5 Pro extraction output
2. Manual verification and correction by comparing with the original dictionary image
3. High confidence corrections for characters like `n` vs `п`
4. **Low confidence for `ь` vs `ъ`** - these are often indistinguishable in the image quality available

**Implication:** The Mathpix + Gemini approach may have an artificially lower error rate due to gold label bias, since the gold labels were partially derived from its own output.

## Recommendations for Future Improvement

1. **Gold Label Verification:** Consult with a native Chukchi language speaker or linguistics expert to verify ambiguous characters (`ь` vs `ъ`) in the gold label dataset
2. **Higher Resolution Images:** Obtain higher quality scans of the original dictionary to better distinguish visually similar characters
3. **Hybrid Approach:** Consider combining PaddleOCR-VL's superior layout detection with Mathpix's superior character recognition
4. **Prompt Engineering:** Further optimize LLM prompts for handling parenthetical expressions and grammar notes
5. **Post-OCR Correction:** Implement rule-based or ML-based post-processing to correct common systematic errors

## Conclusion

The **Mathpix OCR + Gemini 2.5 Pro** approach with comprehensive image preprocessing (deskew, denoise, contrast, sharpen) remains the superior method for automated Chukchi-Russian dictionary extraction, achieving a character error rate of 1.43% and word error rate of 6.80%. While open-source alternatives (PaddleOCR-VL, Qwen3-VL) were explored, they could not match the accuracy of the commercial solution. Key post-processing steps, particularly homoglyph normalization, were essential for accurate evaluation and error analysis.

---

**Generated:** January 24, 2026  
**Dataset:** Chukchi-Russian Bilingual Dictionary  
**Evaluation Metrics:** Character Error Rate (CER), Word Error Rate (WER)  
**Total Entries Evaluated:** 37 dictionary entries across 6 fields
