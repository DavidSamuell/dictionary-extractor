#!/usr/bin/env python3
"""
PaddleOCR VL pipeline with integrated cropping and text extraction.
Processes dictionary pages and prepares crops for LLM-based extraction.
"""

import json
import argparse
from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCRVL


def crop_and_extract_text_blocks(image_path, json_data, crops_dir):
    """
    Crop text blocks from the image and save corresponding text content.

    Args:
        image_path: Path to the input image
        json_data: Parsed JSON data from PaddleOCR VL
        crops_dir: Directory to save the cropped images and text files
    """
    # Load the image
    image = Image.open(image_path)

    # Create crops directory
    crops_path = Path(crops_dir)
    crops_path.mkdir(parents=True, exist_ok=True)

    # Get parsing results
    parsing_res_list = json_data.get("parsing_res_list", [])

    # Filter for text blocks only (skip paragraph_title and other labels)
    text_blocks = [
        block for block in parsing_res_list if block.get("block_label") == "text"
    ]

    print(
        f"\nProcessing {len(text_blocks)} text blocks (filtered from {len(parsing_res_list)} total blocks)..."
    )

    for block in text_blocks:
        block_label = block.get("block_label", "unknown")
        block_id = block.get("block_id", -1)
        block_content = block.get("block_content", "")
        block_bbox = block.get("block_bbox", [])

        if len(block_bbox) == 4 and block_content:
            x1, y1, x2, y2 = block_bbox

            # Crop the block
            cropped = image.crop((x1, y1, x2, y2))

            # Save cropped image
            crop_filename = f"block_{block_id:03d}_text.png"
            crop_path = crops_path / crop_filename
            cropped.save(crop_path)

            # Save text content as Markdown
            md_filename = f"block_{block_id:03d}_text.md"
            md_path = crops_path / md_filename
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(block_content)

            print(
                f"  ✓ Block {block_id}: Saved {crop_filename} ({x2-x1}x{y2-y1}px) and {md_filename}"
            )

    print(f"\n✓ All text blocks saved to: {crops_path}")
    return len(text_blocks)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR VL on dictionary pages and extract text blocks"
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="preprocessing_outputs/preprocessed_test-dict-page.png",
        help="Path to input image (default: preprocessing_outputs/preprocessed_test-dict-page.png)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="paddle_ocr_vl_output",
        help="Output directory prefix (default: paddle_ocr_vl_output)",
    )
    args = parser.parse_args()

    input_image = args.image
    output_dir = args.output

    # Get input filename for naming outputs
    input_name = Path(input_image).stem

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize PaddleOCR VL pipeline
    pipeline = PaddleOCRVL()

    # Run PaddleOCR VL prediction
    print(f"Running PaddleOCR VL on: {input_image}")
    output = pipeline.predict(input_image)

    for res in output:
        res.print()

        # Save standard outputs
        res.save_to_json(save_path=output_dir)
        res.save_to_markdown(save_path=output_dir)
        res.save_to_img(save_path=output_dir)

        print(f"\n✓ Saved standard outputs to {output_dir}/")

        # Load the saved JSON to process crops
        json_path = Path(output_dir) / f"{input_name}_res.json"

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Create crops directory
            crops_dir = Path(output_dir) / f"{input_name}_crops"

            # Crop and extract text blocks
            num_blocks = crop_and_extract_text_blocks(input_image, json_data, crops_dir)

            print(f"\n{'='*60}")
            print(f"Pipeline complete!")
            print(f"  - Processed {num_blocks} text blocks")
            print(f"  - Crops saved to: {crops_dir}")
            print(f"  - Ready for dictionary extraction")
            print(f"{'='*60}")
        else:
            print(f"Warning: JSON output not found at {json_path}")


if __name__ == "__main__":
    main()
