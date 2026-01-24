"""
Annotate image with bounding boxes and labels from PaddleOCR VL output
"""
import json
import cv2
import numpy as np
from pathlib import Path
import argparse


def load_json(json_path):
    """Load JSON file with OCR results"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def annotate_image(image_path, json_data, output_path, thickness=3, font_scale=0.8):
    """
    Annotate image with bounding boxes and labels
    
    Args:
        image_path: Path to input image
        json_data: Parsed JSON data with parsing_res_list
        output_path: Path to save annotated image
        thickness: Line thickness for bounding boxes
        font_scale: Font scale for labels
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Define colors for different block types (BGR format)
    color_map = {
        'paragraph_title': (0, 0, 255),      # Red
        'text': (0, 255, 0),                 # Green
        'image': (255, 0, 0),                # Blue
        'table': (255, 255, 0),              # Cyan
        'figure': (255, 0, 255),             # Magenta
        'formula': (0, 255, 255),            # Yellow
        'default': (255, 128, 0),            # Orange
    }
    
    # Process each block
    parsing_results = json_data.get('parsing_res_list', [])
    
    print(f"Found {len(parsing_results)} blocks to annotate")
    
    for block in parsing_results:
        block_label = block.get('block_label', 'unknown')
        block_bbox = block.get('block_bbox', [])
        block_id = block.get('block_id', -1)
        block_order = block.get('block_order', -1)
        
        if len(block_bbox) != 4:
            print(f"Skipping block {block_id}: invalid bbox {block_bbox}")
            continue
        
        # Get bbox coordinates
        x_min, y_min, x_max, y_max = map(int, block_bbox)
        
        # Get color for this block type
        color = color_map.get(block_label, color_map['default'])
        
        # Draw rectangle
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Prepare label text
        label_text = f"{block_label}"
        if block_order is not None:
            label_text += f" [{block_order}]"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw filled rectangle for text background
        cv2.rectangle(
            img,
            (x_min, y_min - text_height - baseline - 5),
            (x_min + text_width, y_min),
            color,
            -1  # Filled
        )
        
        # Draw label text
        cv2.putText(
            img,
            label_text,
            (x_min, y_min - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White text
            thickness,
            cv2.LINE_AA
        )
        
        print(f"  Block {block_id}: {block_label} at {block_bbox}")
    
    # Save annotated image
    cv2.imwrite(str(output_path), img)
    print(f"\nSaved annotated image to: {output_path}")
    
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Annotate image with bounding boxes from PaddleOCR VL JSON output"
    )
    
    parser.add_argument(
        '-i', '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    
    parser.add_argument(
        '-j', '--json',
        type=str,
        required=True,
        help='Path to JSON file with OCR results'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to save annotated image'
    )
    
    parser.add_argument(
        '--thickness',
        type=int,
        default=3,
        help='Line thickness for bounding boxes (default: 3)'
    )
    
    parser.add_argument(
        '--font-scale',
        type=float,
        default=0.8,
        help='Font scale for labels (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    image_path = Path(args.image)
    json_path = Path(args.json)
    output_path = Path(args.output)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load JSON data
    print(f"Loading JSON from: {json_path}")
    json_data = load_json(json_path)
    
    # Annotate image
    print(f"Annotating image: {image_path}")
    annotate_image(
        image_path,
        json_data,
        output_path,
        thickness=args.thickness,
        font_scale=args.font_scale
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
