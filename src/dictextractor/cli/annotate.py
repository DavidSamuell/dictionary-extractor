"""
CLI: annotate an image with bounding boxes from PaddleOCR VL JSON output.
Usage: python -m dictextractor.cli.annotate [options]
"""

import argparse
import json
from pathlib import Path

from dictextractor.utils.visualization import annotate_image


def main():
    parser = argparse.ArgumentParser(
        description="Annotate image with bounding boxes from PaddleOCR VL JSON output"
    )
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("-j", "--json", required=True, dest="json_path", help="PaddleOCR VL JSON path")
    parser.add_argument("-o", "--output", required=True, help="Output annotated image path")
    parser.add_argument("--thickness", type=int, default=3)
    parser.add_argument("--font-scale", type=float, default=0.8)

    args = parser.parse_args()

    image_path = Path(args.image)
    json_path = Path(args.json_path)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    annotate_image(str(image_path), json_data, args.output, args.thickness, args.font_scale)
    return 0


if __name__ == "__main__":
    exit(main())
