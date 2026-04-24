"""
CLI: run the image preprocessing pipeline.
Usage: python -m dictextractor.cli.preprocess [options]
"""

import argparse
from pathlib import Path

from dictextractor.preprocessing.preprocess import DictionaryPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dictionary page images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps
  python -m dictextractor.cli.preprocess -i page.png -o preprocessing_outputs

  # Run individual ablation images for CER testing
  python -m dictextractor.cli.preprocess -i page.png -o preprocessing_outputs --ablation

  # Skip specific steps
  python -m dictextractor.cli.preprocess -i page.png -o preprocessing_outputs --no-deskew --no-sharpen
        """,
    )
    parser.add_argument("-i", "--image", required=True, help="Input dictionary page image")
    parser.add_argument("-o", "--output-dir", default="preprocessing_outputs", help="Output directory")
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-contrast", action="store_true")
    parser.add_argument("--no-sharpen", action="store_true")
    parser.add_argument("--sharpen-amount", type=float, default=0.5)
    parser.add_argument("--deskew-threshold", type=float, default=0.5)
    parser.add_argument("--ablation", action="store_true",
                        help="Generate individual single-step images for CER ablation testing")
    parser.add_argument("--save-path", help="Explicit path to save final preprocessed image")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1

    preprocessor = DictionaryPreprocessor(str(image_path), output_dir=args.output_dir)

    if args.ablation:
        paths = preprocessor.generate_individual_preprocessed_images(
            deskew_threshold=args.deskew_threshold,
            sharpen_amount=args.sharpen_amount,
        )
        print("\nGenerated ablation images:")
        for step, path in paths.items():
            print(f"  {step}: {path}")

    preprocessor.run_all_steps(
        deskew=not args.no_deskew,
        deskew_threshold=args.deskew_threshold,
        denoise=not args.no_denoise,
        contrast=not args.no_contrast,
        sharpen=not args.no_sharpen,
        sharpen_amount=args.sharpen_amount,
    )
    preprocessor.save_result(args.save_path)
    return 0


if __name__ == "__main__":
    exit(main())
