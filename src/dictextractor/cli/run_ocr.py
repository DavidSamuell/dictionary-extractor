"""
CLI: run an OCR backend on a dictionary page image.
Usage: python -m dictextractor.cli.run_ocr [options]
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run an OCR backend on a dictionary page image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PaddleOCR traditional
  python -m dictextractor.cli.run_ocr -i page.png -o paddle_outputs --backend paddle_traditional
  python -m dictextractor.cli.run_ocr -i page.png -o paddle_outputs --backend paddle_traditional \\
      --text-rec-model cyrillic_PP-OCRv3_mobile_rec --det-db-thresh 0.3

  # PaddleOCR VL
  python -m dictextractor.cli.run_ocr -i page.png -o paddle_ocr_vl_output --backend paddle_vl
        """,
    )
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument("--backend", choices=["paddle_traditional", "paddle_vl"],
                        default="paddle_traditional", help="OCR backend to use")

    # paddle_traditional options
    parser.add_argument("--text-det-model", default=None)
    parser.add_argument("--text-rec-model", default="cyrillic_PP-OCRv3_mobile_rec")
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--det-db-thresh", type=float, default=0.3)
    parser.add_argument("--det-db-box-thresh", type=float, default=0.6)
    parser.add_argument("--det-db-unclip-ratio", type=float, default=1.5)
    parser.add_argument("--det-limit-side-len", type=int, default=960)
    parser.add_argument("--rec-score-thresh", type=float, default=0.5)

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return 1

    try:
        if args.backend == "paddle_traditional":
            from dictextractor.ocr.paddle_traditional import PaddleOCRTraditional
            backend = PaddleOCRTraditional(
                use_gpu=args.use_gpu,
                text_detection_model_name=args.text_det_model,
                text_recognition_model_name=args.text_rec_model,
                det_db_thresh=args.det_db_thresh,
                det_db_box_thresh=args.det_db_box_thresh,
                det_db_unclip_ratio=args.det_db_unclip_ratio,
                det_limit_side_len=args.det_limit_side_len,
                rec_score_thresh=args.rec_score_thresh,
            )
            result = backend.run(str(image_path), output_dir=args.output_dir, padding=args.padding)
            print(f"\nExtracted {len(result.blocks)} entry blocks")

        elif args.backend == "paddle_vl":
            from dictextractor.ocr.paddle_vl import PaddleOCRVLBackend
            backend = PaddleOCRVLBackend()
            result = backend.run(str(image_path), output_dir=args.output_dir)
            print(f"\nDetected {len(result.blocks)} layout blocks ({len(result.text_blocks)} text blocks)")

        return 0

    except Exception as e:
        print(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
