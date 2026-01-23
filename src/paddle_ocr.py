"""
PaddleOCR-based dictionary entry extraction pipeline.
Implements steps 1-7 from PaddleOCRPlan.md:
- CLAHE preprocessing
- PaddleOCR detection
- Column segmentation
- Entry segmentation
- Entry crop generation
- Multi-stage visualization
"""

import cv2
import numpy as np
import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
from paddleocr import PaddleOCR

# Pattern to detect POS tags that typically start a dictionary entry
# Matches: (сущ.), (гл.), (нар.), (прил.), (союз), (межд.), (част.), (предл.), (мест.)
POS_TAG_PATTERN = re.compile(r"\([а-яА-Яa-zA-Z]+\.?\)")


@dataclass
class BBox:
    """Axis-aligned bounding box"""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def cx(self) -> float:
        """Center x coordinate"""
        return (self.x_min + self.x_max) / 2

    @property
    def cy(self) -> float:
        """Center y coordinate"""
        return (self.y_min + self.y_max) / 2

    @property
    def height(self) -> float:
        """Box height"""
        return self.y_max - self.y_min

    @property
    def width(self) -> float:
        """Box width"""
        return self.x_max - self.x_min

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "x_min": float(self.x_min),
            "y_min": float(self.y_min),
            "x_max": float(self.x_max),
            "y_max": float(self.y_max),
        }


@dataclass
class OCRLine:
    """OCR detected line with text and confidence"""

    bbox: BBox
    text: str
    confidence: float
    column: Optional[str] = None  # 'L' or 'R'

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "bbox": self.bbox.to_dict(),
            "text": self.text,
            "confidence": float(self.confidence),
            "column": self.column,
        }


@dataclass
class DictionaryEntry:
    """Dictionary entry containing multiple lines"""

    entry_id: str
    lines: List[OCRLine]
    bbox: BBox
    column: str
    crop_image: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding crop_image)"""
        return {
            "entry_id": self.entry_id,
            "lines": [line.to_dict() for line in self.lines],
            "bbox": self.bbox.to_dict(),
            "column": self.column,
        }


class PaddleOCRExtractor:
    """Extract dictionary entries using PaddleOCR"""

    def __init__(
        self,
        use_gpu: bool = False,
        text_detection_model_name: Optional[str] = None,
        text_recognition_model_name: Optional[str] = "cyrillic_PP-OCRv3_mobile_rec",
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.6,
        det_db_unclip_ratio: float = 1.5,
        det_limit_side_len: int = 960,
    ):
        """
        Initialize PaddleOCR extractor

        Args:
            use_gpu: Whether to use GPU acceleration
            text_detection_model_name: Detection model name (default: PP-OCRv5_server_det)
                                       Options: PP-OCRv5_server_det, PP-OCRv5_mobile_det, etc.
            text_recognition_model_name: Recognition model name (default: cyrillic_PP-OCRv3_mobile_rec)
                                        Best Cyrillic: cyrillic_PP-OCRv3_mobile_rec (94.28% accuracy)
                                        Options: cyrillic_PP-OCRv5_mobile_rec (80.27%),
                                                latin_PP-OCRv5_mobile_rec (84.7%),
                                                en_PP-OCRv5_mobile_rec (85.25%)
                                        See: https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html
            det_db_thresh: DB detection threshold (default: 0.3, lower=more sensitive)
            det_db_box_thresh: Minimum confidence for detected boxes (default: 0.6, lower=more boxes)
            det_db_unclip_ratio: Box expansion ratio (default: 1.5, lower=tighter boxes)
            det_limit_side_len: Max image side length for detection (default: 960, higher=better for small text)
        """
        print(f"Initializing PaddleOCR...")
        print(f"  Text recognition model: {text_recognition_model_name}")
        if text_detection_model_name:
            print(f"  Text detection model: {text_detection_model_name}")
        print(f"  Detection parameters:")
        print(f"    - det_db_thresh: {det_db_thresh}")
        print(f"    - det_db_box_thresh: {det_db_box_thresh}")
        print(f"    - det_db_unclip_ratio: {det_db_unclip_ratio}")
        print(f"    - det_limit_side_len: {det_limit_side_len}")

        ocr_params = {
            "use_textline_orientation": True,  # Use the new parameter instead of use_angle_cls
            "text_recognition_model_name": text_recognition_model_name,
            "det_db_thresh": det_db_thresh,
            "det_db_box_thresh": det_db_box_thresh,
            "det_db_unclip_ratio": det_db_unclip_ratio,
            "det_limit_side_len": det_limit_side_len,
        }

        # Add detection model if specified
        if text_detection_model_name:
            ocr_params["text_detection_model_name"] = text_detection_model_name

        self.ocr = PaddleOCR(**ocr_params)
        print("PaddleOCR initialized successfully")

    def preprocess_image(
        self, image_path: str, output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply grayscale + CLAHE preprocessing

        Args:
            image_path: Path to input image
            output_path: Optional path to save preprocessed image

        Returns:
            Preprocessed grayscale image
        """
        print(f"Preprocessing image: {image_path}")

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (contrast normalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Saved preprocessed image to: {output_path}")

        return img

    def run_ocr(self, image: np.ndarray) -> List[OCRLine]:
        """
        Run PaddleOCR and normalize results

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            List of detected OCR lines with normalized bboxes
        """
        print("Running PaddleOCR detection...")

        # Convert grayscale to BGR if needed (new PaddleOCR API requires 3-channel images)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        result = list(self.ocr.predict(image))

        if result is None or len(result) == 0:
            print("Warning: No text detected by PaddleOCR")
            return []

        lines = []
        # New PaddleOCR API returns OCRResult objects with dt_polys, rec_texts, rec_scores
        ocr_result = result[0]

        # Access the result attributes (new API format)
        polygons = (
            ocr_result.get("dt_polys", [])
            if isinstance(ocr_result, dict)
            else getattr(ocr_result, "dt_polys", [])
        )
        texts = (
            ocr_result.get("rec_texts", [])
            if isinstance(ocr_result, dict)
            else getattr(ocr_result, "rec_texts", [])
        )
        scores = (
            ocr_result.get("rec_scores", [])
            if isinstance(ocr_result, dict)
            else getattr(ocr_result, "rec_scores", [])
        )

        for polygon, text, conf in zip(polygons, texts, scores):
            # Convert polygon to axis-aligned bbox
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bbox = BBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

            lines.append(OCRLine(bbox=bbox, text=text, confidence=float(conf)))

        print(f"Detected {len(lines)} text lines")
        return lines

    def segment_columns(
        self, lines: List[OCRLine], image_width: int
    ) -> Tuple[List[OCRLine], List[OCRLine]]:
        """
        Segment lines into left and right columns using midline split

        Args:
            lines: List of OCR lines
            image_width: Width of the image

        Returns:
            Tuple of (left_lines, right_lines)
        """
        print("Segmenting columns...")
        page_mid_x = image_width / 2

        left_lines = []
        right_lines = []

        for line in lines:
            if line.bbox.cx < page_mid_x:
                line.column = "L"
                left_lines.append(line)
            else:
                line.column = "R"
                right_lines.append(line)

        # Sort by y_min (top to bottom), then x_min (left to right)
        left_lines.sort(key=lambda l: (l.bbox.y_min, l.bbox.x_min))
        right_lines.sort(key=lambda l: (l.bbox.y_min, l.bbox.x_min))

        print(
            f"Left column: {len(left_lines)} lines, Right column: {len(right_lines)} lines"
        )
        return left_lines, right_lines

    def _has_pos_tag(self, text: str) -> bool:
        """Check if text contains a POS tag like (сущ.), (гл.), etc."""
        return bool(POS_TAG_PATTERN.search(text))

    def _is_entry_start(self, text: str) -> bool:
        """
        Check if a line likely starts a new dictionary entry.
        Entry starts typically have: headword followed by POS tag or specific patterns.
        """
        # Check for POS tag presence
        if self._has_pos_tag(text):
            return True

        # Check for cross-reference pattern: "см." at the start
        if text.strip().startswith("см."):
            return False  # This is a continuation, not a new entry

        # Check for phrase markers like "И.B." or numbered examples
        if re.match(r"^[0-9]+\.", text.strip()):
            return False  # Numbered continuation

        return False

    def segment_entries(
        self, lines: List[OCRLine], column: str
    ) -> List[DictionaryEntry]:
        """
        Segment lines into dictionary entries using POS tag detection + gap/indent heuristics

        Args:
            lines: List of OCR lines in a column
            column: Column identifier ('L' or 'R')

        Returns:
            List of dictionary entries
        """
        if not lines:
            return []

        print(f"Segmenting entries in column {column}...")

        # Compute gap statistics
        gaps = [
            lines[i].bbox.y_min - lines[i - 1].bbox.y_max for i in range(1, len(lines))
        ]
        median_gap = np.median(gaps) if gaps else 0
        mad_gap = np.median([abs(g - median_gap) for g in gaps]) if gaps else 0

        # Compute indent statistics
        col_left_margin = min(l.bbox.x_min for l in lines)
        indents = [l.bbox.x_min - col_left_margin for l in lines]
        indent_threshold = np.percentile(indents, 25) if indents else 5

        print(f"  Median gap: {median_gap:.2f}, MAD: {mad_gap:.2f}")
        print(
            f"  Left margin: {col_left_margin:.2f}, Indent threshold: {indent_threshold:.2f}"
        )

        # Segment entries
        entries = []
        current_entry_lines = [lines[0]]

        for i in range(1, len(lines)):
            prev_line = lines[i - 1]
            curr_line = lines[i]

            gap = curr_line.bbox.y_min - prev_line.bbox.y_max
            indent = curr_line.bbox.x_min - col_left_margin
            is_left_aligned = indent < indent_threshold

            # Check if this line contains a POS tag (strong indicator of new entry)
            has_pos = self._has_pos_tag(curr_line.text)

            # Check if this starts a new entry using multiple heuristics:
            # Rule 1: Large gap between lines
            large_gap = gap > median_gap + 2 * mad_gap

            # Rule 2: Line has POS tag AND is reasonably left-aligned (not heavily indented)
            # This catches entries like "acьŋ (сущ.) долг" even without large gaps
            pos_at_margin = has_pos and is_left_aligned

            # Rule 3: Line has POS tag with moderate indent (sub-entries under same root)
            # Allow slightly indented lines with POS tags as new entries
            pos_with_small_indent = has_pos and indent < indent_threshold * 3

            is_new_entry = large_gap or pos_at_margin or pos_with_small_indent

            if is_new_entry:
                # Save previous entry
                entries.append(
                    self._create_entry(current_entry_lines, column, len(entries))
                )
                current_entry_lines = [curr_line]
            else:
                current_entry_lines.append(curr_line)

        # Add final entry
        if current_entry_lines:
            entries.append(
                self._create_entry(current_entry_lines, column, len(entries))
            )

        print(f"  Segmented into {len(entries)} entries")
        return entries

    def _create_entry(
        self, lines: List[OCRLine], column: str, idx: int
    ) -> DictionaryEntry:
        """Create dictionary entry from lines"""
        # Compute union bbox
        x_min = min(l.bbox.x_min for l in lines)
        y_min = min(l.bbox.y_min for l in lines)
        x_max = max(l.bbox.x_max for l in lines)
        y_max = max(l.bbox.y_max for l in lines)

        bbox = BBox(x_min, y_min, x_max, y_max)
        entry_id = f"{column}{idx:03d}"

        return DictionaryEntry(entry_id=entry_id, lines=lines, bbox=bbox, column=column)

    def generate_entry_crops(
        self, entries: List[DictionaryEntry], image: np.ndarray, padding: int = 20
    ):
        """
        Generate cropped images for each entry with padding

        Args:
            entries: List of dictionary entries
            image: Original image
            padding: Padding around entry bbox
        """
        print(f"Generating entry crops with {padding}px padding...")

        for entry in entries:
            # Add padding
            x_min = max(0, int(entry.bbox.x_min - padding))
            y_min = max(0, int(entry.bbox.y_min - padding))
            x_max = min(image.shape[1], int(entry.bbox.x_max + padding))
            y_max = min(image.shape[0], int(entry.bbox.y_max + padding))

            # Crop
            entry.crop_image = image[y_min:y_max, x_min:x_max]

    def visualize_stage1_raw_ocr(
        self, image: np.ndarray, lines: List[OCRLine], output_path: str
    ):
        """
        Stage 1: Visualize raw OCR detection boxes

        Args:
            image: Input image (grayscale)
            lines: List of OCR lines
            output_path: Path to save visualization
        """
        print("Creating Stage 1 visualization: Raw OCR boxes...")

        # Convert grayscale to BGR for colored visualization
        vis = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if len(image.shape) == 2
            else image.copy()
        )

        for line in lines:
            bbox = line.bbox
            # Draw green boxes
            cv2.rectangle(
                vis,
                (int(bbox.x_min), int(bbox.y_min)),
                (int(bbox.x_max), int(bbox.y_max)),
                (0, 255, 0),
                2,
            )
            # Add confidence score
            cv2.putText(
                vis,
                f"{line.confidence:.2f}",
                (int(bbox.x_min), int(bbox.y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        cv2.imwrite(output_path, vis)
        print(f"Saved to: {output_path}")

    def visualize_stage2_columns(
        self,
        image: np.ndarray,
        left_lines: List[OCRLine],
        right_lines: List[OCRLine],
        output_path: str,
    ):
        """
        Stage 2: Visualize column-separated boxes

        Args:
            image: Input image (grayscale)
            left_lines: Lines in left column
            right_lines: Lines in right column
            output_path: Path to save visualization
        """
        print("Creating Stage 2 visualization: Column segmentation...")

        vis = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if len(image.shape) == 2
            else image.copy()
        )

        # Draw left column in blue
        for line in left_lines:
            bbox = line.bbox
            cv2.rectangle(
                vis,
                (int(bbox.x_min), int(bbox.y_min)),
                (int(bbox.x_max), int(bbox.y_max)),
                (255, 0, 0),
                2,
            )

        # Draw right column in red
        for line in right_lines:
            bbox = line.bbox
            cv2.rectangle(
                vis,
                (int(bbox.x_min), int(bbox.y_min)),
                (int(bbox.x_max), int(bbox.y_max)),
                (0, 0, 255),
                2,
            )

        # Add legend
        cv2.putText(
            vis, "Left Column", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
        )
        cv2.putText(
            vis, "Right Column", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        cv2.imwrite(output_path, vis)
        print(f"Saved to: {output_path}")

    def visualize_stage3_entries(
        self, image: np.ndarray, entries: List[DictionaryEntry], output_path: str
    ):
        """
        Stage 3: Visualize entry-grouped boxes with IDs

        Args:
            image: Input image (grayscale)
            entries: List of dictionary entries
            output_path: Path to save visualization
        """
        print("Creating Stage 3 visualization: Entry segmentation...")

        vis = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if len(image.shape) == 2
            else image.copy()
        )

        # Color map for entries
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for i, entry in enumerate(entries):
            color = colors[i % len(colors)]
            bbox = entry.bbox

            # Draw thick entry bounding box
            cv2.rectangle(
                vis,
                (int(bbox.x_min), int(bbox.y_min)),
                (int(bbox.x_max), int(bbox.y_max)),
                color,
                3,
            )

            # Add entry ID
            cv2.putText(
                vis,
                entry.entry_id,
                (int(bbox.x_min), int(bbox.y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            # Draw individual line boxes within entry
            for line in entry.lines:
                lb = line.bbox
                cv2.rectangle(
                    vis,
                    (int(lb.x_min), int(lb.y_min)),
                    (int(lb.x_max), int(lb.y_max)),
                    color,
                    1,
                )

        cv2.imwrite(output_path, vis)
        print(f"Saved to: {output_path}")

    def save_ocr_results(self, lines: List[OCRLine], output_path: str):
        """Save OCR results to JSON"""
        data = [line.to_dict() for line in lines]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved OCR results to: {output_path}")

    def save_entries(self, entries: List[DictionaryEntry], output_path: str):
        """Save entry metadata to JSON"""
        data = [entry.to_dict() for entry in entries]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved entries metadata to: {output_path}")

    def save_entry_crops(self, entries: List[DictionaryEntry], output_dir: Path):
        """Save individual entry crop images"""
        crops_dir = output_dir / "entry_crops"
        crops_dir.mkdir(exist_ok=True, parents=True)

        print(f"Saving {len(entries)} entry crops...")
        for entry in entries:
            if entry.crop_image is not None:
                crop_path = crops_dir / f"{entry.entry_id}.png"
                cv2.imwrite(str(crop_path), entry.crop_image)

        print(f"Saved entry crops to: {crops_dir}")

    def extract_entries(
        self, image_path: str, output_dir: str, padding: int = 20
    ) -> List[DictionaryEntry]:
        """
        Main pipeline: Extract dictionary entries from image

        Args:
            image_path: Path to input dictionary page image
            output_dir: Directory to save outputs
            padding: Padding for entry crops

        Returns:
            List of extracted dictionary entries
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        print("=" * 70)
        print("PADDLEOCR DICTIONARY ENTRY EXTRACTION PIPELINE")
        print("=" * 70)

        # Step 1: Preprocessing
        preprocessed = self.preprocess_image(
            image_path, str(output_path / "preprocessed.png")
        )

        # Step 2: Run OCR
        lines = self.run_ocr(preprocessed)
        if not lines:
            print("Error: No text detected!")
            return []

        # Save raw OCR results
        self.save_ocr_results(lines, str(output_path / "ocr_results.json"))

        # Visualization Stage 1: Raw OCR
        self.visualize_stage1_raw_ocr(
            preprocessed, lines, str(output_path / "stage1_raw_ocr.png")
        )

        # Step 3: Column segmentation
        left_lines, right_lines = self.segment_columns(lines, preprocessed.shape[1])

        # Visualization Stage 2: Columns
        self.visualize_stage2_columns(
            preprocessed,
            left_lines,
            right_lines,
            str(output_path / "stage2_columns.png"),
        )

        # Step 4: Entry segmentation
        left_entries = self.segment_entries(left_lines, "L")
        right_entries = self.segment_entries(right_lines, "R")
        all_entries = left_entries + right_entries

        # Step 5: Generate entry crops
        self.generate_entry_crops(all_entries, preprocessed, padding)

        # Visualization Stage 3: Entries
        self.visualize_stage3_entries(
            preprocessed, all_entries, str(output_path / "stage3_entries.png")
        )

        # Save outputs
        self.save_entries(all_entries, str(output_path / "entries.json"))
        self.save_entry_crops(all_entries, output_path)

        print("=" * 70)
        print(f"EXTRACTION COMPLETE")
        print(f"Total entries extracted: {len(all_entries)}")
        print(f"  Left column: {len(left_entries)} entries")
        print(f"  Right column: {len(right_entries)} entries")
        print(f"Output directory: {output_path}")
        print("=" * 70)

        return all_entries


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Extract dictionary entries using PaddleOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python paddle_ocr.py -i page.png -o paddle_outputs
  
  # With custom padding and language
  python paddle_ocr.py -i page.png -o paddle_outputs --padding 30 --text-rec-model cyrillic_PP-OCRv3_mobile_rec
  
  # Use GPU acceleration
  python paddle_ocr.py -i page.png -o paddle_outputs --use-gpu
  
  # Fine-tune detection granularity for more detailed detection
  python paddle_ocr.py -i page.png -o paddle_outputs --det-db-thresh 0.2 --det-db-box-thresh 0.5 --det-db-unclip-ratio 1.3
  
  # Higher resolution detection for small text
  python paddle_ocr.py -i page.png -o paddle_outputs --det-limit-side-len 1280
        """,
    )

    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Input dictionary page image"
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )

    parser.add_argument(
        "--text-det-model",
        type=str,
        default=None,
        help="Text detection model name (default: PP-OCRv5_server_det). Options: PP-OCRv5_mobile_det, PP-OCRv4_server_det, etc.",
    )

    parser.add_argument(
        "--text-rec-model",
        type=str,
        default="cyrillic_PP-OCRv3_mobile_rec",
        help="Text recognition model name (default: cyrillic_PP-OCRv3_mobile_rec with 94.28%% accuracy). Options: cyrillic_PP-OCRv5_mobile_rec, latin_PP-OCRv5_mobile_rec, en_PP-OCRv5_mobile_rec, etc.",
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding around entry crops in pixels (default: 20)",
    )

    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU acceleration for OCR"
    )

    # Detection granularity parameters
    parser.add_argument(
        "--det-db-thresh",
        type=float,
        default=0.3,
        help="DB detection threshold (default: 0.3). Lower values (0.1-0.2) detect more/fainter text.",
    )

    parser.add_argument(
        "--det-db-box-thresh",
        type=float,
        default=0.6,
        help="Minimum confidence for detected boxes (default: 0.6). Lower values keep more boxes.",
    )

    parser.add_argument(
        "--det-db-unclip-ratio",
        type=float,
        default=1.5,
        help="Box expansion ratio (default: 1.5). Lower values (1.0-1.3) give tighter boxes for finer granularity.",
    )

    parser.add_argument(
        "--det-limit-side-len",
        type=int,
        default=960,
        help="Max image side length for detection (default: 960). Higher values (1280+) better for small text.",
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.image).exists():
        print(f"Error: Input image not found: {args.image}")
        return 1

    try:
        # Run extraction
        extractor = PaddleOCRExtractor(
            use_gpu=args.use_gpu,
            text_detection_model_name=args.text_det_model,
            text_recognition_model_name=args.text_rec_model,
            det_db_thresh=args.det_db_thresh,
            det_db_box_thresh=args.det_db_box_thresh,
            det_db_unclip_ratio=args.det_db_unclip_ratio,
            det_limit_side_len=args.det_limit_side_len,
        )
        entries = extractor.extract_entries(args.image, args.output_dir, args.padding)

        if entries:
            print(f"\n✓ Successfully extracted {len(entries)} entries")
            return 0
        else:
            print("\n✗ No entries extracted")
            return 1

    except Exception as e:
        print(f"\n✗ Error during extraction: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
