"""
PaddleOCR traditional pipeline backend (detection + recognition).
Migrated from src/paddle_ocr.py.
"""

import cv2
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from dictextractor.ocr.base import OCRBackend
from dictextractor.schemas.ocr_result import BBox, OCRLine, OCRBlock, OCRPageResult
from dictextractor.utils.visualization import (
    visualize_stage1_raw_ocr,
    visualize_stage2_columns,
    visualize_stage3_entries,
)

POS_TAG_PATTERN = re.compile(r"\([а-яА-Яa-zA-Z]+\.?\)")


@dataclass
class _Entry:
    """Internal entry type used during segmentation (not exposed publicly)."""

    entry_id: str
    lines: List[OCRLine]
    bbox: BBox
    column: str
    crop_image: Optional[np.ndarray] = None

    def to_dict(self):
        return {
            "entry_id": self.entry_id,
            "lines": [l.to_dict() for l in self.lines],
            "bbox": self.bbox.to_dict(),
            "column": self.column,
        }


class PaddleOCRTraditional(OCRBackend):
    """
    OCR backend using PaddleOCR text detection + recognition.
    Performs column segmentation and entry grouping via gap statistics.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        text_detection_model_name: Optional[str] = None,
        text_recognition_model_name: str = "cyrillic_PP-OCRv3_mobile_rec",
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.6,
        det_db_unclip_ratio: float = 1.5,
        det_limit_side_len: int = 960,
        rec_score_thresh: float = 0.5,
    ):
        from paddleocr import PaddleOCR

        print("Initializing PaddleOCR...")
        self.rec_score_thresh = rec_score_thresh

        ocr_params = {
            "use_textline_orientation": True,
            "text_recognition_model_name": text_recognition_model_name,
            "det_db_thresh": det_db_thresh,
            "det_db_box_thresh": det_db_box_thresh,
            "det_db_unclip_ratio": det_db_unclip_ratio,
            "det_limit_side_len": det_limit_side_len,
        }
        if text_detection_model_name:
            ocr_params["text_detection_model_name"] = text_detection_model_name

        self._ocr = PaddleOCR(**ocr_params)
        print("PaddleOCR initialized successfully")

    @property
    def name(self) -> str:
        return "paddle_traditional"

    @property
    def supports_layout_analysis(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # OCRBackend interface
    # ------------------------------------------------------------------

    def run(self, image_path: str, output_dir: Optional[str] = None, padding: int = 20, **kwargs) -> OCRPageResult:
        """
        Run the full PaddleOCR pipeline and return a unified OCRPageResult.

        Side effects: If output_dir is provided, saves JSON results, crops,
        and stage visualization images to that directory.
        """
        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(exist_ok=True, parents=True)

        preprocessed = self._preprocess_image(
            image_path, str(out_path / "preprocessed.png") if out_path else None
        )
        lines = self._run_ocr(preprocessed)

        if out_path:
            self._save_ocr_results(lines, str(out_path / "ocr_results.json"))
            visualize_stage1_raw_ocr(preprocessed, lines, str(out_path / "stage1_raw_ocr.png"))

        left_lines, right_lines = self._segment_columns(lines, preprocessed.shape[1])

        if out_path:
            visualize_stage2_columns(preprocessed, left_lines, right_lines, str(out_path / "stage2_columns.png"))

        left_entries = self._segment_entries(left_lines, "L")
        right_entries = self._segment_entries(right_lines, "R")
        all_entries = left_entries + right_entries

        self._generate_entry_crops(all_entries, preprocessed, padding)

        if out_path:
            visualize_stage3_entries(preprocessed, all_entries, str(out_path / "stage3_entries.png"))
            self._save_entries(all_entries, str(out_path / "entries.json"))
            self._save_entry_crops(all_entries, out_path)

        # Build OCRPageResult: one OCRBlock per segmented entry
        blocks = []
        for entry in all_entries:
            block = OCRBlock(
                block_id=int(entry.entry_id[1:]),  # strip L/R prefix
                bbox=entry.bbox,
                lines=entry.lines,
                label="text",
                content="\n".join(l.text for l in entry.lines),
                crop_image=entry.crop_image,
            )
            blocks.append(block)

        return OCRPageResult(source_image=image_path, backend=self.name, blocks=blocks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        if output_path:
            cv2.imwrite(output_path, img)
        return img

    def _run_ocr(self, image: np.ndarray) -> List[OCRLine]:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        result = list(self._ocr.predict(image))
        if not result:
            return []

        ocr_result = result[0]

        def _get(obj, key):
            return obj.get(key, []) if isinstance(obj, dict) else getattr(obj, key, [])

        polygons = _get(ocr_result, "dt_polys")
        texts = _get(ocr_result, "rec_texts")
        scores = _get(ocr_result, "rec_scores")

        lines = []
        for polygon, text, conf in zip(polygons, texts, scores):
            if float(conf) < self.rec_score_thresh:
                continue
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bbox = BBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))
            lines.append(OCRLine(bbox=bbox, text=text, confidence=float(conf)))
        return lines

    def _segment_columns(self, lines: List[OCRLine], image_width: int) -> Tuple[List[OCRLine], List[OCRLine]]:
        mid = image_width / 2
        left, right = [], []
        for line in lines:
            if line.bbox.cx < mid:
                line.column = "L"
                left.append(line)
            else:
                line.column = "R"
                right.append(line)
        left.sort(key=lambda l: (l.bbox.y_min, l.bbox.x_min))
        right.sort(key=lambda l: (l.bbox.y_min, l.bbox.x_min))
        return left, right

    def _segment_entries(self, lines: List[OCRLine], column: str, gap_multiplier: float = 2.0) -> List[_Entry]:
        if not lines:
            return []
        gaps = [lines[i].bbox.y_min - lines[i - 1].bbox.y_max for i in range(1, len(lines))]
        median_gap = float(np.median(gaps)) if gaps else 0
        mad_gap = float(np.median([abs(g - median_gap) for g in gaps])) if gaps else 0

        entries, current = [], [lines[0]]
        for i in range(1, len(lines)):
            gap = lines[i].bbox.y_min - lines[i - 1].bbox.y_max
            if gap > median_gap + gap_multiplier * mad_gap:
                entries.append(self._create_entry(current, column, len(entries)))
                current = [lines[i]]
            else:
                current.append(lines[i])
        if current:
            entries.append(self._create_entry(current, column, len(entries)))
        return entries

    def _create_entry(self, lines: List[OCRLine], column: str, idx: int) -> _Entry:
        bbox = BBox(
            x_min=min(l.bbox.x_min for l in lines),
            y_min=min(l.bbox.y_min for l in lines),
            x_max=max(l.bbox.x_max for l in lines),
            y_max=max(l.bbox.y_max for l in lines),
        )
        return _Entry(entry_id=f"{column}{idx:03d}", lines=lines, bbox=bbox, column=column)

    def _generate_entry_crops(self, entries: List[_Entry], image: np.ndarray, padding: int = 20):
        for entry in entries:
            x_min = max(0, int(entry.bbox.x_min - padding))
            y_min = max(0, int(entry.bbox.y_min - padding))
            x_max = min(image.shape[1], int(entry.bbox.x_max + padding))
            y_max = min(image.shape[0], int(entry.bbox.y_max + padding))
            entry.crop_image = image[y_min:y_max, x_min:x_max]

    def _save_ocr_results(self, lines: List[OCRLine], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([l.to_dict() for l in lines], f, ensure_ascii=False, indent=2)

    def _save_entries(self, entries: List[_Entry], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in entries], f, ensure_ascii=False, indent=2)

    def _save_entry_crops(self, entries: List[_Entry], output_dir: Path):
        crops_dir = output_dir / "entry_crops"
        crops_dir.mkdir(exist_ok=True, parents=True)
        for entry in entries:
            if entry.crop_image is not None:
                cv2.imwrite(str(crops_dir / f"{entry.entry_id}.png"), entry.crop_image)
