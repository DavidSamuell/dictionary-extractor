"""
PaddleOCR Vision-Language (VL) backend.
Migrated from src/paddle_ocr_vl.py.
"""

import json
from pathlib import Path
from typing import Optional

from dictextractor.ocr.base import OCRBackend
from dictextractor.schemas.ocr_result import BBox, OCRBlock, OCRPageResult


class PaddleOCRVLBackend(OCRBackend):
    """
    OCR backend using PaddleOCR's Vision-Language model for layout-aware
    document understanding. Produces labelled text blocks with bounding boxes.
    """

    def __init__(self):
        from paddleocr import PaddleOCRVL
        self._pipeline = PaddleOCRVL()

    @property
    def name(self) -> str:
        return "paddle_vl"

    @property
    def supports_layout_analysis(self) -> bool:
        return True

    def run(self, image_path: str, output_dir: Optional[str] = None, save_crops: bool = True, **kwargs) -> OCRPageResult:
        """
        Run PaddleOCR VL on a single image.

        Args:
            image_path: Path to the input image.
            output_dir: Directory to save JSON, markdown, image outputs, and crops.
            save_crops: Whether to save cropped text blocks to output_dir.

        Returns:
            OCRPageResult with one OCRBlock per detected layout block.
        """
        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        input_name = Path(image_path).stem
        results = list(self._pipeline.predict(image_path))

        json_data = {}
        for res in results:
            if out_path:
                res.save_to_json(save_path=str(out_path))
                res.save_to_markdown(save_path=str(out_path))
                res.save_to_img(save_path=str(out_path))

        # Load saved JSON for block parsing
        if out_path:
            json_path = out_path / f"{input_name}_res.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

        blocks = self._parse_blocks(json_data)

        if out_path and save_crops:
            self._save_text_block_crops(image_path, json_data, out_path / f"{input_name}_crops")

        return OCRPageResult(source_image=image_path, backend=self.name, blocks=blocks)

    def _parse_blocks(self, json_data: dict) -> list:
        from dictextractor.schemas.ocr_result import OCRBlock, BBox
        blocks = []
        for block in json_data.get("parsing_res_list", []):
            bbox_raw = block.get("block_bbox", [])
            if len(bbox_raw) != 4:
                continue
            bbox = BBox(x_min=bbox_raw[0], y_min=bbox_raw[1], x_max=bbox_raw[2], y_max=bbox_raw[3])
            blocks.append(OCRBlock(
                block_id=block.get("block_id", -1),
                bbox=bbox,
                label=block.get("block_label", "text"),
                content=block.get("block_content", ""),
            ))
        return blocks

    def _save_text_block_crops(self, image_path: str, json_data: dict, crops_dir: Path):
        """Crop each text block from the image and save as paired .png + .md files."""
        from PIL import Image
        crops_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(image_path)

        text_blocks = [b for b in json_data.get("parsing_res_list", []) if b.get("block_label") == "text"]
        print(f"\nProcessing {len(text_blocks)} text blocks → {crops_dir}")

        for block in text_blocks:
            block_id = block.get("block_id", -1)
            content = block.get("block_content", "")
            bbox = block.get("block_bbox", [])
            if len(bbox) == 4 and content:
                x1, y1, x2, y2 = bbox
                cropped = img.crop((x1, y1, x2, y2))
                cropped.save(crops_dir / f"block_{block_id:03d}_text.png")
                (crops_dir / f"block_{block_id:03d}_text.md").write_text(content, encoding="utf-8")
                print(f"  Block {block_id}: saved crop ({x2-x1}x{y2-y1}px)")

        print(f"Text block crops saved to: {crops_dir}")
