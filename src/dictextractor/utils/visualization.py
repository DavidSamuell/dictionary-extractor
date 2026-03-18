"""
Visualization helpers for OCR output inspection and debugging.
Includes block annotation (from annotate_ocr_blocks.py) and
multi-stage PaddleOCR pipeline visualizations (from paddle_ocr.py).
"""

from typing import List
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Block annotation (PaddleOCR VL JSON output)
# ---------------------------------------------------------------------------

_BLOCK_LABEL_COLORS = {
    "paragraph_title": (0, 0, 255),    # Red
    "text":            (0, 255, 0),    # Green
    "image":           (255, 0, 0),    # Blue
    "table":           (255, 255, 0),  # Cyan
    "figure":          (255, 0, 255),  # Magenta
    "formula":         (0, 255, 255),  # Yellow
    "default":         (255, 128, 0),  # Orange
}


def annotate_image(
    image_path: str,
    json_data: dict,
    output_path: str,
    thickness: int = 3,
    font_scale: float = 0.8,
) -> np.ndarray:
    """
    Draw colour-coded bounding boxes and labels from PaddleOCR VL JSON onto an image.

    Args:
        image_path: Path to the source image.
        json_data: Parsed JSON dict containing a 'parsing_res_list' key.
        output_path: Where to save the annotated image.
        thickness: Rectangle and text stroke thickness.
        font_scale: OpenCV font scale for label text.

    Returns:
        Annotated BGR image array.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    blocks = json_data.get("parsing_res_list", [])
    print(f"Found {len(blocks)} blocks to annotate")

    for block in blocks:
        block_label = block.get("block_label", "unknown")
        block_bbox = block.get("block_bbox", [])
        block_id = block.get("block_id", -1)
        block_order = block.get("block_order", -1)

        if len(block_bbox) != 4:
            print(f"Skipping block {block_id}: invalid bbox {block_bbox}")
            continue

        x_min, y_min, x_max, y_max = map(int, block_bbox)
        color = _BLOCK_LABEL_COLORS.get(block_label, _BLOCK_LABEL_COLORS["default"])

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        label_text = block_label
        if block_order is not None:
            label_text += f" [{block_order}]"

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            img,
            (x_min, y_min - text_h - baseline - 5),
            (x_min + text_w, y_min),
            color,
            -1,
        )
        cv2.putText(
            img,
            label_text,
            (x_min, y_min - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        print(f"  Block {block_id}: {block_label} at {block_bbox}")

    cv2.imwrite(str(output_path), img)
    print(f"\nSaved annotated image to: {output_path}")
    return img


# ---------------------------------------------------------------------------
# PaddleOCR traditional pipeline stage visualizations
# ---------------------------------------------------------------------------

def visualize_stage1_raw_ocr(
    image: np.ndarray, lines: list, output_path: str
) -> None:
    """Stage 1: draw raw OCR detection boxes (green) with confidence scores."""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for line in lines:
        b = line.bbox
        cv2.rectangle(vis, (int(b.x_min), int(b.y_min)), (int(b.x_max), int(b.y_max)), (0, 255, 0), 2)
        cv2.putText(vis, f"{line.confidence:.2f}", (int(b.x_min), int(b.y_min - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(output_path, vis)
    print(f"Stage 1 visualization saved to: {output_path}")


def visualize_stage2_columns(
    image: np.ndarray, left_lines: list, right_lines: list, output_path: str
) -> None:
    """Stage 2: draw left column boxes (blue) and right column boxes (red)."""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for line in left_lines:
        b = line.bbox
        cv2.rectangle(vis, (int(b.x_min), int(b.y_min)), (int(b.x_max), int(b.y_max)), (255, 0, 0), 2)
    for line in right_lines:
        b = line.bbox
        cv2.rectangle(vis, (int(b.x_min), int(b.y_min)), (int(b.x_max), int(b.y_max)), (0, 0, 255), 2)
    cv2.putText(vis, "Left Column",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(vis, "Right Column", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(output_path, vis)
    print(f"Stage 2 visualization saved to: {output_path}")


def visualize_stage3_entries(
    image: np.ndarray, entries: list, output_path: str
) -> None:
    """Stage 3: draw entry bounding boxes with cycling colours and entry IDs."""
    _ENTRY_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for i, entry in enumerate(entries):
        color = _ENTRY_COLORS[i % len(_ENTRY_COLORS)]
        b = entry.bbox
        cv2.rectangle(vis, (int(b.x_min), int(b.y_min)), (int(b.x_max), int(b.y_max)), color, 3)
        cv2.putText(vis, entry.entry_id, (int(b.x_min), int(b.y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        for line in entry.lines:
            lb = line.bbox
            cv2.rectangle(vis, (int(lb.x_min), int(lb.y_min)), (int(lb.x_max), int(lb.y_max)), color, 1)
    cv2.imwrite(output_path, vis)
    print(f"Stage 3 visualization saved to: {output_path}")
