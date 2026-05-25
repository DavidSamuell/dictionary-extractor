"""
Image loading and encoding helpers shared across OCR and extraction modules.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Anthropic (via OpenRouter) rejects inline images above 5 MB.
_MAX_LLM_IMAGE_BYTES = 4_500_000


def encode_image_base64(image_path: str) -> str:
    """
    Encode an image file to a base64 string suitable for LLM API calls.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64-encoded string of the image bytes.
    """
    raw, _ = _read_bytes_for_llm(Path(image_path))
    return base64.b64encode(raw).decode("utf-8")


def image_data_url(image_path: str, mime_type: str = "image/png") -> str:
    """
    Build a data URL from an image file.

    Large raster images are re-encoded as JPEG so provider limits (e.g. Anthropic
    5 MB) are respected.

    Args:
        image_path: Path to the image file.
        mime_type: MIME type for the data URL (default: 'image/png').

    Returns:
        Data URL string of the form 'data:<mime>;base64,<data>'.
    """
    raw, out_mime = _read_bytes_for_llm(Path(image_path), mime_type=mime_type)
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{out_mime};base64,{encoded}"


def _read_bytes_for_llm(
    path: Path,
    *,
    mime_type: str = "image/png",
) -> tuple[bytes, str]:
    """Return payload bytes and MIME type, compressing rasters when needed."""
    raw = path.read_bytes()
    if mime_type == "application/pdf" or len(raw) <= _MAX_LLM_IMAGE_BYTES:
        return raw, mime_type

    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    original_size = len(raw)
    for scale in (1.0, 0.85, 0.7, 0.55, 0.45):
        candidate = img
        if scale < 1.0:
            width, height = img.size
            candidate = img.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        for quality in (85, 70, 55, 40):
            buf = io.BytesIO()
            candidate.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            if len(data) <= _MAX_LLM_IMAGE_BYTES:
                logger.info(
                    "Compressed %s for LLM API: %d -> %d bytes (JPEG q=%d scale=%.2f)",
                    path.name,
                    original_size,
                    len(data),
                    quality,
                    scale,
                )
                return data, "image/jpeg"

    raise ValueError(
        f"Could not compress {path} below {_MAX_LLM_IMAGE_BYTES} bytes for LLM API"
    )


def resolve_mime_type(image_path: str) -> str:
    """
    Infer the MIME type from the file extension.

    Args:
        image_path: Path to the image file.

    Returns:
        MIME type string.
    """
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
    }
    return mime_map.get(ext, "image/png")
