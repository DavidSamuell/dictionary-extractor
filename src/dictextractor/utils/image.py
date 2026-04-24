"""
Image loading and encoding helpers shared across OCR and extraction modules.
"""

import base64
from pathlib import Path


def encode_image_base64(image_path: str) -> str:
    """
    Encode an image file to a base64 string suitable for LLM API calls.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64-encoded string of the image bytes.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_data_url(image_path: str, mime_type: str = "image/png") -> str:
    """
    Build a data URL from an image file.

    Args:
        image_path: Path to the image file.
        mime_type: MIME type for the data URL (default: 'image/png').

    Returns:
        Data URL string of the form 'data:<mime>;base64,<data>'.
    """
    encoded = encode_image_base64(image_path)
    return f"data:{mime_type};base64,{encoded}"


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
