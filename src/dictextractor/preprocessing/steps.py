"""
Pure preprocessing step functions.
Each function takes a grayscale numpy array and returns a transformed array.
The pipeline.py module composes these steps.
"""

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale. Returns unchanged if already grayscale."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def deskew(img: np.ndarray, skip_threshold: float = 0.5) -> tuple[np.ndarray, float]:
    """
    Deskew the image using Hough line detection.

    Args:
        img: Grayscale image.
        skip_threshold: Do not rotate if detected angle is smaller than this (degrees).

    Returns:
        Tuple of (corrected image, applied angle in degrees). Angle is 0.0 if skipped.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return img, 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return img, 0.0

    skew_angle = float(np.median(angles))

    if abs(skew_angle) <= skip_threshold:
        return img, 0.0

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    border_value = 255 if len(img.shape) == 2 else (255, 255, 255)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated, skew_angle


def denoise(img: np.ndarray, d: int = 5, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter denoising. Preserves edges while reducing noise.

    Args:
        img: Grayscale image.
        d: Pixel neighbourhood diameter.
        sigma_color: Filter sigma in colour space.
        sigma_space: Filter sigma in coordinate space.
    """
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def contrast_normalization(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Helps with small Cyrillic characters on low-contrast scans.

    Args:
        img: Grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_size: Grid cell size for histogram equalization.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


def sharpen(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """
    Apply unsharp mask sharpening.

    Args:
        img: Grayscale image.
        amount: Sharpening strength (0.5–2.0 recommended).
    """
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.0 + amount, gaussian, -amount, 0)
