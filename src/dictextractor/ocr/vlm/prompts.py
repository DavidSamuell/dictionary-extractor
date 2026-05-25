"""Prompt helpers for VLM OCR backends that accept Stage-1-style context."""

from __future__ import annotations

from pathlib import Path

from dictextractor.llm.prompts import stage_1_user
from dictextractor.utils.io import read_docx_text

_OCR_HINT_EXTS = (".md", ".mmd", ".txt", ".docx")
_IMAGE_ALPHABET_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def find_ocr_hint_file(ocr_dir: Path, stem: str) -> Path | None:
    """Find an OCR hint file in ``ocr_dir`` whose stem matches the page stem."""
    for ext in _OCR_HINT_EXTS:
        candidate = ocr_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_alphabet_text(alphabet_path: str | None) -> str:
    """Load a text alphabet file for prompt injection (empty if unset or image-only)."""
    if not alphabet_path:
        return ""
    path = Path(alphabet_path)
    suffix = path.suffix.lower()
    if suffix in _IMAGE_ALPHABET_EXTS:
        return ""
    if suffix == ".docx":
        return read_docx_text(str(path))
    return path.read_text(encoding="utf-8")


def load_ocr_hint_text(ocr_file: Path | None) -> str:
    """Load OCR hint text from a per-page hint file."""
    if ocr_file is None:
        return ""
    suffix = ocr_file.suffix.lower()
    if suffix == ".docx":
        return read_docx_text(str(ocr_file))
    return ocr_file.read_text(encoding="utf-8")


def build_stage1_context_prompt(
    *,
    alphabet_text: str = "",
    ocr_hint: str = "",
) -> str:
    """Build Stage-1 user context (alphabet + OCR hint) for prompt-capable VLMs."""
    return stage_1_user(alphabet_text=alphabet_text, ocr_hint=ocr_hint).strip()
