"""Non-empty artifact checks for VLM OCR resume logic."""

from __future__ import annotations

import json
from pathlib import Path


def file_has_non_empty_text(path: Path) -> bool:
    """Return True if *path* exists and contains non-whitespace text."""
    if not path.is_file():
        return False
    return bool(path.read_text(encoding="utf-8").strip())


def paddle_res_json_has_blocks(path: Path) -> bool:
    """Return True if a Paddle ``*_res.json`` has at least one non-empty block."""
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    for item in data.get("parsing_res_list") or []:
        if str(item.get("block_content") or "").strip():
            return True
    return False


def paddle_page_has_content(page_dir: Path, *, stem: str) -> bool:
    """True when Paddle OCR produced parseable non-empty layout blocks."""
    candidates = [page_dir / f"{stem}_res.json", *sorted(page_dir.glob("*_res.json"))]
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if paddle_res_json_has_blocks(path):
            return True
    return False


def _text_result_json_has_content(page_dir: Path) -> bool:
    """True when ``result.json`` contains a non-empty ``text`` field."""
    result_path = page_dir / "result.json"
    if not result_path.is_file():
        return False
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        text = data.get("text")
        return isinstance(text, str) and bool(text.strip())
    except json.JSONDecodeError:
        return False


def glm_page_has_content(page_dir: Path) -> bool:
    """True when GLM-OCR produced non-empty transcript text."""
    if file_has_non_empty_text(page_dir / "output.txt"):
        return True
    if file_has_non_empty_text(page_dir / "output.md"):
        return True
    return _text_result_json_has_content(page_dir)
