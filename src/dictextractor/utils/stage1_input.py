"""Resolve Stage-1 transcript files for Stage-2 consumption."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

Stage1InputPreference = Literal["auto", "column", "flat"]


def stage1_tsv_path(page_dir: Path, stem: str) -> Path:
    """Column transcription TSV for a page."""
    return page_dir / f"{stem}_stage1.tsv"


def stage1_flat_path(page_dir: Path, stem: str) -> Path:
    """Flat transcription text for a page."""
    return page_dir / f"{stem}_stage1_flat.txt"


def stage1_transcript_kind(path: Path) -> Stage1InputPreference:
    """Return ``column`` or ``flat`` from the resolved file name."""
    if path.name.endswith("_stage1_flat.txt"):
        return "flat"
    return "column"


def resolve_stage1_transcript_path(
    page_dir: Path,
    stem: str,
    preference: Stage1InputPreference = "auto",
) -> Optional[Path]:
    """
    Pick the Stage-1 transcript file Stage 2 should read.

    Args:
        page_dir: ``outputs/stage-1/<experiment>/<stem>/``
        stem: Page stem (e.g. ``page_1``).
        preference: ``auto`` prefers column TSV, then flat; ``column`` / ``flat``
            require that artifact only.

    Returns:
        Resolved path, or ``None`` if nothing matches the preference.
    """
    tsv = stage1_tsv_path(page_dir, stem)
    flat = stage1_flat_path(page_dir, stem)
    if preference == "column":
        return tsv if tsv.is_file() else None
    if preference == "flat":
        return flat if flat.is_file() else None
    if tsv.is_file():
        return tsv
    if flat.is_file():
        return flat
    return None
