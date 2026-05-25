"""Tests for markdown-only stage-1 flat backfill script."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))

from backfill_stage1_flat_from_markdown import (  # noqa: E402
    discover_markdown_page_dirs,
    main,
)


def _make_page(tmp_path: Path, *, experiment: str, stem: str, md_text: str) -> Path:
    page_dir = (
        tmp_path
        / "Lang-English"
        / "outputs"
        / "stage-1"
        / experiment
        / stem
    )
    page_dir.mkdir(parents=True)
    (page_dir / "output.md").write_text(md_text, encoding="utf-8")
    return page_dir


def test_discover_markdown_page_dirs_filters_to_markdown(tmp_path: Path) -> None:
    _make_page(tmp_path, experiment="Mathpix-OCR", stem="page_1", md_text="hello")
    no_md = (
        tmp_path
        / "Lang-English"
        / "outputs"
        / "stage-1"
        / "Mathpix-OCR"
        / "page_2"
    )
    no_md.mkdir(parents=True)

    found = discover_markdown_page_dirs(tmp_path, ["Mathpix-OCR"])

    assert len(found) == 1
    assert found[0][1] == "page_1"


def test_main_writes_flat_from_markdown(tmp_path: Path) -> None:
    page_dir = _make_page(
        tmp_path,
        experiment="MinerU2.5-Pro",
        stem="page_3",
        md_text="**bold** word",
    )

    rc = main(
        [
            "--samples-dir",
            str(tmp_path),
            "--experiment",
            "MinerU2.5-Pro",
            "--overwrite",
        ]
    )

    assert rc == 0
    flat_path = page_dir / "page_3_stage1_flat.txt"
    assert flat_path.is_file()
    assert "<b>bold</b>" in flat_path.read_text(encoding="utf-8")
