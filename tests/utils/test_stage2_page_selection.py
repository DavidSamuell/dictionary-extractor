"""Tests for Stage-2 one-page selection."""

from __future__ import annotations

from pathlib import Path

from dictextractor.utils.stage2_page_selection import (
    list_stage2_gold_stems,
    select_one_stage2_page,
)


def test_list_stage2_gold_stems(tmp_path: Path) -> None:
    (tmp_path / "stage-2-gold" / "page_3").mkdir(parents=True)
    (tmp_path / "stage-2-gold" / "page_3" / "page_3.mdf.txt").write_text(
        "\\lx test", encoding="utf-8"
    )
    (tmp_path / "stage-2-gold" / "page_9").mkdir(parents=True)

    assert list_stage2_gold_stems(tmp_path) == ["page_3"]


def test_select_prefers_stage2_gold_page(tmp_path: Path) -> None:
    snippets = [
        tmp_path / "snippets" / "page_1.png",
        tmp_path / "snippets" / "page_3.png",
    ]
    for path in snippets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    outputs = tmp_path / "outputs"
    (outputs / "stage-2-gold" / "page_3").mkdir(parents=True)
    (outputs / "stage-2-gold" / "page_3" / "page_3.mdf.txt").write_text(
        "\\lx test", encoding="utf-8"
    )

    selected = select_one_stage2_page(snippets, outputs, "flat")
    assert selected == [snippets[1]]


def test_select_falls_back_to_first_stage1_gold(tmp_path: Path) -> None:
    snippets = [
        tmp_path / "snippets" / "page_1.png",
        tmp_path / "snippets" / "page_2.png",
    ]
    for path in snippets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    outputs = tmp_path / "outputs"
    gold_page = outputs / "stage-1-gold" / "page_2"
    gold_page.mkdir(parents=True)
    (gold_page / "page_2_stage1_GOLD_flat.txt").write_text("lemma", encoding="utf-8")

    selected = select_one_stage2_page(snippets, outputs, "flat")
    assert selected == [snippets[1]]


def test_select_falls_back_to_first_snippet(tmp_path: Path) -> None:
    snippets = [
        tmp_path / "snippets" / "page_11.png",
        tmp_path / "snippets" / "page_10.png",
    ]
    for path in snippets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    selected = select_one_stage2_page(snippets, tmp_path / "outputs", "flat")
    assert selected == [snippets[1]]


def test_select_uses_numeric_not_lexicographic_order(tmp_path: Path) -> None:
    snippets = [
        tmp_path / "snippets" / "page_244.png",
        tmp_path / "snippets" / "page_74.png",
        tmp_path / "snippets" / "page_75.png",
    ]
    for path in snippets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    selected = select_one_stage2_page(snippets, tmp_path / "outputs", "flat")
    assert selected == [snippets[1]]


def test_select_stage1_gold_uses_lowest_page_number(tmp_path: Path) -> None:
    snippets = [
        tmp_path / "snippets" / "page_75.png",
        tmp_path / "snippets" / "page_74.png",
        tmp_path / "snippets" / "page_244.png",
    ]
    for path in snippets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    outputs = tmp_path / "outputs"
    for stem in ("page_74", "page_244"):
        gold_page = outputs / "stage-1-gold" / stem
        gold_page.mkdir(parents=True)
        (gold_page / f"{stem}_stage1_GOLD_flat.txt").write_text("lemma", encoding="utf-8")

    selected = select_one_stage2_page(snippets, outputs, "flat")
    assert selected == [snippets[1]]
