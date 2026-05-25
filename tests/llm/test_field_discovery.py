"""Tests for Pass 1 marker cheat sheet caching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dictextractor.llm.field_discovery import (
    load_gold_cheatsheet,
    load_or_discover_cheatsheet,
)
from dictextractor.schemas.field_cheatsheet import DictionaryMarkerCheatsheet, MarkerLine


def _sample_sheet() -> DictionaryMarkerCheatsheet:
    return DictionaryMarkerCheatsheet(
        dictionary_name="Test Dict",
        markers=[MarkerLine(marker="lx", description="headword")],
        rules=["One \\lx per main entry."],
    )


def test_load_or_discover_uses_experiment_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "outputs" / "stage-2" / "exp_a" / "field_cheatsheet.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps(_sample_sheet().model_dump(), ensure_ascii=False),
        encoding="utf-8",
    )

    def _should_not_run(**_kwargs: object) -> DictionaryMarkerCheatsheet:
        raise AssertionError("discover_field_cheatsheet should not be called")

    sheet = load_or_discover_cheatsheet(
        cache_path,
        force_refresh=False,
        transcription="ignored",
        sample_image=tmp_path / "page.png",
        intro_images=[],
        model="test/model",
    )
    assert sheet.dictionary_name == "Test Dict"


def test_load_or_discover_force_refresh_overwrites_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "outputs" / "stage-2" / "exp_a" / "field_cheatsheet.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps({"dictionary_name": "Stale", "markers": [], "rules": []}),
        encoding="utf-8",
    )

    refreshed = _sample_sheet()

    def _fake_discover(**_kwargs: object) -> DictionaryMarkerCheatsheet:
        return refreshed

    monkeypatch.setattr(
        "dictextractor.llm.field_discovery.discover_field_cheatsheet",
        _fake_discover,
    )

    sheet = load_or_discover_cheatsheet(
        cache_path,
        force_refresh=True,
        transcription="sample",
        sample_image=tmp_path / "page.png",
        intro_images=[],
        model="test/model",
    )
    assert sheet.dictionary_name == "Test Dict"
    reloaded = DictionaryMarkerCheatsheet.model_validate_json(
        cache_path.read_text(encoding="utf-8")
    )
    assert reloaded.dictionary_name == "Test Dict"


def test_load_gold_cheatsheet(tmp_path: Path) -> None:
    entry_dir = tmp_path / "Sample-English"
    gold_path = entry_dir / "outputs" / "stage-2-gold" / "field_cheatsheet.json"
    gold_path.parent.mkdir(parents=True)
    gold_path.write_text(
        json.dumps(_sample_sheet().model_dump(), ensure_ascii=False),
        encoding="utf-8",
    )

    sheet = load_gold_cheatsheet(entry_dir)
    assert sheet.dictionary_name == "Test Dict"
    assert sheet.markers[0].marker == "lx"


def test_load_gold_cheatsheet_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Gold field cheat sheet not found"):
        load_gold_cheatsheet(tmp_path / "Missing-English")
