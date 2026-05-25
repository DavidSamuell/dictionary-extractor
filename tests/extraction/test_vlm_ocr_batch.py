"""Tests for multi-experiment VLM OCR batch helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from dictextractor.extraction.vlm_ocr import (
    alphabet_disabled_for_experiment,
    run_vlm_ocr_batch,
)


def test_alphabet_disabled_for_experiment_global_flag() -> None:
    assert alphabet_disabled_for_experiment(
        "GLM-OCR-flat_alpha",
        global_no_alphabet=True,
    )


def test_alphabet_disabled_for_experiment_noalpha_suffix() -> None:
    assert alphabet_disabled_for_experiment(
        "GLM-OCR-flat_noalpha",
        global_no_alphabet=False,
    )
    assert not alphabet_disabled_for_experiment(
        "GLM-OCR-flat_alpha",
        global_no_alphabet=False,
    )


@patch("dictextractor.extraction.vlm_ocr.run_vlm_ocr_entry", return_value=0)
@patch("dictextractor.extraction.vlm_ocr.create_vlm_runner")
def test_run_vlm_ocr_batch_loops_experiments_with_one_load(
    mock_create: MagicMock,
    mock_entry: MagicMock,
    tmp_path: Path,
) -> None:
    entry = tmp_path / "Lang-English"
    snippets = entry / "snippets"
    snippets.mkdir(parents=True)
    (snippets / "page_1.pdf").write_bytes(b"%PDF-1.4")
    (entry / "alphabet.txt").write_text("abc", encoding="utf-8")
    (entry / "mathpix").mkdir()
    (entry / "mathpix" / "page_1.txt").write_text("hint", encoding="utf-8")

    runner = MagicMock()
    runner.spec.key = "glm-ocr"
    runner.spec.product_label = "GLM-OCR"
    mock_create.return_value = runner

    args = Namespace(
        vlm_model="glm-ocr",
        experiment_names=["GLM-OCR-flat_alpha", "GLM-OCR-flat_noalpha"],
        experiment_name="GLM-OCR-flat_alpha",
        no_alphabet=False,
        no_ocr_hint=True,
        overwrite=False,
        limit=None,
        vlm_dpi=200,
        ocr_text=None,
        glm_ocr_prompt=None,
        glm_max_new_tokens=None,
        glm_backend="transformers",
    )

    rc = run_vlm_ocr_batch(args, [entry])

    assert rc == 0
    runner.load.assert_called_once()
    runner.unload.assert_called_once()
    assert mock_entry.call_count == 2
    assert args.experiment_name == "GLM-OCR-flat_alpha"
    assert args.no_alphabet is False
