"""Tests for VLM OCR prompt helpers."""

from __future__ import annotations

from dictextractor.ocr.vlm.prompts import build_stage1_context_prompt


def test_build_stage1_context_prompt_includes_context() -> None:
    prompt = build_stage1_context_prompt(
        alphabet_text="a\nb",
        ocr_hint="hint line",
    )
    assert "<alphabet>" in prompt
    assert "<ocr_reference>" in prompt


def test_build_stage1_context_prompt_bare() -> None:
    prompt = build_stage1_context_prompt()
    assert "<alphabet>" not in prompt
    assert "<ocr_reference>" not in prompt
