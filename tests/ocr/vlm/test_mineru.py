"""Tests for MinerU VLM OCR tuning helpers."""

from __future__ import annotations

import pytest

from dictextractor.ocr.vlm.mineru import (
    DEFAULT_MINERU_BATCH_SIZE,
    DEFAULT_MINERU_LAYOUT_MAX_NEW_TOKENS,
    DEFAULT_MINERU_MAX_NEW_TOKENS,
    build_mineru_sampling_params,
    resolve_mineru_backend,
    resolve_mineru_batch_size,
)


def test_resolve_mineru_backend_explicit() -> None:
    assert resolve_mineru_backend("vllm") == "vllm"
    assert resolve_mineru_backend("transformers") == "transformers"


def test_resolve_mineru_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_BACKEND", "vllm")
    assert resolve_mineru_backend(None) == "vllm"


def test_resolve_mineru_batch_size_prefers_explicit() -> None:
    assert resolve_mineru_batch_size(16) == 16


def test_resolve_mineru_batch_size_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_VL_BATCH_SIZE", "12")
    assert resolve_mineru_batch_size(None) == 12


def test_resolve_mineru_batch_size_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINERU_VL_BATCH_SIZE", raising=False)
    assert resolve_mineru_batch_size(None) == DEFAULT_MINERU_BATCH_SIZE


def test_build_mineru_sampling_params_caps_generation() -> None:
    pytest.importorskip("mineru_vl_utils")
    params = build_mineru_sampling_params(256, layout_max_new_tokens=4096)
    assert params["[default]"].max_new_tokens == 256
    assert params["text"].max_new_tokens == 256
    assert params["[layout]"].max_new_tokens == 4096


def test_build_mineru_sampling_params_defaults() -> None:
    pytest.importorskip("mineru_vl_utils")
    params = build_mineru_sampling_params()
    assert params["[default]"].max_new_tokens == DEFAULT_MINERU_MAX_NEW_TOKENS
    assert params["[layout]"].max_new_tokens == DEFAULT_MINERU_LAYOUT_MAX_NEW_TOKENS
