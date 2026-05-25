"""Tests for Paddle GenAI vLLM server auto-start helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dictextractor.ocr.vlm.paddle_genai_server import (
    DEFAULT_PADDLE_GENAI_PORT,
    PaddleGenaiServerManager,
    paddle_genai_server_url,
    resolve_paddle_genai_port,
    should_auto_start_paddle_genai_server,
)


def test_paddle_genai_server_url() -> None:
    assert paddle_genai_server_url("127.0.0.1", 8765) == "http://127.0.0.1:8765/v1"


def test_resolve_paddle_genai_port_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PADDLE_VL_SERVER_PORT", "9001")
    assert resolve_paddle_genai_port(None) == 9001


def test_resolve_paddle_genai_port_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PADDLE_VL_SERVER_PORT", raising=False)
    assert resolve_paddle_genai_port(None) == DEFAULT_PADDLE_GENAI_PORT


def test_should_auto_start_paddle_only_for_paddle_model() -> None:
    args = Namespace(
        vlm_model="mineru2.5-pro",
        paddle_auto_vllm_server=True,
        paddle_vl_rec_server_url=None,
    )
    assert should_auto_start_paddle_genai_server(args) is False


def test_should_auto_start_respects_external_url() -> None:
    args = Namespace(
        vlm_model="paddleocr-vl-1.5",
        paddle_auto_vllm_server=True,
        paddle_vl_rec_server_url="http://127.0.0.1:9999/v1",
    )
    assert should_auto_start_paddle_genai_server(args) is False


def test_should_auto_start_default_on() -> None:
    args = Namespace(
        vlm_model="paddleocr-vl-1.5",
        paddle_auto_vllm_server=True,
        paddle_vl_rec_server_url=None,
    )
    assert should_auto_start_paddle_genai_server(args) is True


@patch("dictextractor.ocr.vlm.paddle_genai_server._health_ok", return_value=True)
def test_manager_reuses_healthy_server(mock_health: MagicMock) -> None:
    mgr = PaddleGenaiServerManager(port=8765, server_python=Path("/bin/python"))
    url = mgr.start()
    assert url == "http://127.0.0.1:8765/v1"
    mock_health.assert_called_once()
    mgr.stop()
