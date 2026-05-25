"""Shared protocol for Pass 1 outputs rendered into Pass 2 prompts."""

from __future__ import annotations

from typing import Protocol


class FieldMapPrompt(Protocol):
    """Pass 1 field map — profile or cheat sheet — with a prompt renderer."""

    def format_prompt_block(self) -> str:
        """Render as Pass 2 field-map context."""
