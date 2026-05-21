"""Layout blocks for OCR adapter v1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutBlock:
    """Normalized layout element for transcript serialization."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    category: str

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def y_min(self) -> float:
        return self.y0

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1e-6)
