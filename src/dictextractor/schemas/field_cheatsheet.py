"""Compact Pass-1 marker cheat sheet for Pass 2 direct MDF extraction."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class MarkerLine(BaseModel):
    """One MDF marker used in this dictionary."""

    marker: str = Field(description="Two-letter MDF code without backslash, e.g. lx, gn.")
    description: str = Field(description="One-line role description for Pass 2.")


class DictionaryMarkerCheatsheet(BaseModel):
    """
    Pass 1 output in cheat-sheet mode — markers + rules, no typography map.

    Rendered for Pass 2 in the same style as the static Chukchi experiment script.
    """

    dictionary_name: str = ""
    markers: List[MarkerLine] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    abbreviations: dict[str, str] = Field(default_factory=dict)

    def format_prompt_block(self) -> str:
        """Render as Pass 2 field map (static cheat-sheet style)."""
        title = self.dictionary_name or "this dictionary"
        lines = [
            f"MDF markers for {title} (use these exactly):",
            "",
        ]
        for m in self.markers:
            code = m.marker.lstrip("\\")
            lines.append(f"\\{code}   {m.description}")
        if self.rules:
            lines.append("")
            lines.append("Rules:")
            for rule in self.rules:
                lines.append(f"- {rule}")
        if self.abbreviations:
            lines.append("")
            lines.append("Abbreviations:")
            for abbr, meaning in sorted(self.abbreviations.items()):
                lines.append(f"- {abbr} → {meaning}")
        return "\n".join(lines)
