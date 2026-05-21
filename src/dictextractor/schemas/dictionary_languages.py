"""Per-dictionary language roles for Stage 2 and MDF export."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

LayoutType = Literal["bilingual", "inline_trilingual", "column_trilingual"]

# Default SIL Toolbox gloss markers for up to three target languages.
_DEFAULT_MDF_MARKERS = ("ge", "gn", "gf")


class SourceLanguageConfig(BaseModel):
    """Vernacular / headword language (\\lx)."""

    language: str = Field(description="Human-readable source language name.")
    code: str = Field(description="Short stable code, e.g. na, chukchi.")
    mdf_lexeme: str = Field(default="lx", description="MDF marker for the headword.")
    column_id: Optional[str] = Field(
        default=None,
        description="Stage-1 column_id for headwords when layout is column_trilingual.",
    )


class TargetLanguageConfig(BaseModel):
    """One gloss/translation language."""

    language: str = Field(description="Human-readable target language name.")
    code: str = Field(
        description="Key for DictionaryEntry.target_glosses, e.g. en, zh, tr."
    )
    mdf_marker: str = Field(
        description="MDF gloss marker for this target, e.g. ge, gn, gf."
    )
    column_id: Optional[str] = Field(
        default=None,
        description="Stage-1 column_id for this gloss when layout is column_trilingual.",
    )


class DictionaryLanguagesConfig(BaseModel):
    """Loaded from dictionary_languages.yaml in each sample folder."""

    layout: LayoutType = Field(
        description=(
            "bilingual: one target mixed or single column; "
            "inline_trilingual: multiple targets in one entry block; "
            "column_trilingual: each target in its own column."
        )
    )
    source: SourceLanguageConfig
    targets: List[TargetLanguageConfig] = Field(min_length=1)
    writing_system: str = Field(
        default="",
        description="From dictionary_metadata.csv when matched.",
    )
    metadata_archive: str = Field(
        default="",
        description="First CSV column (archive id) when matched.",
    )

    def target_codes(self) -> List[str]:
        """Stable keys for target_glosses."""
        return [t.code for t in self.targets]

    def format_prompt_block(self) -> str:
        """Inject into Stage 2 user prompt."""
        lines = [
            "<dictionary_languages>",
            f"Layout: {self.layout}",
            f"Source ({self.source.code}): {self.source.language} → headword (\\{self.source.mdf_lexeme})",
        ]
        if self.source.column_id:
            lines.append(
                f"  Read headwords from column_id={self.source.column_id!r} in the transcription."
            )
        lines.append("Target glosses → target_glosses map (one key per language):")
        for t in self.targets:
            col = f", column_id={t.column_id!r}" if t.column_id else ""
            lines.append(
                f"  - target_glosses[{t.code!r}]: {t.language} (MDF \\{t.mdf_marker}){col}"
            )
        if self.layout == "inline_trilingual":
            lines.append(
                "  Split English vs other targets from typography and intro order within "
                "each entry block; do not merge into one string."
            )
        elif self.layout == "column_trilingual":
            lines.append(
                "  Align glosses with the matching column lines for each entry; "
                "headword from the source column only."
            )
        else:
            lines.append(
                "  Put the translation-language gloss in the single target key; "
                "leave gloss empty (use target_glosses only)."
            )
        lines.append(
            "Always leave legacy field gloss empty. Use definition for longer \\de text."
        )
        lines.append("</dictionary_languages>")
        return "\n".join(lines)
