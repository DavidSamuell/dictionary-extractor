"""Per-dictionary language roles for Stage 2 and MDF export."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

LayoutType = Literal["bilingual", "inline_trilingual", "column_trilingual"]


class SourceLanguageConfig(BaseModel):
    """Vernacular / headword language."""

    model_config = ConfigDict(extra="ignore")

    language: str = Field(description="Human-readable source language name.")
    code: str = Field(description="Short stable code, e.g. na, chukchi.")
    column_id: Optional[str] = Field(
        default=None,
        description="Stage-1 column_id for headwords when layout is column_trilingual.",
    )


class TargetLanguageConfig(BaseModel):
    """One gloss/translation language."""

    model_config = ConfigDict(extra="ignore")

    language: str = Field(description="Human-readable target language name.")
    code: str = Field(
        description="Stable language code used at export, e.g. en, zh, tr."
    )
    column_id: Optional[str] = Field(
        default=None,
        description="Stage-1 column_id for this gloss when layout is column_trilingual.",
    )


class DictionaryLanguagesConfig(BaseModel):
    """Loaded from dictionary_languages.yaml in each sample folder."""

    model_config = ConfigDict(extra="ignore")

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
        """Stable language codes for MDF export."""
        return [t.code for t in self.targets]

    def format_prompt_block(self) -> str:
        """Inject into Stage 2 user prompt (schema fields only — no MDF markers)."""
        primary = self.targets[0]
        lines = [
            "<dictionary_languages>",
            f"Layout: {self.layout}",
            f"Source language ({self.source.code}): {self.source.language} → headword field",
        ]
        if self.source.column_id:
            lines.append(
                f"  Read headwords from column_id={self.source.column_id!r} in the transcription."
            )
        lines.append(
            f"Primary target ({primary.language}) → gloss field — short translation text."
        )
        if primary.column_id:
            lines.append(f"  Read from column_id={primary.column_id!r} when column layout applies.")
        if len(self.targets) > 1:
            secondary = self.targets[1]
            lines.append(
                f"Second target ({secondary.language}) → gloss_secondary field."
            )
            if secondary.column_id:
                lines.append(
                    f"  Read from column_id={secondary.column_id!r} when column layout applies."
                )
        if self.layout == "inline_trilingual":
            lines.append(
                "  Split targets by typography and intro order within each entry block; "
                "do not merge languages into one string."
            )
        elif self.layout == "column_trilingual":
            lines.append(
                "  Align each target gloss with its column line; headword from the "
                "source column only."
            )
        else:
            lines.append(
                "  Bilingual: non-bold translation text beside each bold headword "
                "belongs in gloss. Never leave gloss empty when translation text appears "
                "in the transcription. Use usage_note only for italic parenthetical "
                "usage or domain notes."
            )
        lines.append(
            "usage_note: parenthetical or italic usage/domain expansions only — "
            "not numbered inline senses, not the primary translation."
        )
        lines.append(
            "Example (bilingual): <b>lemma</b> translate (usage note) → "
            "gloss='translate', usage_note='usage note'."
        )
        lines.append("</dictionary_languages>")
        return "\n".join(lines)
