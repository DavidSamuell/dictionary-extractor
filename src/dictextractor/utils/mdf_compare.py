"""
Compare produced MDF text against a gold reference (block-level metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def normalize_marker(line: str) -> str:
    """Map equivalent gloss/usage markers for comparison."""
    if line.startswith("\\ge "):
        return "\\gn " + line[4:]
    if line.startswith("\\de "):
        return "\\un " + line[4:]
    return line


def normalize_mdf_blocks(text: str) -> List[List[str]]:
    """Split MDF into blocks of normalized marker lines."""
    blocks: List[List[str]] = []
    for block in text.strip().split("\n\n"):
        lines = [normalize_marker(ln.strip()) for ln in block.splitlines() if ln.strip()]
        if lines:
            blocks.append(lines)
    return blocks


def _headword_from_block(block: List[str]) -> str:
    for line in block:
        if line.startswith("\\lx "):
            return line[4:].strip()
    return ""


@dataclass
class MdfCompareReport:
    """Block-level MDF comparison against gold."""

    gold_records: int = 0
    produced_records: int = 0
    exact_block_matches: int = 0
    missing_blocks: int = 0
    extra_blocks: int = 0
    gold_gloss_lines: int = 0
    produced_gloss_lines: int = 0
    headword_recall: float = 0.0
    headword_precision: float = 0.0
    missing_block_samples: List[List[str]] = field(default_factory=list)
    extra_block_samples: List[List[str]] = field(default_factory=list)

    @property
    def gloss_line_ratio(self) -> float:
        if not self.gold_gloss_lines:
            return 0.0
        return self.produced_gloss_lines / self.gold_gloss_lines

    @property
    def ok(self) -> bool:
        return self.missing_blocks == 0 and self.extra_blocks == 0


def compare_mdf_to_gold(produced: str, gold: str) -> MdfCompareReport:
    """Compare produced MDF to gold with marker normalization."""
    prod_blocks = normalize_mdf_blocks(produced)
    gold_blocks = normalize_mdf_blocks(gold)

    matched = [b for b in gold_blocks if b in prod_blocks]
    missing = [b for b in gold_blocks if b not in prod_blocks]
    extra = [b for b in prod_blocks if b not in gold_blocks]

    gold_gn = sum(1 for b in gold_blocks for ln in b if ln.startswith("\\gn "))
    prod_gn = sum(1 for b in prod_blocks for ln in b if ln.startswith("\\gn "))
    gold_hw = {_headword_from_block(b) for b in gold_blocks}
    prod_hw = {_headword_from_block(b) for b in prod_blocks}

    return MdfCompareReport(
        gold_records=len(gold_blocks),
        produced_records=len(prod_blocks),
        exact_block_matches=len(matched),
        missing_blocks=len(missing),
        extra_blocks=len(extra),
        gold_gloss_lines=gold_gn,
        produced_gloss_lines=prod_gn,
        headword_recall=len(gold_hw & prod_hw) / len(gold_hw) if gold_hw else 0.0,
        headword_precision=len(gold_hw & prod_hw) / len(prod_hw) if prod_hw else 0.0,
        missing_block_samples=missing[:5],
        extra_block_samples=extra[:5],
    )
