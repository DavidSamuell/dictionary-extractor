"""Save Pass 2 direct MDF outputs and optional gold comparison."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from dictextractor.utils.mdf_compare import MdfCompareReport, compare_mdf_to_gold

logger = logging.getLogger(__name__)


def save_direct_mdf_outputs(
    *,
    mdf_text: str,
    output_base: Path,
    gold_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Write ``.mdf.txt`` and optional ``_gold_compare.json`` validation report.

    Args:
        mdf_text: Produced MDF body.
        output_base: Path whose stem is the page prefix (e.g. ``page_3/page_3.tsv``).
        gold_path: Optional gold MDF file for comparison.

    Returns:
        Dict with output paths and comparison summary.
    """
    mdf_path = output_base.with_suffix(".mdf.txt")
    mdf_path.write_text(mdf_text, encoding="utf-8")
    print(f"Saved MDF to {mdf_path}")

    result: Dict[str, Any] = {"mdf": str(mdf_path)}
    if gold_path and gold_path.is_file():
        report = compare_mdf_to_gold(mdf_text, gold_path.read_text(encoding="utf-8"))
        compare_path = output_base.with_name(output_base.stem + "_gold_compare.json")
        compare_path.write_text(
            json.dumps(
                {
                    "gold_path": str(gold_path),
                    "ok": report.ok,
                    "gold_records": report.gold_records,
                    "produced_records": report.produced_records,
                    "exact_block_matches": report.exact_block_matches,
                    "missing_blocks": report.missing_blocks,
                    "extra_blocks": report.extra_blocks,
                    "gold_gloss_lines": report.gold_gloss_lines,
                    "produced_gloss_lines": report.produced_gloss_lines,
                    "gloss_line_ratio": report.gloss_line_ratio,
                    "headword_recall": report.headword_recall,
                    "headword_precision": report.headword_precision,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _log_compare_report(report, page_label=output_base.stem)
        print(
            f"Gold comparison: {report.exact_block_matches}/{report.gold_records} "
            f"exact blocks, {report.produced_gloss_lines}/{report.gold_gloss_lines} "
            f"gloss lines, headword recall {report.headword_recall:.1%}"
        )
        print(f"Saved gold comparison to {compare_path}")
        result["gold_compare"] = str(compare_path)
        result["gold_compare_ok"] = report.ok
    return result


def _log_compare_report(report: MdfCompareReport, *, page_label: str = "") -> None:
    prefix = f"{page_label}: " if page_label else ""
    logger.info(
        "%sMDF vs gold — exact %d/%d, gloss lines %d/%d, headword recall %.1f%%",
        prefix,
        report.exact_block_matches,
        report.gold_records,
        report.produced_gloss_lines,
        report.gold_gloss_lines,
        report.headword_recall * 100,
    )
