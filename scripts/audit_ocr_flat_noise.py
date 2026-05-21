#!/usr/bin/env python3
"""Scan OCR flat files for remaining noise patterns (stdout summary)."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

SAMPLES = _REPO / "assets/dictionaries/samples"
OCR_EXPS = ("MinerU2.5-Pro", "PaddleOCR-VL-1.5", "GLM-OCR")

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("latex_cmd", re.compile(r"\\[a-zA-Z]{2,}")),
    ("html_entity", re.compile(r"&#\w+;|&[a-z]+;")),
    ("xi_garbage", re.compile(r"x<i>\{\d+\}", re.I)),
    ("swastika", re.compile(r"卍{2,}")),
    ("stray_lt", re.compile(r"<(?!(?:/?)(?:b|i)\b|/?N>)", re.I)),
    ("digit_run", re.compile(r"^\d[\d\s]{79,}$")),
    ("greek_spam", re.compile(r"[\u03a5\u0393]{20,}")),
    ("char_repeat", re.compile(r"(.)\1{11,}")),
    ("modifier", re.compile(r"[\u2070-\u209f]{5,}")),
    ("checkbox", re.compile(r"☐")),
    ("replacement", re.compile(r"\ufffd")),
    ("math_delim", re.compile(r"\\\(|\\\)|\\\[|\\\]")),
]


def main() -> int:
    by_pat: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    for exp in OCR_EXPS:
        for flat in sorted(SAMPLES.glob(f"*/outputs/stage-1/{exp}/*/*_stage1_flat.txt")):
            for i, line in enumerate(
                flat.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if not line.strip():
                    continue
                for name, pat in PATTERNS:
                    if pat.search(line):
                        by_pat[name] += 1
                        examples.setdefault(name, [])
                        if len(examples[name]) < 2:
                            rel = flat.relative_to(SAMPLES)
                            examples[name].append(f"{rel}:{i} {line[:80]!r}")

    print("Remaining noise (line hits across 126 OCR flats):")
    for name, count in by_pat.most_common():
        print(f"  {name}: {count}")
        for ex in examples.get(name, []):
            print(f"    {ex}")
    if not by_pat:
        print("  (none)")
    return 0 if not by_pat else 1


if __name__ == "__main__":
    raise SystemExit(main())
