"""Load and apply MDF marker substitution groups."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Mapping

import yaml

DEFAULT_MARKER_SUB_LIST_PATH = (
    Path(__file__).resolve().parents[4] / "assets" / "evaluation" / "mdf_marker_sub_list.yaml"
)

_BUILTIN_SUB_LIST: dict[str, frozenset[str]] = {
    "gn": frozenset({"gn", "dn"}),
    "dn": frozenset({"gn", "dn"}),
    "de": frozenset({"de", "ge"}),
    "ge": frozenset({"de", "ge"}),
}


def _build_lookup(groups: list[list[str]]) -> dict[str, frozenset[str]]:
    lookup: dict[str, frozenset[str]] = {}
    for group in groups:
        frozen = frozenset(group)
        for marker in group:
            lookup[marker] = frozen
    return lookup


@lru_cache(maxsize=4)
def load_marker_sub_list(path: str | None = None) -> Mapping[str, FrozenSet[str]]:
    """Load marker substitution lookup from YAML or fall back to built-ins."""
    yaml_path = Path(path) if path else DEFAULT_MARKER_SUB_LIST_PATH

    if not yaml_path.is_file():
        return _BUILTIN_SUB_LIST

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    groups = data.get("equivalence_groups", [])
    if not groups:
        return _BUILTIN_SUB_LIST
    return _build_lookup(groups)


def markers_equivalent(gold: str, pred: str, *, sub_list_path: str | None = None) -> bool:
    """Return whether two marker codes belong to the same substitution group."""
    if gold == pred:
        return True
    lookup = load_marker_sub_list(sub_list_path)
    gold_group = lookup.get(gold, frozenset({gold}))
    pred_group = lookup.get(pred, frozenset({pred}))
    return bool(gold_group & pred_group)
