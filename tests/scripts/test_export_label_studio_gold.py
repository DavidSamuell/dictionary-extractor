"""Tests for Label Studio GOLD export helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))

from export_label_studio_gold import (  # noqa: E402
    Annotation,
    AnnotationResult,
    LabelStudioTask,
    TaskData,
    TextAreaValue,
    columns_have_body_lines,
    resolve_export_columns,
    task_data_text_by_column,
)


def _task(
    *,
    page_name: str = "page_3",
    prefill_left: str = "line one",
    annotations: list[Annotation] | None = None,
) -> LabelStudioTask:
    return LabelStudioTask(
        id=1,
        data=TaskData(
            page_name=page_name,
            language="Shilluk-English",
            left_text=prefill_left,
        ),
        annotations=annotations or [],
    )


def test_prefill_when_no_annotation() -> None:
    columns, source = resolve_export_columns(_task(), include_prefill=True)
    assert source == "prefill"
    assert columns["left"] == "line one"


def test_prefill_when_empty_submission() -> None:
    empty = Annotation(
        id=1,
        result=[
            AnnotationResult(
                from_name="left_text",
                type="textarea",
                value=TextAreaValue(text=[""]),
            )
        ],
    )
    columns, source = resolve_export_columns(
        _task(annotations=[empty]), include_prefill=True
    )
    assert source == "prefill"
    assert columns["left"] == "line one"


def test_annotation_preferred_over_prefill() -> None:
    ann = Annotation(
        id=1,
        result=[
            AnnotationResult(
                from_name="left_text",
                type="textarea",
                value=TextAreaValue(text=["edited line"]),
            )
        ],
    )
    columns, source = resolve_export_columns(
        _task(annotations=[ann]), include_prefill=True
    )
    assert source == "annotation"
    assert columns["left"] == "edited line"


def test_skip_without_prefill_flag() -> None:
    columns, source = resolve_export_columns(_task(), include_prefill=False)
    assert source == "skip"
    assert columns == {}


def test_task_data_maps_body_text_to_single() -> None:
    task = LabelStudioTask(
        id=2,
        data=TaskData(
            page_name="page_1",
            language="X",
            body_text="only body",
        ),
    )
    cols = task_data_text_by_column(task)
    assert cols["single"] == "only body"
    assert columns_have_body_lines(cols)
