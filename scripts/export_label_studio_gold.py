"""Export submitted Label Studio post-edits back to stage-1 GOLD TSV files.

The script matches Label Studio projects created by ``label-studio/setup.py``
to subdirectories under a samples root, then writes each submitted page
annotation to the experiment-agnostic gold location:
``<samples-dir>/<language>/outputs/stage-1-gold/<page>/<page>_stage1_GOLD.tsv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

# Gold for these languages is maintained locally, not from Label Studio.
# Syncing would overwrite hand-edited stage-1-gold TSVs on disk.
EXCLUDED_LANGUAGES: frozenset[str] = frozenset({"Circassian-English-Turkish"})

AUTH_SCHEMES = {"auto", "Token", "Bearer", "PAT"}
TEXT_FIELD_TO_COLUMN: dict[str, str] = {
    "header_text": "header",
    "body_text": "single",
    "left_text": "left",
    "middle_text": "middle",
    "right_text": "right",
    "footer_text": "footer",
}
BODY_COLUMNS = {"single", "left", "middle", "right"}


class LabelStudioExportError(RuntimeError):
    """Raised when Label Studio data cannot be exported safely."""


class TextAreaValue(BaseModel):
    """TextArea payload from a Label Studio annotation result."""

    model_config = ConfigDict(extra="ignore")

    text: list[str] = Field(default_factory=list)


class AnnotationResult(BaseModel):
    """Single Label Studio annotation result item."""

    model_config = ConfigDict(extra="ignore")

    from_name: str
    type: str
    value: TextAreaValue


class Annotation(BaseModel):
    """Submitted Label Studio annotation."""

    model_config = ConfigDict(extra="ignore")

    id: int
    created_at: str | None = None
    updated_at: str | None = None
    result: list[AnnotationResult] = Field(default_factory=list)


class TaskData(BaseModel):
    """Task metadata and import-time OCR prefill from ``label-studio/setup.py``."""

    model_config = ConfigDict(extra="ignore")

    page_name: str
    language: str
    header_text: str = ""
    footer_text: str = ""
    body_text: str = ""
    left_text: str = ""
    middle_text: str = ""
    right_text: str = ""


class LabelStudioTask(BaseModel):
    """Label Studio task with submitted annotations."""

    model_config = ConfigDict(extra="ignore")

    id: int
    data: TaskData
    annotations: list[Annotation] = Field(default_factory=list)


class LabelStudioProject(BaseModel):
    """Label Studio project summary."""

    model_config = ConfigDict(extra="ignore")

    id: int
    title: str


class LabelStudioClient:
    """Small Label Studio API client for project/task export."""

    def __init__(
        self,
        base_url: str,
        token: str,
        auth_scheme: str = "auto",
        access_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.auth_scheme = auth_scheme
        self._active_auth_scheme = "Token" if auth_scheme == "auto" else auth_scheme
        self._tried_bearer = False
        self._tried_pat_refresh = False
        self.session = requests.Session()
        if access_token:
            self._set_bearer_token(access_token)
        elif auth_scheme == "PAT":
            self._refresh_pat_access_token()
        else:
            self._set_auth_header()

    def list_projects(self) -> list[LabelStudioProject]:
        """Return all visible projects.

        Raises:
            requests.HTTPError: If the Label Studio API request fails.
            ValidationError: If the response shape is unexpected.
        """
        return [
            LabelStudioProject.model_validate(item)
            for item in self._get_paginated("/api/projects/")
        ]

    def list_project_tasks(self, project_id: int) -> list[LabelStudioTask]:
        """Return all tasks for a project, including submitted annotations.

        Raises:
            requests.HTTPError: If the Label Studio API request fails.
            ValidationError: If the response shape is unexpected.
        """
        params = {"page_size": 100}
        items = self._get_paginated(f"/api/projects/{project_id}/tasks/", params)
        return [LabelStudioTask.model_validate(item) for item in items]

    def _get_paginated(
        self,
        path: str,
        params: dict[str, int] | None = None,
    ) -> list[dict[str, object]]:
        """Fetch a paginated Label Studio list endpoint."""
        url = self._url(path)
        page_params = dict(params or {})
        page_params.setdefault("page_size", 1000)
        results: list[dict[str, object]] = []

        while url:
            response = self.session.get(url, params=page_params)
            while self._retry_after_unauthorized(response):
                response = self.session.get(url, params=page_params)
            response.raise_for_status()
            payload = response.json()
            page_items, next_url = _extract_page(payload)
            results.extend(page_items)
            url = next_url
            page_params = {}

        return results

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _set_auth_header(self) -> None:
        self.session.headers.update({"Authorization": f"{self._active_auth_scheme} {self.token}"})

    def _set_bearer_token(self, token: str) -> None:
        self._active_auth_scheme = "Bearer"
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _refresh_pat_access_token(self) -> None:
        response = self.session.post(self._url("/api/token/refresh/"), json={"refresh": self.token})
        response.raise_for_status()
        access_token = response.json().get("access")
        if not isinstance(access_token, str) or not access_token:
            raise LabelStudioExportError("Label Studio token refresh response did not include an access token")
        self._tried_pat_refresh = True
        self._set_bearer_token(access_token)

    def _retry_after_unauthorized(self, response: requests.Response) -> bool:
        if response.status_code != 401:
            return False
        if self.auth_scheme == "PAT":
            logger.info("Bearer auth failed; refreshing PAT access token")
            self._refresh_pat_access_token()
            return True
        if self.auth_scheme != "auto":
            return False
        if not self._tried_bearer:
            logger.info("Legacy token auth failed; retrying with Bearer auth")
            self._tried_bearer = True
            self._active_auth_scheme = "Bearer"
            self._set_auth_header()
            return True
        logger.info("Bearer auth failed; refreshing PAT before retrying")
        self._refresh_pat_access_token()
        return True


def _extract_page(payload: object) -> tuple[list[dict[str, object]], str | None]:
    """Normalize Label Studio paginated and plain-list responses."""
    if isinstance(payload, list):
        return [_as_dict(item) for item in payload], None

    if isinstance(payload, dict):
        raw_items = payload.get("results", payload.get("tasks", []))
        if not isinstance(raw_items, list):
            raise LabelStudioExportError("Label Studio response has no list payload")
        next_url = payload.get("next")
        return [_as_dict(item) for item in raw_items], next_url if isinstance(next_url, str) else None

    raise LabelStudioExportError("Unexpected Label Studio response type")


def _as_dict(item: object) -> dict[str, object]:
    """Validate one raw API item is a JSON object."""
    if not isinstance(item, dict):
        raise LabelStudioExportError("Expected Label Studio list item to be an object")
    return item


def project_title_for_language(language: str) -> str:
    """Return the project title used by ``label-studio/setup.py``."""
    title = f"Post-Edit: {language}"
    return language[:50] if len(title) > 50 else title


def latest_annotation(task: LabelStudioTask) -> Annotation | None:
    """Select the newest submitted annotation for a task."""
    if not task.annotations:
        return None
    return max(task.annotations, key=lambda ann: ann.updated_at or ann.created_at or "")


def annotation_text_by_column(annotation: Annotation) -> dict[str, str]:
    """Extract submitted TextArea values keyed by stage-1 column id."""
    columns: dict[str, str] = {}
    for result in annotation.result:
        column = TEXT_FIELD_TO_COLUMN.get(result.from_name)
        if result.type != "textarea" or column is None:
            continue
        columns[column] = "\n".join(result.value.text)
    return columns


def task_data_text_by_column(task: LabelStudioTask) -> dict[str, str]:
    """Map Label Studio task import fields to stage-1 column ids."""
    data = task.data
    return {
        "header": data.header_text,
        "footer": data.footer_text,
        "single": data.body_text,
        "left": data.left_text,
        "middle": data.middle_text,
        "right": data.right_text,
    }


def columns_have_body_lines(columns: dict[str, str]) -> bool:
    """Return True if any column contributes at least one non-empty TSV line."""
    for column in ("header", "single", "left", "middle", "right", "footer"):
        if split_annotation_lines(columns.get(column, "")):
            return True
    return False


def resolve_export_columns(
    task: LabelStudioTask,
    *,
    include_prefill: bool,
) -> tuple[dict[str, str], str]:
    """
    Choose column text for GOLD export.

    Returns:
        (columns, source) where source is ``annotation``, ``prefill``, or ``skip``.
    """
    annotation = latest_annotation(task)
    if annotation is not None:
        columns = annotation_text_by_column(annotation)
        if columns_have_body_lines(columns):
            return columns, "annotation"
        if not include_prefill:
            return columns, "skip"

    if include_prefill:
        columns = task_data_text_by_column(task)
        if columns_have_body_lines(columns):
            return columns, "prefill"

    return {}, "skip"


def write_gold_tsv(samples_dir: Path, task: LabelStudioTask, columns: dict[str, str]) -> Path:
    """Write one task annotation to the experiment-agnostic gold location."""
    page_dir = (
        samples_dir
        / task.data.language
        / "outputs"
        / "stage-1-gold"
        / task.data.page_name
    )
    output_tsv = page_dir / f"{task.data.page_name}_stage1_GOLD.tsv"
    rows = build_tsv_rows(columns)
    page_dir.mkdir(parents=True, exist_ok=True)

    with output_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["column_id", "line_number", "text"])
        writer.writerows(rows)

    return output_tsv


def build_tsv_rows(columns: dict[str, str]) -> list[tuple[str, str, str]]:
    """Convert section text into stage-1 TSV rows.

    Complexity:
        Time O(n), space O(n), where n is the number of submitted text lines.
    """
    rows: list[tuple[str, str, str]] = []
    for column in ("header", "single", "left", "middle", "right", "footer"):
        lines = split_annotation_lines(columns.get(column, ""))
        for line_number, line in enumerate(lines, start=1):
            number = str(line_number) if column in BODY_COLUMNS else ""
            rows.append((column, number, line))
    return rows


def split_annotation_lines(text: str) -> list[str]:
    """Split a submitted text area into non-empty TSV text rows."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def export_language(
    client: LabelStudioClient,
    samples_dir: Path,
    project: LabelStudioProject,
    *,
    include_prefill: bool = False,
) -> int:
    """Export tasks from one Label Studio project (submitted and/or import prefill)."""
    tasks = client.list_project_tasks(project.id)
    written = 0

    for index, task in enumerate(tasks, start=1):
        columns, source = resolve_export_columns(task, include_prefill=include_prefill)
        if source == "skip":
            logger.info(
                "  [%d/%d] task %d (%s) skipped — no submitted text and no prefill",
                index,
                len(tasks),
                task.id,
                task.data.page_name,
            )
            continue
        output_path = write_gold_tsv(samples_dir, task, columns)
        logger.info(
            "  [%d/%d] wrote %s (%s)",
            index,
            len(tasks),
            output_path,
            source,
        )
        written += 1

    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-dir",
        "--output-dir",
        dest="samples_dir",
        type=Path,
        default=Path("assets/dictionaries/samples-2"),
        help="Samples root containing language subdirectories.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Only export these language subfolders.",
    )
    parser.add_argument(
        "--ls-url",
        default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"),
        help="Label Studio base URL.",
    )
    parser.add_argument(
        "--ls-token",
        default=os.getenv("LABEL_STUDIO_TOKEN"),
        help="Label Studio API token. Can also be set via LABEL_STUDIO_TOKEN.",
    )
    parser.add_argument(
        "--ls-access-token",
        default=os.getenv("LABEL_STUDIO_ACCESS_TOKEN"),
        help="Optional short-lived Label Studio access token for Bearer auth.",
    )
    parser.add_argument(
        "--ls-auth-scheme",
        choices=sorted(AUTH_SCHEMES),
        default=os.getenv("LABEL_STUDIO_AUTH_SCHEME", "auto"),
        help="Authorization scheme: auto, Token, Bearer, or PAT (refresh token -> access token).",
    )
    parser.add_argument(
        "--include-prefill",
        action="store_true",
        help=(
            "For tasks with no submitted annotation, or an empty submission "
            "(OCR accepted as-is), export import-time task data (header_text, "
            "left_text, etc.) from Label Studio setup."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Label Studio GOLD TSV export."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.ls_token:
        logger.error("No Label Studio token provided. Use --ls-token or LABEL_STUDIO_TOKEN.")
        return 1
    if not args.samples_dir.is_dir():
        logger.error("Samples directory not found: %s", args.samples_dir)
        return 1

    try:
        return run_export(
            args.samples_dir,
            args.ls_url,
            args.ls_token,
            args.ls_auth_scheme,
            args.ls_access_token,
            args.languages,
            args.include_prefill,
        )
    except (requests.HTTPError, ValidationError, LabelStudioExportError) as error:
        logger.error("Export failed: %s", error)
        return 1


def run_export(
    samples_dir: Path,
    ls_url: str,
    ls_token: str,
    ls_auth_scheme: str,
    ls_access_token: str | None,
    languages: list[str] | None,
    include_prefill: bool = False,
) -> int:
    """Export selected Label Studio projects into the samples directory."""
    client = LabelStudioClient(ls_url, ls_token, ls_auth_scheme, access_token=ls_access_token)
    projects = {project.title: project for project in client.list_projects()}
    entry_dirs = selected_entry_dirs(samples_dir, languages)
    total_written = 0

    logger.info("Connected to Label Studio (%d projects)", len(projects))
    logger.info("Exporting %d dictionary entries", len(entry_dirs))
    for index, entry_dir in enumerate(entry_dirs, start=1):
        title = project_title_for_language(entry_dir.name)
        project = projects.get(title)
        if project is None:
            logger.warning("[%d/%d] no project found for %s", index, len(entry_dirs), entry_dir.name)
            continue
        logger.info("[%d/%d] exporting %s (project id=%d)", index, len(entry_dirs), entry_dir.name, project.id)
        total_written += export_language(
            client, samples_dir, project, include_prefill=include_prefill
        )

    logger.info("Done. Wrote %d GOLD TSV files.", total_written)
    return 0


def selected_entry_dirs(samples_dir: Path, languages: list[str] | None) -> list[Path]:
    """Return sample subdirectories selected for export."""
    entries = sorted(path for path in samples_dir.iterdir() if path.is_dir())

    skipped = [entry.name for entry in entries if entry.name in EXCLUDED_LANGUAGES]
    if skipped:
        logger.info(
            "Skipping Label Studio sync (local gold only): %s",
            ", ".join(sorted(skipped)),
        )
    entries = [entry for entry in entries if entry.name not in EXCLUDED_LANGUAGES]

    if languages is None:
        return entries

    requested = set(languages)
    if requested & EXCLUDED_LANGUAGES:
        logger.warning(
            "--languages includes excluded entries (ignored): %s",
            ", ".join(sorted(requested & EXCLUDED_LANGUAGES)),
        )
    selected = [entry for entry in entries if entry.name in requested]
    missing = sorted(requested - EXCLUDED_LANGUAGES - {entry.name for entry in selected})
    if missing:
        logger.warning("Requested languages not found under samples dir: %s", ", ".join(missing))
    return selected


if __name__ == "__main__":
    sys.exit(main())
