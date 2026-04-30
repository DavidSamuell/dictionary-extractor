"""Set up Label Studio projects for post-editing OCR transcriptions.

Creates one Label Studio project per dictionary language pair, renders
snippet PDFs to PNGs, uploads page images, and imports stage-1 TSV
transcriptions as pre-filled tasks.

Usage:
    conda activate label-studio
    python scripts/label_studio_setup.py \
        --samples-dir assets/dictionaries/samples-2 \
        --ls-url http://localhost:8080 \
        --ls-token <your-legacy-token>

Environment:
    LABEL_STUDIO_URL  — Label Studio base URL (default: http://localhost:8080)
    LABEL_STUDIO_TOKEN — API token (legacy or PAT)
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import shutil
import sys
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LABEL_CONFIG = """\
<View>
  <Style>
    .container { display: flex; gap: 16px; height: 90vh; }
    .left-panel { flex: 1; overflow: auto; border: 1px solid #ddd; border-radius: 8px; padding: 8px; background: #fafafa; }
    .right-panel { flex: 1; overflow: auto; }
    .col-header { font-weight: 700; font-size: 14px; margin: 12px 0 4px; color: #333; border-bottom: 2px solid #4a86e8; padding-bottom: 4px; }
    .instructions { font-size: 13px; color: #666; margin-bottom: 12px; padding: 8px; background: #fff8e1; border-radius: 4px; border-left: 3px solid #ffc107; }
  </Style>

  <View className="container">
    <!-- Left: original page image -->
    <View className="left-panel">
      <Header value="Original Dictionary Page" size="4"/>
      <Image name="page_image" value="$image_url" zoomControl="true" rotateControl="true"/>
    </View>

    <!-- Right: editable transcription -->
    <View className="right-panel">
      <Header value="OCR Transcription — Post-Edit" size="4"/>
      <View className="instructions">
        <HyperText name="help" value="Compare with the original page on the left. Fix any character errors, missing text, or formatting issues in the transcription boxes below."/>
      </View>

      <View className="col-header">
        <Header value="Left Column" size="5"/>
      </View>
      <TextArea name="left_text" toName="page_image"
                value="$left_text" rows="20" editable="true"
                maxSubmissions="1" showSubmitButton="false"/>

      <View className="col-header">
        <Header value="Right Column" size="5"/>
      </View>
      <TextArea name="right_text" toName="page_image"
                value="$right_text" rows="20" editable="true"
                maxSubmissions="1" showSubmitButton="false"/>
    </View>
  </View>
</View>
"""


def _render_pdf_to_png(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    """Render each page of a PDF to PNG. Returns list of output paths.

    Complexity: O(n) where n is the number of pages in the PDF.
    """
    try:
        import pymupdf
    except ImportError:
        logger.error("pymupdf not installed — run: pip install pymupdf")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    results: list[Path] = []
    try:
        for page_idx in range(doc.page_count):
            suffix = "" if doc.page_count == 1 else f"_p{page_idx + 1}"
            out_path = output_dir / f"{pdf_path.stem}{suffix}.png"
            if out_path.exists():
                results.append(out_path)
                continue
            pix = doc.load_page(page_idx).get_pixmap(dpi=dpi)
            pix.save(str(out_path))
            results.append(out_path)
    finally:
        doc.close()
    return results


def _read_stage1_tsv(tsv_path: Path) -> dict[str, str]:
    """Read a stage-1 TSV and return {left_text, right_text} as joined strings."""
    left_lines: list[str] = []
    right_lines: list[str] = []

    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            col = row.get("column_id", "").strip()
            text = row.get("text", "").strip()
            if col == "left":
                left_lines.append(text)
            elif col == "right":
                right_lines.append(text)

    return {
        "left_text": "\n".join(left_lines),
        "right_text": "\n".join(right_lines),
    }


class LabelStudioClient:
    """Thin wrapper around Label Studio HTTP API."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {token}"})

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def list_projects(self) -> list[dict]:
        resp = self.session.get(self._url("/api/projects/"), params={"page_size": 1000})
        resp.raise_for_status()
        return resp.json().get("results", [])

    def create_project(self, title: str, label_config: str, description: str = "") -> dict | None:
        resp = self.session.post(self._url("/api/projects/"), json={
            "title": title,
            "description": description,
            "label_config": label_config,
            "is_published": True,
            "show_skip_button": True,
            "enable_empty_annotation": True,
        })
        if not resp.ok:
            logger.error("Failed to create project '%s': %s", title, resp.text[:300])
            return None
        return resp.json()

    def import_tasks(self, project_id: int, tasks: list[dict]) -> dict:
        resp = self.session.post(
            self._url(f"/api/projects/{project_id}/import"),
            json=tasks,
        )
        resp.raise_for_status()
        return resp.json()

    def create_local_storage(self, project_id: int, document_root: str) -> dict | None:
        """Create a local file storage connection for the project."""
        resp = self.session.post(
            self._url("/api/storages/localfiles"),
            json={
                "path": document_root,
                "project": project_id,
                "use_blob_urls": True,
                "title": "Page Renders",
            },
        )
        if resp.ok:
            return resp.json()
        logger.warning("Failed to create local storage: %s", resp.text[:200])
        return None

    def upload_file(self, project_id: int, file_path: Path) -> str:
        """Upload a file and return its serving URL."""
        with file_path.open("rb") as f:
            resp = self.session.post(
                self._url(f"/api/projects/{project_id}/import"),
                files={"file": (file_path.name, f, "image/png")},
            )
        resp.raise_for_status()
        return resp.json()


def setup_project_for_entry(
    client: LabelStudioClient,
    entry_dir: Path,
    render_dir: Path,
    *,
    overwrite: bool = False,
) -> int | None:
    """Create a Label Studio project for one dictionary entry.

    Returns the project ID on success, None if skipped.
    """
    entry_name = entry_dir.name
    stage1_dir = entry_dir / "outputs" / "stage-1"
    snippets_dir = entry_dir / "snippets"

    if not stage1_dir.is_dir():
        logger.warning("Skipping %s: no outputs/stage-1/ folder", entry_name)
        return None
    if not snippets_dir.is_dir():
        logger.warning("Skipping %s: no snippets/ folder", entry_name)
        return None

    existing = {p["title"]: p["id"] for p in client.list_projects()}
    project_title = f"Post-Edit: {entry_name}"
    if len(project_title) > 50:
        project_title = entry_name[:50]

    if project_title in existing and not overwrite:
        logger.info("Project '%s' already exists (id=%d), skipping", project_title, existing[project_title])
        return existing[project_title]

    project = client.create_project(
        title=project_title,
        label_config=LABEL_CONFIG,
        description=f"Post-editing OCR transcription for {entry_name} dictionary pages.",
    )
    if project is None:
        return None
    project_id = project["id"]
    logger.info("Created project '%s' (id=%d)", project_title, project_id)

    entry_render_dir = render_dir / entry_name
    entry_render_dir.mkdir(parents=True, exist_ok=True)

    # Connect local file storage so /data/local-files/ URLs resolve
    storage = client.create_local_storage(
        project_id,
        str(entry_render_dir.resolve()),
    )
    if storage:
        logger.info("  Local storage connected (id=%s)", storage.get("id"))
    tasks: list[dict] = []

    page_dirs = sorted(d for d in stage1_dir.iterdir() if d.is_dir())
    for page_dir in page_dirs:
        tsv_files = list(page_dir.glob("*_stage1.tsv"))
        if not tsv_files:
            continue

        tsv_path = tsv_files[0]
        page_stem = tsv_path.stem.replace("_stage1", "")

        pdf_path = snippets_dir / f"{page_stem}.pdf"
        if not pdf_path.exists():
            img_candidates = list(snippets_dir.glob(f"{page_stem}.*"))
            if img_candidates:
                pdf_path = img_candidates[0]
            else:
                logger.warning("  No snippet found for %s", page_stem)
                continue

        if pdf_path.suffix.lower() == ".pdf":
            rendered = _render_pdf_to_png(pdf_path, entry_render_dir)
            if not rendered:
                logger.warning("  Failed to render %s", pdf_path.name)
                continue
            image_path = rendered[0]
        else:
            dest = entry_render_dir / pdf_path.name
            if not dest.exists():
                shutil.copy2(pdf_path, dest)
            image_path = dest

        texts = _read_stage1_tsv(tsv_path)

        # Local file URL: path relative to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
        task = {
            "data": {
                "image_url": f"/data/local-files/?d={entry_name}/{image_path.name}",
                "left_text": texts["left_text"],
                "right_text": texts["right_text"],
                "page_name": page_stem,
                "language": entry_name,
            },
        }
        tasks.append(task)
        logger.info("  Prepared task for %s", page_stem)

    if not tasks:
        logger.warning("No tasks to import for %s", entry_name)
        return project_id

    result = client.import_tasks(project_id, tasks)
    task_count = result.get("task_count", len(tasks))
    logger.info("Imported %d tasks into project '%s'", task_count, project_title)
    return project_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up Label Studio projects for OCR post-editing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("assets/dictionaries/samples-2"),
        help="Root directory with language subfolders",
    )
    parser.add_argument(
        "--languages", nargs="+", default=None,
        help="Only process these language subfolders",
    )
    parser.add_argument(
        "--ls-url",
        default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"),
        help="Label Studio base URL",
    )
    parser.add_argument(
        "--ls-token",
        default=os.getenv("LABEL_STUDIO_TOKEN"),
        help="Label Studio API token (legacy or PAT)",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=Path(".label-studio-renders"),
        help="Directory for rendered PNG page images",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing projects")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.ls_token:
        logger.error("No Label Studio token provided. Use --ls-token or set LABEL_STUDIO_TOKEN.")
        return 1

    if not args.samples_dir.is_dir():
        logger.error("Samples directory not found: %s", args.samples_dir)
        return 1

    client = LabelStudioClient(args.ls_url, args.ls_token)

    try:
        projects = client.list_projects()
        logger.info("Connected to Label Studio (%d existing projects)", len(projects))
    except requests.HTTPError as e:
        logger.error("Failed to connect to Label Studio: %s", e)
        return 1

    entries = sorted(p for p in args.samples_dir.iterdir() if p.is_dir())
    if args.languages:
        requested = set(args.languages)
        entries = [e for e in entries if e.name in requested]

    logger.info("Processing %d dictionary entries", len(entries))

    created = 0
    for entry_dir in entries:
        project_id = setup_project_for_entry(
            client, entry_dir, args.render_dir, overwrite=args.overwrite,
        )
        if project_id is not None:
            created += 1

    logger.info("Done. %d projects set up.", created)
    return 0


if __name__ == "__main__":
    sys.exit(main())
