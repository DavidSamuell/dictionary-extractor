"""Thin client for the Mathpix Convert PDF API.

Submits snippet PDFs (or raster images wrapped as single-page PDFs) for
conversion, polls until processing is complete, and downloads markdown
(and optional sidecars). See https://docs.mathpix.com/#process-a-pdf
for the authoritative API contract.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from dictextractor.ocr.vlm.page_inputs import IMAGE_SUFFIXES, PDF_SUFFIX

logger = logging.getLogger(__name__)

MATHPIX_PDF_ENDPOINT = "https://api.mathpix.com/v3/pdf"
SNIPPET_SUFFIXES = IMAGE_SUFFIXES | {PDF_SUFFIX}


class MathpixConvertError(RuntimeError):
    """Raised when the Mathpix Convert API reports an error or a timeout."""


def prepare_mathpix_upload(snippet: Path, cache_dir: Path) -> Path:
    """Return a PDF path suitable for ``POST v3/pdf`` (wrap PNG/images if needed)."""
    suffix = snippet.suffix.lower()
    if suffix == PDF_SUFFIX:
        return snippet
    if suffix not in IMAGE_SUFFIXES:
        raise MathpixConvertError(
            f"Unsupported Mathpix snippet type {snippet.suffix}: {snippet}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    upload_pdf = cache_dir / f"{snippet.stem}.upload.pdf"
    src_mtime = snippet.stat().st_mtime
    if upload_pdf.is_file() and upload_pdf.stat().st_mtime >= src_mtime:
        logger.debug("Reusing cached upload PDF for %s -> %s", snippet.name, upload_pdf)
        return upload_pdf

    import fitz

    img = fitz.open(snippet)
    try:
        if img.page_count == 0:
            raise MathpixConvertError(f"Image snippet has no pages: {snippet}")
        rect = img[0].rect
        pdf = fitz.open()
        try:
            page = pdf.new_page(width=rect.width, height=rect.height)
            page.insert_image(page.rect, filename=str(snippet))
            pdf.save(upload_pdf)
        finally:
            pdf.close()
    finally:
        img.close()

    logger.info("Wrapped image snippet %s -> %s", snippet.name, upload_pdf.name)
    return upload_pdf


@dataclass(frozen=True)
class MathpixCredentials:
    app_id: str
    app_key: str

    @classmethod
    def from_env(cls) -> "MathpixCredentials":
        app_id = os.environ.get("MATHPIX_APP_ID")
        app_key = os.environ.get("MATHPIX_APP_KEY")
        if not app_id or not app_key:
            raise MathpixConvertError(
                "MATHPIX_APP_ID and MATHPIX_APP_KEY must be set in the environment."
            )
        return cls(app_id=app_id, app_key=app_key)

    @property
    def headers(self) -> dict[str, str]:
        return {"app_id": self.app_id, "app_key": self.app_key}


class MathpixConvertClient:
    """Submit-poll-download wrapper around the Mathpix Convert PDF API."""

    def __init__(
        self,
        credentials: MathpixCredentials | None = None,
        *,
        poll_interval_seconds: float = 3.0,
        max_wait_seconds: float = 600.0,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        self._credentials = credentials or MathpixCredentials.from_env()
        self._poll_interval = poll_interval_seconds
        self._max_wait = max_wait_seconds
        self._request_timeout = request_timeout_seconds

    def convert_pdf_to_docx(self, pdf_path: Path, output_path: Path) -> Path:
        """Convert a single PDF to DOCX via Mathpix and write to ``output_path``."""
        return self.convert_pdf_page(pdf_path, docx_path=output_path)

    def convert_pdf_to_md(self, pdf_path: Path, output_path: Path) -> Path:
        """Convert a single PDF to markdown via Mathpix and write to ``output_path``."""
        return self.convert_pdf_page(pdf_path, md_path=output_path)

    def convert_pdf_page(
        self,
        snippet_path: Path,
        *,
        md_path: Path | None = None,
        docx_path: Path | None = None,
        lines_json_path: Path | None = None,
        upload_cache_dir: Path | None = None,
    ) -> Path:
        """Convert a snippet PDF/image and download markdown/DOCX plus optional sidecar."""
        formats: dict[str, bool] = {}
        if md_path is not None:
            formats["md"] = True
        if docx_path is not None:
            formats["docx"] = True
        if not formats:
            formats = {"md": True}

        cache_dir = upload_cache_dir or snippet_path.parent
        upload_path = prepare_mathpix_upload(snippet_path, cache_dir)
        pdf_id = self._submit(upload_path, conversion_formats=formats)
        logger.info("Submitted %s -> pdf_id=%s", snippet_path.name, pdf_id)
        self._wait_until_complete(pdf_id)

        primary = md_path or docx_path
        if md_path is not None:
            self._download_md(pdf_id, md_path)
        if docx_path is not None:
            self._download_docx(pdf_id, docx_path)
        if lines_json_path is not None:
            self._download_lines_json(pdf_id, lines_json_path)
        if primary is None:
            raise MathpixConvertError("convert_pdf_page requires md_path or docx_path")
        return primary

    def _submit(
        self,
        upload_path: Path,
        *,
        conversion_formats: dict[str, bool] | None = None,
    ) -> str:
        options = {"conversion_formats": conversion_formats or {"md": True}}
        mime_type = mimetypes.guess_type(upload_path.name)[0] or "application/pdf"
        with upload_path.open("rb") as fh:
            files = {"file": (upload_path.name, fh, mime_type)}
            data = {"options_json": json.dumps(options)}
            response = requests.post(
                MATHPIX_PDF_ENDPOINT,
                headers=self._credentials.headers,
                files=files,
                data=data,
                timeout=self._request_timeout,
            )
        if response.status_code >= 400:
            raise MathpixConvertError(
                f"Mathpix submit failed ({response.status_code}): {response.text}"
            )
        payload = response.json()
        pdf_id = payload.get("pdf_id")
        if not pdf_id:
            raise MathpixConvertError(f"Mathpix response missing pdf_id: {payload}")
        return pdf_id

    def _wait_until_complete(self, pdf_id: str) -> None:
        status_url = f"{MATHPIX_PDF_ENDPOINT}/{pdf_id}"
        deadline = time.monotonic() + self._max_wait
        while True:
            response = requests.get(
                status_url,
                headers=self._credentials.headers,
                timeout=self._request_timeout,
            )
            if response.status_code >= 400:
                raise MathpixConvertError(
                    f"Mathpix status check failed ({response.status_code}): {response.text}"
                )
            payload = response.json()
            status = payload.get("status")
            logger.debug("pdf_id=%s status=%s", pdf_id, status)

            if status == "completed":
                return
            if status == "error":
                raise MathpixConvertError(f"Mathpix reported error for {pdf_id}: {payload}")

            if time.monotonic() > deadline:
                raise MathpixConvertError(
                    f"Timed out waiting for Mathpix conversion of {pdf_id} (last status: {status})"
                )
            time.sleep(self._poll_interval)

    def _download_docx(self, pdf_id: str, output_path: Path) -> None:
        docx_url = f"{MATHPIX_PDF_ENDPOINT}/{pdf_id}.docx"
        response = requests.get(
            docx_url,
            headers=self._credentials.headers,
            timeout=self._request_timeout,
        )
        if response.status_code >= 400:
            raise MathpixConvertError(
                f"Mathpix docx download failed ({response.status_code}): {response.text}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)

    def _download_md(self, pdf_id: str, output_path: Path) -> None:
        md_url = f"{MATHPIX_PDF_ENDPOINT}/{pdf_id}.md"
        response = requests.get(
            md_url,
            headers=self._credentials.headers,
            timeout=self._request_timeout,
        )
        if response.status_code >= 400:
            raise MathpixConvertError(
                f"Mathpix md download failed ({response.status_code}): {response.text}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)

    def _download_lines_json(self, pdf_id: str, output_path: Path) -> None:
        lines_url = f"{MATHPIX_PDF_ENDPOINT}/{pdf_id}.lines.json"
        response = requests.get(
            lines_url,
            headers=self._credentials.headers,
            timeout=self._request_timeout,
        )
        if response.status_code >= 400:
            raise MathpixConvertError(
                f"Mathpix lines.json download failed ({response.status_code}): "
                f"{response.text}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
