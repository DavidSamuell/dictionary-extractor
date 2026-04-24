"""Thin client for the Mathpix Convert PDF API.

Submits a PDF for conversion, polls until processing is complete, and
downloads the resulting .docx. See https://docs.mathpix.com/#process-a-pdf
for the authoritative API contract.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

MATHPIX_PDF_ENDPOINT = "https://api.mathpix.com/v3/pdf"


class MathpixConvertError(RuntimeError):
    """Raised when the Mathpix Convert API reports an error or a timeout."""


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
        """Convert a single PDF to DOCX via Mathpix and write to ``output_path``.

        Returns the output path on success. Raises ``MathpixConvertError``
        on API errors or timeouts.
        """
        pdf_id = self._submit(pdf_path)
        logger.info("Submitted %s -> pdf_id=%s", pdf_path.name, pdf_id)
        self._wait_until_complete(pdf_id)
        self._download_docx(pdf_id, output_path)
        return output_path

    def _submit(self, pdf_path: Path) -> str:
        options = {"conversion_formats": {"docx": True}}
        with pdf_path.open("rb") as fh:
            files = {"file": (pdf_path.name, fh, "application/pdf")}
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
