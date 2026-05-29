"""
ingestion/connectors/pdf_connector.py
Ingests PDF files from a local directory or directly from S3.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional

import boto3
import pdfplumber

from ingestion.connectors.base import BaseConnector, Document

logger = logging.getLogger(__name__)


class PDFConnector(BaseConnector):
    """
    Fetches PDFs from:
      - A local directory  (source_dir)
      - An S3 prefix       (s3_bucket + s3_prefix)

    Priority: local dir > S3 if both are provided.
    """

    @property
    def name(self) -> str:
        return "pdf"

    def __init__(
        self,
        source_dir: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "",
        aws_region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        self.source_dir = Path(source_dir) if source_dir else None
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self._s3_client = None

        if s3_bucket:
            self._s3_client = boto3.client(
                "s3",
                region_name=aws_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )

    # ── helpers ────────────────────────────────────────────────────────────

    def _pdf_to_text(self, data: bytes) -> str:
        """Extract full text from raw PDF bytes using pdfplumber."""
        text_parts: List[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        return "\n".join(text_parts)

    def _make_doc(self, content: str, filename: str, url: Optional[str] = None) -> Document:
        source_id = hashlib.sha256(filename.encode()).hexdigest()[:16]
        title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        return Document(
            source="pdf",
            source_id=source_id,
            title=title,
            content=content,
            url=url,
            file_path=filename,
            metadata={"filename": filename, "char_count": len(content)},
        )

    # ── fetch strategies ───────────────────────────────────────────────────

    def _fetch_from_dir(self) -> Iterator[Document]:
        assert self.source_dir is not None
        pdf_files = list(self.source_dir.rglob("*.pdf"))
        logger.info("Found %d PDF(s) in %s", len(pdf_files), self.source_dir)

        for path in pdf_files:
            try:
                data = path.read_bytes()
                text = self._pdf_to_text(data)
                if not text.strip():
                    logger.warning("Empty text from %s — skipping.", path)
                    continue
                yield self._make_doc(text, str(path))
            except Exception as exc:
                logger.error("Failed to parse %s: %s", path, exc)

    def _fetch_from_s3(self) -> Iterator[Document]:
        assert self._s3_client is not None and self.s3_bucket is not None

        paginator = self._s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if not key.lower().endswith(".pdf"):
                    continue
                try:
                    response = self._s3_client.get_object(Bucket=self.s3_bucket, Key=key)
                    data = response["Body"].read()
                    text = self._pdf_to_text(data)
                    if not text.strip():
                        logger.warning("Empty text from s3://%s/%s — skipping.", self.s3_bucket, key)
                        continue
                    url = f"s3://{self.s3_bucket}/{key}"
                    yield self._make_doc(text, key, url=url)
                except Exception as exc:
                    logger.error("Failed to fetch s3://%s/%s: %s", self.s3_bucket, key, exc)

    # ── public API ─────────────────────────────────────────────────────────

    def fetch(self) -> Iterator[Document]:
        if self.source_dir and self.source_dir.exists():
            yield from self._fetch_from_dir()
        elif self.s3_bucket and self._s3_client:
            yield from self._fetch_from_s3()
        else:
            raise ValueError("PDFConnector requires either source_dir or s3_bucket.")
