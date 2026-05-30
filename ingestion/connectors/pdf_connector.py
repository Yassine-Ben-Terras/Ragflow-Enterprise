"""
ingestion/connectors/pdf_connector.py
Ingests PDF files from a local directory.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Iterator, List

import pdfplumber

from ingestion.connectors.base import BaseConnector, Document

logger = logging.getLogger(__name__)


class PDFConnector(BaseConnector):

    @property
    def name(self) -> str:
        return "pdf"

    def __init__(self, source_dir: str) -> None:
        self.source_dir = Path(source_dir)

    def _pdf_to_text(self, data: bytes) -> str:
        text_parts: List[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    def fetch(self) -> Iterator[Document]:
        if not self.source_dir.exists():
            raise ValueError(f"Directory not found: {self.source_dir}")

        pdf_files = list(self.source_dir.rglob("*.pdf"))
        logger.info("Found %d PDF(s) in %s", len(pdf_files), self.source_dir)

        for path in pdf_files:
            try:
                text = self._pdf_to_text(path.read_bytes())
                if not text.strip():
                    logger.warning("Empty text from %s — skipping.", path)
                    continue
                source_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
                title = path.stem.replace("_", " ").replace("-", " ").title()
                yield Document(
                    source="pdf",
                    source_id=source_id,
                    title=title,
                    content=text,
                    file_path=str(path),
                    metadata={"filename": path.name, "char_count": len(text)},
                )
            except Exception as exc:
                logger.error("Failed to parse %s: %s", path, exc)
