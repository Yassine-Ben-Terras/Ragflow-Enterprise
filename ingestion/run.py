"""
ingestion/run.py
Orchestrates the full Phase-1 pipeline:
  connectors → chunker → S3 storage

Usage (one-shot):
    python -m ingestion.run
"""

from __future__ import annotations

import logging
from typing import List, Optional

from config.settings import settings
from ingestion.chunking.smart_chunker import SmartChunker
from ingestion.connectors.base import BaseConnector
from ingestion.connectors.confluence import ConfluenceConnector
from ingestion.connectors.git_connector import GitConnector
from ingestion.connectors.pdf_connector import PDFConnector
from ingestion.storage.s3_storage import S3Storage

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def build_connectors(pdf_dir: Optional[str] = None) -> List[BaseConnector]:
    connectors: List[BaseConnector] = []

    # ── PDF ──────────────────────────────────────────────
    if pdf_dir or settings.s3_bucket_name:
        connectors.append(
            PDFConnector(
                source_dir=pdf_dir,
                s3_bucket=settings.s3_bucket_name if not pdf_dir else None,
                s3_prefix=settings.s3_prefix,
                aws_region=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id or None,
                aws_secret_access_key=settings.aws_secret_access_key or None,
            )
        )

    # ── Confluence ───────────────────────────────────────
    if settings.confluence_url and settings.confluence_api_token:
        connectors.append(
            ConfluenceConnector(
                base_url=settings.confluence_url,
                username=settings.confluence_username,
                api_token=settings.confluence_api_token,
                spaces=settings.confluence_spaces,
            )
        )

    # ── Git ──────────────────────────────────────────────
    if settings.git_repos:
        connectors.append(
            GitConnector(
                repos=settings.git_repos,
                branch=settings.git_branch,
                file_extensions=settings.git_file_extensions,
            )
        )

    return connectors


def run_ingestion(pdf_dir: Optional[str] = None, skip_existing: bool = True) -> dict:
    """
    Full ingestion run.

    Returns a summary dict with counts per connector.
    """
    chunker = SmartChunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        strategy=settings.chunking_strategy,
    )

    storage = S3Storage(
        bucket=settings.s3_bucket_name,
        prefix="ragflow",
        region=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )

    summary: dict = {}

    for connector in build_connectors(pdf_dir=pdf_dir):
        doc_count = 0
        chunk_count = 0
        skipped = 0

        for doc in connector.fetch():
            if skip_existing and storage.document_exists(doc):
                skipped += 1
                logger.debug("Skipping already-stored doc: %s", doc.source_id)
                continue

            storage.save_document(doc)
            chunks = chunker.chunk(doc)
            storage.save_chunks(chunks)

            doc_count += 1
            chunk_count += len(chunks)

        summary[connector.name] = {
            "documents": doc_count,
            "chunks": chunk_count,
            "skipped": skipped,
        }
        logger.info(
            "[%s] docs=%d  chunks=%d  skipped=%d",
            connector.name,
            doc_count,
            chunk_count,
            skipped,
        )

    return summary


if __name__ == "__main__":
    result = run_ingestion()
    logger.info("Ingestion complete: %s", result)
