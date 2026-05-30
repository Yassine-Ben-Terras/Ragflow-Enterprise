"""
ingestion/run.py
Orchestrates Phase-1: connectors → chunker → local storage

Usage:
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
from ingestion.storage.local_storage import LocalStorage

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def build_connectors() -> List[BaseConnector]:
    connectors: List[BaseConnector] = []

    if settings.pdf_source_dir:
        connectors.append(PDFConnector(source_dir=settings.pdf_source_dir))

    if settings.confluence_url and settings.confluence_api_token:
        connectors.append(ConfluenceConnector(
            base_url=settings.confluence_url,
            username=settings.confluence_username,
            api_token=settings.confluence_api_token,
            spaces=settings.confluence_spaces,
        ))

    if settings.git_repos:
        connectors.append(GitConnector(
            repos=settings.git_repos,
            branch=settings.git_branch,
            file_extensions=settings.git_file_extensions,
        ))

    return connectors


def run_ingestion(skip_existing: bool = True) -> dict:
    chunker = SmartChunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        strategy=settings.chunking_strategy,
    )
    storage = LocalStorage(data_dir=settings.data_dir)
    summary: dict = {}

    for connector in build_connectors():
        doc_count = chunk_count = skipped = 0

        for doc in connector.fetch():
            if skip_existing and storage.document_exists(doc):
                skipped += 1
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
        logger.info("[%s] docs=%d  chunks=%d  skipped=%d",
                    connector.name, doc_count, chunk_count, skipped)

    return summary


if __name__ == "__main__":
    result = run_ingestion()
    logger.info("Ingestion complete: %s", result)
