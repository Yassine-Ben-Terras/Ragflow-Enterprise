"""
ingestion/connectors/base.py
Abstract base class that every source connector must implement.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Canonical unit of raw content before chunking."""

    source: str                          # connector type: "pdf" | "confluence" | "git"
    source_id: str                       # unique identifier within the source
    title: str
    content: str
    url: Optional[str] = None
    file_path: Optional[str] = None
    language: str = "en"
    metadata: dict = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "file_path": self.file_path,
            "language": self.language,
            "metadata": self.metadata,
            "ingested_at": self.ingested_at.isoformat(),
        }


class BaseConnector(abc.ABC):
    """
    Interface every source connector must implement.

    Usage:
        class MyConnector(BaseConnector):
            def fetch(self) -> Iterator[Document]:
                yield Document(...)
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""

    @abc.abstractmethod
    def fetch(self) -> Iterator[Document]:
        """
        Yield Document objects from the source.
        Must be a generator so callers can stream large datasets.
        """

    def run(self) -> list[Document]:
        """Convenience wrapper: collect all documents into a list."""
        logger.info("[%s] Starting fetch …", self.name)
        docs: list[Document] = []
        for doc in self.fetch():
            docs.append(doc)
            logger.debug("[%s] Fetched: %s", self.name, doc.source_id)
        logger.info("[%s] Fetched %d document(s).", self.name, len(docs))
        return docs
