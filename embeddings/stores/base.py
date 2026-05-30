"""
embeddings/stores/base.py
Abstract interface every vector store must implement.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VectorRecord:
    """A chunk + its embedding, ready to upsert."""
    chunk_id: str
    doc_source_id: str
    doc_source: str
    doc_title: str
    text: str
    vector: List[float]
    url: Optional[str] = None
    file_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """One result returned from a similarity search."""
    chunk_id: str
    doc_title: str
    doc_source: str
    text: str
    score: float
    url: Optional[str] = None
    file_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseVectorStore(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def create_collection(self, dimension: int, recreate: bool = False) -> None:
        """Create (or recreate) the collection / table with HNSW index."""

    @abc.abstractmethod
    def upsert(self, records: List[VectorRecord]) -> None:
        """Insert or update a batch of vector records."""

    @abc.abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Return top-k most similar results."""

    @abc.abstractmethod
    def count(self) -> int:
        """Return total number of vectors stored."""
