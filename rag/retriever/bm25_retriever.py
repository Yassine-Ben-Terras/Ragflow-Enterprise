"""
rag/retriever/bm25_retriever.py
BM25 sparse retrieval over all locally stored chunks.

Uses the `rank-bm25` library (pure Python, no external service needed).
The index is built lazily on first query and cached in memory.

BM25 excels at:
  - Exact keyword matches
  - Rare technical terms
  - Queries where semantic similarity alone misses precise tokens
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass
from typing import List, Optional

from embeddings.stores.base import SearchResult
from ingestion.storage.local_storage import LocalStorage

logger = logging.getLogger(__name__)

_PUNCT = str.maketrans("", "", string.punctuation)


@dataclass
class _IndexedChunk:
    chunk_id: str
    doc_source_id: str
    doc_source: str
    doc_title: str
    text: str
    url: Optional[str]
    file_path: Optional[str]
    metadata: dict


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return text.lower().translate(_PUNCT).split()


class BM25Retriever:
    """
    Args:
        data_dir:  Root of local storage (same as LocalStorage data_dir).
        top_k:     Number of results to return.
    """

    def __init__(self, data_dir: str = "data", top_k: int = 20) -> None:
        self.top_k = top_k
        self._data_dir = data_dir
        self._bm25 = None
        self._index: List[_IndexedChunk] = []

    # ── lazy index build ────────────────────────────────────────────────

    def _build_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required. Install with: pip install rank-bm25"
            ) from exc

        storage = LocalStorage(data_dir=self._data_dir)
        chunks = storage.load_all_chunks()

        if not chunks:
            logger.warning("BM25: no chunks found in '%s'. Run ingestion first.", self._data_dir)
            self._bm25 = None
            return

        self._index = [
            _IndexedChunk(
                chunk_id=c.chunk_id,
                doc_source_id=c.doc_source_id,
                doc_source=c.doc_source,
                doc_title=c.doc_title,
                text=c.text,
                url=c.url,
                file_path=c.file_path,
                metadata=c.metadata,
            )
            for c in chunks
        ]

        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built: %d chunks indexed.", len(self._index))

    def _ensure_index(self) -> None:
        if self._bm25 is None:
            self._build_index()

    # ── public API ──────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[SearchResult]:
        self._ensure_index()

        if self._bm25 is None or not self._index:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair (score, index), sort descending, take top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: self.top_k]

        results: List[SearchResult] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            chunk = self._index[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_title=chunk.doc_title,
                    doc_source=chunk.doc_source,
                    text=chunk.text,
                    score=float(score),
                    url=chunk.url,
                    file_path=chunk.file_path,
                    metadata=chunk.metadata,
                )
            )

        logger.debug("BM25 retrieval: query=%r → %d results", query[:60], len(results))
        return results

    def invalidate_index(self) -> None:
        """Call after new ingestion to force index rebuild on next query."""
        self._bm25 = None
        self._index = []
        logger.info("BM25 index invalidated.")
