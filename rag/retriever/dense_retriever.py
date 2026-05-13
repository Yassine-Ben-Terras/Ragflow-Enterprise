"""
rag/retriever/dense_retriever.py
Dense retrieval using the vector store built in Phase 2.
Embeds the query and returns top-k semantically similar chunks.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from embeddings.pipeline import build_embedder, build_vector_store
from embeddings.stores.base import SearchResult

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Args:
        top_k:            Number of results to return.
        score_threshold:  Minimum cosine similarity score (0–1). None = no filter.
    """

    def __init__(
        self,
        top_k: int = 20,
        score_threshold: Optional[float] = None,
    ) -> None:
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._embedder = build_embedder()
        self._store = build_vector_store()

    def retrieve(self, query: str) -> List[SearchResult]:
        query_vector = self._embedder.embed_query(query)
        results = self._store.search(
            query_vector=query_vector,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )
        logger.debug("Dense retrieval: query=%r → %d results", query[:60], len(results))
        return results
