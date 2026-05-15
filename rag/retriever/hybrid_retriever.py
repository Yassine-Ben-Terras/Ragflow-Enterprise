"""
rag/retriever/hybrid_retriever.py
Hybrid retriever: combines Dense (semantic) + BM25 (lexical) results
using Reciprocal Rank Fusion (RRF).

RRF formula:  score(d) = Σ  1 / (k + rank_i(d))
              where k=60 is a smoothing constant, rank_i is position in list i.

RRF is rank-based — no score normalisation needed across heterogeneous systems.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

from embeddings.stores.base import SearchResult
from rag.retriever.bm25_retriever import BM25Retriever
from rag.retriever.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)

_RRF_K = 60   # standard RRF smoothing constant


class HybridRetriever:
    """
    Args:
        top_k:         Final number of results after fusion.
        dense_top_k:   Candidates fetched from dense retrieval.
        bm25_top_k:    Candidates fetched from BM25.
        dense_weight:  Relative weight applied to dense RRF scores.
        bm25_weight:   Relative weight applied to BM25 RRF scores.
        data_dir:      Local storage directory for BM25.
        score_threshold: Minimum cosine similarity for dense retrieval.
    """

    def __init__(
        self,
        top_k: int = 10,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        data_dir: str = "data",
        score_threshold: Optional[float] = None,
    ) -> None:
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self._dense = DenseRetriever(top_k=dense_top_k, score_threshold=score_threshold)
        self._bm25 = BM25Retriever(data_dir=data_dir, top_k=bm25_top_k)

    # ── fusion ───────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_scores(results: List[SearchResult], weight: float) -> Dict[str, float]:
        """Compute weighted RRF contribution for a ranked list."""
        scores: Dict[str, float] = {}
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] = weight * (1.0 / (_RRF_K + rank))
        return scores

    def _fuse(
        self,
        dense_results: List[SearchResult],
        bm25_results: List[SearchResult],
    ) -> List[SearchResult]:
        # Build a lookup: chunk_id → SearchResult (dense takes priority for metadata)
        all_results: Dict[str, SearchResult] = {}
        for r in bm25_results:
            all_results[r.chunk_id] = r
        for r in dense_results:
            all_results[r.chunk_id] = r  # overwrite with dense (richer score context)

        dense_scores = self._rrf_scores(dense_results, self.dense_weight)
        bm25_scores  = self._rrf_scores(bm25_results,  self.bm25_weight)

        # Sum contributions
        combined: Dict[str, float] = defaultdict(float)
        for chunk_id, score in dense_scores.items():
            combined[chunk_id] += score
        for chunk_id, score in bm25_scores.items():
            combined[chunk_id] += score

        # Sort by fused score descending
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[: self.top_k]

        fused: List[SearchResult] = []
        for chunk_id, fused_score in ranked:
            result = all_results[chunk_id]
            fused.append(SearchResult(
                chunk_id=result.chunk_id,
                doc_title=result.doc_title,
                doc_source=result.doc_source,
                text=result.text,
                score=fused_score,
                url=result.url,
                file_path=result.file_path,
                metadata={**result.metadata, "rrf_score": fused_score},
            ))
        return fused

    # ── public API ───────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[SearchResult]:
        dense_results = self._dense.retrieve(query)
        bm25_results  = self._bm25.retrieve(query)

        logger.debug(
            "Hybrid: dense=%d  bm25=%d → fusing…",
            len(dense_results), len(bm25_results),
        )

        fused = self._fuse(dense_results, bm25_results)
        logger.debug("Hybrid: fused → %d results (top_k=%d)", len(fused), self.top_k)
        return fused
