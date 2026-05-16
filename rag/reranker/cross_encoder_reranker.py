"""
rag/reranker/cross_encoder_reranker.py
Cross-encoder reranker: scores (query, passage) pairs jointly for higher precision.

Unlike bi-encoders (used in dense retrieval), cross-encoders attend to both
query and passage simultaneously — much more accurate but too slow for full
corpus search. We apply them only to the small candidate set from the hybrid retriever.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~22M params, fast on CPU
  - Trained on MS-MARCO passage ranking

Alternative: cross-encoder/ms-marco-MiniLM-L-12-v2 (heavier, more accurate)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from embeddings.stores.base import SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Args:
        model_name:  HuggingFace cross-encoder model id.
        top_k:       Number of results to return after reranking.
        device:      "cpu" | "cuda" | "mps" | None (auto-detect).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        top_k: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self._device = device
        self._model = None   # lazy load

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading cross-encoder model: %s …", self.model_name)
        self._model = CrossEncoder(self.model_name, device=self._device)
        logger.info("Cross-encoder loaded.")

    def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """
        Score each (query, passage) pair and return top_k sorted by cross-encoder score.

        Args:
            query:       User question.
            candidates:  Results from hybrid retrieval.

        Returns:
            Re-ranked list of SearchResult, limited to top_k.
        """
        if not candidates:
            return []

        self._load_model()

        pairs = [(query, c.text) for c in candidates]
        scores: List[float] = self._model.predict(pairs).tolist()

        # Attach cross-encoder score and sort
        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )[: self.top_k]

        reranked: List[SearchResult] = []
        for ce_score, result in scored:
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    doc_title=result.doc_title,
                    doc_source=result.doc_source,
                    text=result.text,
                    score=float(ce_score),
                    url=result.url,
                    file_path=result.file_path,
                    metadata={
                        **result.metadata,
                        "cross_encoder_score": float(ce_score),
                        "original_rrf_score": result.metadata.get("rrf_score", result.score),
                    },
                )
            )

        logger.debug(
            "Reranked %d candidates → top %d  (scores: %.3f … %.3f)",
            len(candidates),
            len(reranked),
            reranked[0].score if reranked else 0.0,
            reranked[-1].score if reranked else 0.0,
        )
        return reranked
