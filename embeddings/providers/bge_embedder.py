"""
embeddings/providers/bge_embedder.py
Local embeddings using BAAI/bge-m3 via sentence-transformers.

BGE-M3 supports:
  - Dense retrieval (1024-dim)
  - Sparse retrieval (lexical weights)
  - Multi-vector retrieval (ColBERT-style)

We use dense vectors here; sparse/multi-vector will be activated in Phase 3.

Model is downloaded automatically on first run (~2.3 GB).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from embeddings.providers.base import BaseEmbedder

logger = logging.getLogger(__name__)

_BGE_DIMENSION = 1024


class BGEEmbedder(BaseEmbedder):
    """
    Args:
        model_name:  HuggingFace model id. Default: "BAAI/bge-m3".
        batch_size:  Texts per inference batch.
        device:      "cpu" | "cuda" | "mps" | None (auto-detect).
        normalize:   L2-normalize vectors (recommended for cosine similarity).
    """

    @property
    def name(self) -> str:
        return "bge"

    @property
    def dimension(self) -> int:
        return _BGE_DIMENSION

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None      # lazy load
        self._device = device

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for BGE embeddings. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading BGE model: %s (first run may take a while)…", self.model_name)
        self._model = SentenceTransformer(
            self.model_name,
            device=self._device,
        )
        logger.info("BGE model loaded on device: %s", self._model.device)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._load_model()

        # Replace empty strings
        texts = [t if t.strip() else " " for t in texts]

        all_vectors: List[List[float]] = []
        total_batches = -(-len(texts) // self.batch_size)

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self._model.encode(
                batch,
                batch_size=len(batch),
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            all_vectors.extend(vectors.tolist())
            logger.debug(
                "BGE embedded batch %d/%d (%d texts)",
                i // self.batch_size + 1,
                total_batches,
                len(batch),
            )

        return all_vectors
