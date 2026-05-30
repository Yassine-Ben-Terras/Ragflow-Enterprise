"""
embeddings/providers/openai_embedder.py
Embeds text using the OpenAI Embeddings API.

Default model: text-embedding-3-small (1536-dim, cheapest, fast)
Also supports:  text-embedding-3-large (3072-dim, best quality)
"""

from __future__ import annotations

import logging
import time
from typing import List

from openai import OpenAI, RateLimitError

from embeddings.providers.base import BaseEmbedder

logger = logging.getLogger(__name__)

_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0   # seconds


class OpenAIEmbedder(BaseEmbedder):
    """
    Args:
        api_key:    OpenAI API key (or reads OPENAI_API_KEY from env).
        model:      Embedding model name.
        batch_size: Max texts per API call (OpenAI allows up to 2048).
    """

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        return _DIMENSIONS.get(self.model, 1536)

    def __init__(
        self,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key or None)  # falls back to OPENAI_API_KEY env var

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        all_vectors: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Replace empty strings — OpenAI rejects them
            batch = [t if t.strip() else " " for t in batch]

            for attempt in range(_MAX_RETRIES):
                try:
                    response = self._client.embeddings.create(
                        model=self.model,
                        input=batch,
                    )
                    vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                    all_vectors.extend(vectors)
                    logger.debug(
                        "OpenAI embedded batch %d/%d (%d texts)",
                        i // self.batch_size + 1,
                        -(-len(texts) // self.batch_size),
                        len(batch),
                    )
                    break
                except RateLimitError:
                    wait = _RETRY_DELAY * (2 ** attempt)
                    logger.warning("Rate limited — retrying in %.1fs (attempt %d)", wait, attempt + 1)
                    time.sleep(wait)
                except Exception as exc:
                    logger.error("OpenAI embedding error: %s", exc)
                    raise

        return all_vectors
