"""
embeddings/providers/base.py
Abstract interface every embedding provider must implement.
"""

from __future__ import annotations

import abc
from typing import List


class BaseEmbedder(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Provider name: 'openai' | 'bge'"""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Vector dimension produced by this model."""

    @abc.abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.
        Returns a list of float vectors, one per input text.
        """

    def embed_query(self, text: str) -> List[float]:
        """Convenience: embed a single query string."""
        return self.embed_texts([text])[0]
