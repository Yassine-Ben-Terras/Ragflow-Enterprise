"""
embeddings/stores/qdrant_store.py
Qdrant vector store with HNSW indexing for <100ms similarity search.

Run locally:
    docker run -p 6333:6333 qdrant/qdrant
"""

from __future__ import annotations

import logging
from typing import List, Optional

from embeddings.stores.base import BaseVectorStore, SearchResult, VectorRecord

logger = logging.getLogger(__name__)

# HNSW parameters — balance between build time and search speed/accuracy
_HNSW_M = 16             # number of edges per node (higher = better recall, more RAM)
_HNSW_EF_CONSTRUCT = 200 # build-time search width (higher = better quality index)


class QdrantStore(BaseVectorStore):
    """
    Args:
        host:        Qdrant server host.
        port:        Qdrant server port.
        collection:  Collection name.
    """

    @property
    def name(self) -> str:
        return "qdrant"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "ragflow",
    ) -> None:
        self.collection = collection
        self._client = None
        self._host = host
        self._port = port

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client is required. Install with: pip install qdrant-client"
                ) from exc
            self._client = QdrantClient(host=self._host, port=self._port)
            logger.info("Connected to Qdrant at %s:%d", self._host, self._port)
        return self._client

    def create_collection(self, dimension: int, recreate: bool = False) -> None:
        from qdrant_client.models import (
            Distance,
            HnswConfigDiff,
            VectorParams,
        )

        client = self._get_client()

        existing = [c.name for c in client.get_collections().collections]
        if self.collection in existing:
            if recreate:
                client.delete_collection(self.collection)
                logger.info("Deleted existing collection '%s'.", self.collection)
            else:
                logger.info("Collection '%s' already exists — skipping creation.", self.collection)
                return

        client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=_HNSW_M,
                    ef_construct=_HNSW_EF_CONSTRUCT,
                ),
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' (dim=%d, HNSW m=%d ef=%d).",
            self.collection, dimension, _HNSW_M, _HNSW_EF_CONSTRUCT,
        )

    def upsert(self, records: List[VectorRecord]) -> None:
        from qdrant_client.models import PointStruct

        client = self._get_client()
        points = [
            PointStruct(
                id=abs(hash(r.chunk_id)) % (2 ** 63),   # Qdrant needs uint64
                vector=r.vector,
                payload={
                    "chunk_id": r.chunk_id,
                    "doc_source_id": r.doc_source_id,
                    "doc_source": r.doc_source,
                    "doc_title": r.doc_title,
                    "text": r.text,
                    "url": r.url,
                    "file_path": r.file_path,
                    **r.metadata,
                },
            )
            for r in records
        ]
        client.upsert(collection_name=self.collection, points=points)
        logger.debug("Upserted %d vectors into '%s'.", len(records), self.collection)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        client = self._get_client()
        results = client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            SearchResult(
                chunk_id=r.payload.get("chunk_id", ""),
                doc_title=r.payload.get("doc_title", ""),
                doc_source=r.payload.get("doc_source", ""),
                text=r.payload.get("text", ""),
                score=r.score,
                url=r.payload.get("url"),
                file_path=r.payload.get("file_path"),
                metadata={k: v for k, v in r.payload.items()
                           if k not in ("chunk_id", "doc_title", "doc_source", "text", "url", "file_path")},
            )
            for r in results
        ]

    def count(self) -> int:
        client = self._get_client()
        info = client.get_collection(self.collection)
        return info.vectors_count or 0
