"""
embeddings/pipeline.py
Phase-2 pipeline: loads chunks from local storage → embeds → upserts into vector store.

Usage:
    python -m embeddings.pipeline
"""

from __future__ import annotations

import logging
from typing import List

from config.settings import settings
from embeddings.providers.base import BaseEmbedder
from embeddings.stores.base import BaseVectorStore, VectorRecord
from ingestion.chunking.smart_chunker import Chunk
from ingestion.storage.local_storage import LocalStorage

logger = logging.getLogger(__name__)


def build_embedder() -> BaseEmbedder:
    if settings.embedding_provider == "bge":
        from embeddings.providers.bge_embedder import BGEEmbedder
        return BGEEmbedder(
            model_name=settings.bge_model_name,
            batch_size=settings.embedding_batch_size,
        )
    # default: openai
    from embeddings.providers.openai_embedder import OpenAIEmbedder
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )


def build_vector_store() -> BaseVectorStore:
    if settings.vector_store == "pgvector":
        from embeddings.stores.pgvector_store import PGVectorStore
        return PGVectorStore(dsn=settings.pgvector_dsn)
    # default: qdrant
    from embeddings.stores.qdrant_store import QdrantStore
    return QdrantStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection,
    )


def chunks_to_records(chunks: List[Chunk], vectors: List[List[float]]) -> List[VectorRecord]:
    return [
        VectorRecord(
            chunk_id=c.chunk_id,
            doc_source_id=c.doc_source_id,
            doc_source=c.doc_source,
            doc_title=c.doc_title,
            text=c.text,
            vector=v,
            url=c.url,
            file_path=c.file_path,
            metadata=c.metadata,
        )
        for c, v in zip(chunks, vectors)
    ]


def run_embedding_pipeline(recreate_collection: bool = False) -> dict:
    """
    Load all chunks from local storage, embed them, and upsert into the vector store.
    Returns a summary dict.
    """
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    storage = LocalStorage(data_dir=settings.data_dir)
    embedder = build_embedder()
    store = build_vector_store()

    # ── Load chunks ───────────────────────────────────────────────────────
    logger.info("Loading chunks from local storage…")
    chunks = storage.load_all_chunks()
    if not chunks:
        logger.warning("No chunks found. Run ingestion first: python -m ingestion.run")
        return {"chunks_embedded": 0, "provider": embedder.name, "store": store.name}

    logger.info("Loaded %d chunks. Provider=%s  Store=%s", len(chunks), embedder.name, store.name)

    # ── Ensure collection exists ──────────────────────────────────────────
    store.create_collection(dimension=embedder.dimension, recreate=recreate_collection)

    # ── Embed in batches ─────────────────────────────────────────────────
    total_upserted = 0
    batch_size = settings.embedding_batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]

        vectors = embedder.embed_texts(texts)
        records = chunks_to_records(batch, vectors)
        store.upsert(records)

        total_upserted += len(records)
        logger.info(
            "Progress: %d / %d chunks embedded and stored.",
            total_upserted, len(chunks),
        )

    total_in_store = store.count()
    logger.info(
        "Embedding pipeline complete. Upserted=%d  Total in store=%d",
        total_upserted, total_in_store,
    )

    return {
        "chunks_embedded": total_upserted,
        "total_in_store": total_in_store,
        "provider": embedder.name,
        "store": store.name,
        "dimension": embedder.dimension,
    }


if __name__ == "__main__":
    result = run_embedding_pipeline()
    print(result)
