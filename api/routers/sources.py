"""
api/routers/sources.py
Returns a summary of all indexed sources from local storage.
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter

from api.schemas import SourceSummary, SourcesResponse
from config.settings import settings
from ingestion.storage.local_storage import LocalStorage

router = APIRouter()


@router.get("", response_model=SourcesResponse)
async def list_sources() -> SourcesResponse:
    storage = LocalStorage(data_dir=settings.data_dir)
    chunks = storage.load_all_chunks()

    # Aggregate by source type
    source_chunks: dict[str, int] = defaultdict(int)
    source_docs: dict[str, set] = defaultdict(set)

    for chunk in chunks:
        source_chunks[chunk.doc_source] += 1
        source_docs[chunk.doc_source].add(chunk.doc_source_id)

    summaries = [
        SourceSummary(
            source=source,
            total_chunks=source_chunks[source],
            doc_count=len(source_docs[source]),
        )
        for source in sorted(source_chunks)
    ]

    return SourcesResponse(
        sources=summaries,
        total_chunks=sum(source_chunks.values()),
    )
