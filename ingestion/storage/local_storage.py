"""
ingestion/storage/local_storage.py
Persists raw Documents and Chunks to the local filesystem as JSON / JSONL.

Layout:
  data/
  ├── raw/<source>/<source_id>.json          ← raw Document
  └── chunks/<source>/<doc_source_id>.jsonl  ← all chunks for a doc
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from ingestion.chunking.smart_chunker import Chunk
from ingestion.connectors.base import Document

logger = logging.getLogger(__name__)


class LocalStorage:
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self._raw_dir = self.data_dir / "raw"
        self._chunks_dir = self.data_dir / "chunks"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_dir.mkdir(parents=True, exist_ok=True)

    # ── paths ────────────────────────────────────────────────────────────

    def _raw_path(self, doc: Document) -> Path:
        d = self._raw_dir / doc.source
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{doc.source_id}.json"

    def _chunks_path(self, doc_source: str, doc_source_id: str) -> Path:
        d = self._chunks_dir / doc_source
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{doc_source_id}.jsonl"

    # ── public API ────────────────────────────────────────────────────────

    def save_document(self, doc: Document) -> Path:
        path = self._raw_path(doc)
        path.write_text(json.dumps(doc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Saved document → %s", path)
        return path

    def save_chunks(self, chunks: List[Chunk]) -> Path:
        if not chunks:
            raise ValueError("chunks list is empty")
        path = self._chunks_path(chunks[0].doc_source, chunks[0].doc_source_id)
        lines = "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks)
        path.write_text(lines, encoding="utf-8")
        logger.debug("Saved %d chunk(s) → %s", len(chunks), path)
        return path

    def load_chunks(self, doc_source: str, doc_source_id: str) -> List[Chunk]:
        path = self._chunks_path(doc_source, doc_source_id)
        if not path.exists():
            return []
        chunks: List[Chunk] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            chunks.append(Chunk(
                chunk_id=d["chunk_id"],
                doc_source_id=d["doc_source_id"],
                doc_source=d["doc_source"],
                doc_title=d["doc_title"],
                text=d["text"],
                chunk_index=d["chunk_index"],
                total_chunks=d["total_chunks"],
                url=d.get("url"),
                file_path=d.get("file_path"),
                metadata=d.get("metadata", {}),
            ))
        return chunks

    def load_all_chunks(self) -> List[Chunk]:
        """Load every chunk from every source — used by the embedding pipeline."""
        all_chunks: List[Chunk] = []
        for jsonl_file in self._chunks_dir.rglob("*.jsonl"):
            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                all_chunks.append(Chunk(
                    chunk_id=d["chunk_id"],
                    doc_source_id=d["doc_source_id"],
                    doc_source=d["doc_source"],
                    doc_title=d["doc_title"],
                    text=d["text"],
                    chunk_index=d["chunk_index"],
                    total_chunks=d["total_chunks"],
                    url=d.get("url"),
                    file_path=d.get("file_path"),
                    metadata=d.get("metadata", {}),
                ))
        return all_chunks

    def document_exists(self, doc: Document) -> bool:
        return self._raw_path(doc).exists()

    def list_chunk_files(self, source: Optional[str] = None) -> List[Path]:
        base = self._chunks_dir / source if source else self._chunks_dir
        return list(base.rglob("*.jsonl")) if base.exists() else []
