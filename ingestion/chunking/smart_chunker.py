"""
ingestion/chunking/smart_chunker.py
Splits raw Documents into smaller Chunks suitable for embedding.

Three strategies:
  - recursive  : LangChain-style recursive character splitting (default, fast)
  - sentence   : Split on sentence boundaries, then merge up to chunk_size
  - semantic   : Placeholder — will be upgraded in Phase 2 with embedding-aware splitting
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List

from ingestion.connectors.base import Document

logger = logging.getLogger(__name__)

# Separators tried in order for recursive strategy
_RECURSIVE_SEPS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


@dataclass
class Chunk:
    """A document fragment ready for embedding."""

    chunk_id: str
    doc_source_id: str
    doc_source: str
    doc_title: str
    text: str
    chunk_index: int
    total_chunks: int               # filled in after all chunks are created
    url: str | None = None
    file_path: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_source_id": self.doc_source_id,
            "doc_source": self.doc_source,
            "doc_title": self.doc_title,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "url": self.url,
            "file_path": self.file_path,
            "metadata": self.metadata,
        }


class SmartChunker:
    """
    Converts a Document into a list of Chunks.

    Args:
        chunk_size:  Target token / character count per chunk.
        overlap:     Overlap between consecutive chunks.
        strategy:    "recursive" | "sentence" | "semantic"
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        strategy: str = "recursive",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

    # ── strategy dispatchers ─────────────────────────────────────────────

    def chunk(self, doc: Document) -> List[Chunk]:
        if self.strategy == "sentence":
            raw_chunks = self._sentence_split(doc.content)
        elif self.strategy == "semantic":
            raw_chunks = self._semantic_split(doc.content)
        else:
            raw_chunks = self._recursive_split(doc.content, _RECURSIVE_SEPS)

        chunks: List[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_source_id=doc.source_id,
                    doc_source=doc.source,
                    doc_title=doc.title,
                    text=text,
                    chunk_index=idx,
                    total_chunks=0,           # patched below
                    url=doc.url,
                    file_path=doc.file_path,
                    metadata={**doc.metadata, "strategy": self.strategy},
                )
            )

        total = len(chunks)
        for c in chunks:
            c.total_chunks = total

        logger.debug(
            "Document '%s' → %d chunk(s) [strategy=%s]",
            doc.source_id,
            total,
            self.strategy,
        )
        return chunks

    # ── recursive split ──────────────────────────────────────────────────

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text by trying separators in order.
        Applies overlap between consecutive chunks.
        """
        if not separators:
            return self._by_size(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep and sep in text:
            splits = text.split(sep)
        else:
            return self._recursive_split(text, remaining_seps)

        chunks: List[str] = []
        current = ""

        for split in splits:
            candidate = (current + sep + split).lstrip(sep) if current else split
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If split itself is too large, recurse on it
                if len(split) > self.chunk_size:
                    sub = self._recursive_split(split, remaining_seps)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    current = split

        if current:
            chunks.append(current)

        return self._apply_overlap(chunks)

    def _by_size(self, text: str) -> List[str]:
        """Hard character-size fallback."""
        step = self.chunk_size - self.overlap
        return [text[i : i + self.chunk_size] for i in range(0, len(text), max(step, 1))]

    # ── sentence split ───────────────────────────────────────────────────

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def _sentence_split(self, text: str) -> List[str]:
        sentences = self._SENTENCE_RE.split(text)
        chunks: List[str] = []
        current = ""

        for sent in sentences:
            candidate = (current + " " + sent).strip() if current else sent
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sent

        if current:
            chunks.append(current)

        return self._apply_overlap(chunks)

    # ── semantic split (stub) ────────────────────────────────────────────

    def _semantic_split(self, text: str) -> List[str]:
        """
        Placeholder: will be replaced in Phase 2 with embedding-aware
        split that groups semantically similar sentences together.
        Falls back to recursive for now.
        """
        logger.warning("Semantic chunking is not yet implemented — falling back to recursive.")
        return self._recursive_split(text, _RECURSIVE_SEPS)

    # ── overlap helper ───────────────────────────────────────────────────

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Prepend the tail of the previous chunk to the current one."""
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.overlap :]
            result.append(tail + " " + chunks[i])

        return result
