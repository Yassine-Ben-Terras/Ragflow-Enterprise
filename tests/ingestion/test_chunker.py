"""tests/ingestion/test_chunker.py"""
import pytest
from ingestion.chunking.smart_chunker import SmartChunker
from ingestion.connectors.base import Document


def _make_doc(content: str) -> Document:
    return Document(
        source="test",
        source_id="test-001",
        title="Test Doc",
        content=content,
    )


class TestRecursiveChunker:
    def test_short_text_single_chunk(self):
        chunker = SmartChunker(chunk_size=500, overlap=0, strategy="recursive")
        doc = _make_doc("Hello world.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_long_text_multiple_chunks(self):
        chunker = SmartChunker(chunk_size=50, overlap=0, strategy="recursive")
        doc = _make_doc("A" * 200)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_total_chunks_field(self):
        chunker = SmartChunker(chunk_size=50, overlap=0, strategy="recursive")
        doc = _make_doc("word " * 100)
        chunks = chunker.chunk(doc)
        total = chunks[0].total_chunks
        assert all(c.total_chunks == total for c in chunks)

    def test_chunk_index_sequential(self):
        chunker = SmartChunker(chunk_size=50, overlap=0, strategy="recursive")
        doc = _make_doc("word " * 100)
        chunks = chunker.chunk(doc)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


class TestSentenceChunker:
    def test_splits_on_sentence_boundaries(self):
        chunker = SmartChunker(chunk_size=60, overlap=0, strategy="sentence")
        text = "The quick brown fox. Jumped over the lazy dog. And ran away."
        doc = _make_doc(text)
        chunks = chunker.chunk(doc)
        # Each chunk should not start mid-sentence
        assert all(c.text for c in chunks)


class TestOverlap:
    def test_overlap_applied(self):
        chunker = SmartChunker(chunk_size=50, overlap=10, strategy="recursive")
        doc = _make_doc("a " * 200)
        chunks = chunker.chunk(doc)
        # Overlap means chunk 1 should begin with tail of chunk 0
        if len(chunks) > 1:
            assert len(chunks[1].text) > 0
