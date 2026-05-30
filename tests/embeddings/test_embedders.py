"""tests/embeddings/test_embedders.py"""
from unittest.mock import MagicMock, patch
import pytest

from embeddings.providers.base import BaseEmbedder


class TestBaseEmbedder:
    def test_embed_query_calls_embed_texts(self):
        class DummyEmbedder(BaseEmbedder):
            @property
            def name(self): return "dummy"
            @property
            def dimension(self): return 4
            def embed_texts(self, texts):
                return [[0.1, 0.2, 0.3, 0.4]] * len(texts)

        emb = DummyEmbedder()
        vec = emb.embed_query("hello")
        assert len(vec) == 4
        assert vec == [0.1, 0.2, 0.3, 0.4]


class TestOpenAIEmbedder:
    def test_batch_splitting(self):
        """Texts exceeding batch_size should be split into multiple API calls."""
        from embeddings.providers.openai_embedder import OpenAIEmbedder

        emb = OpenAIEmbedder(api_key="fake", batch_size=2)

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(index=0, embedding=[0.1, 0.2]),
            MagicMock(index=1, embedding=[0.3, 0.4]),
        ]

        with patch.object(emb._client.embeddings, "create", return_value=mock_response) as mock_create:
            results = emb.embed_texts(["a", "b", "c", "d"])
            assert mock_create.call_count == 2   # 4 texts / batch_size 2
            assert len(results) == 4

    def test_empty_string_replaced(self):
        """Empty strings must be replaced with a space before sending to API."""
        from embeddings.providers.openai_embedder import OpenAIEmbedder

        emb = OpenAIEmbedder(api_key="fake", batch_size=10)
        mock_response = MagicMock()
        mock_response.data = [MagicMock(index=0, embedding=[0.0])]

        with patch.object(emb._client.embeddings, "create", return_value=mock_response) as mock_create:
            emb.embed_texts([""])
            call_args = mock_create.call_args
            assert call_args.kwargs["input"] == [" "]


class TestPipelineBuilder:
    def test_build_embedder_openai(self, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings, "embedding_provider", "openai")
        monkeypatch.setattr(settings, "openai_api_key", "fake-key")

        from embeddings.pipeline import build_embedder
        emb = build_embedder()
        assert emb.name == "openai"

    def test_build_vector_store_qdrant(self, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings, "vector_store", "qdrant")

        from embeddings.pipeline import build_vector_store
        store = build_vector_store()
        assert store.name == "qdrant"

    def test_build_vector_store_pgvector(self, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings, "vector_store", "pgvector")

        from embeddings.pipeline import build_vector_store
        store = build_vector_store()
        assert store.name == "pgvector"
