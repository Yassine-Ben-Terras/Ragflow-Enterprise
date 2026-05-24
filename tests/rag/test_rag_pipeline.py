"""tests/rag/test_rag_pipeline.py"""
from unittest.mock import MagicMock, patch
import pytest

from embeddings.stores.base import SearchResult
from rag.prompt.prompt_builder import PromptBuilder
from rag.reranker.cross_encoder_reranker import CrossEncoderReranker
from rag.retriever.hybrid_retriever import HybridRetriever


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_result(chunk_id: str, text: str, score: float, title: str = "Doc") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_title=title,
        doc_source="test",
        text=text,
        score=score,
        url=f"https://example.com/{chunk_id}",
    )


# ── PromptBuilder ─────────────────────────────────────────────────────────────

class TestPromptBuilder:
    def test_citations_numbered_from_one(self):
        builder = PromptBuilder()
        results = [
            _make_result("c1", "Paris is the capital of France.", 0.9, "Geography"),
            _make_result("c2", "The Eiffel Tower is in Paris.", 0.8, "Landmarks"),
        ]
        prompt = builder.build("Where is the Eiffel Tower?", results)
        assert len(prompt.citations) == 2
        assert prompt.citations[0].index == 1
        assert prompt.citations[1].index == 2

    def test_source_labels_in_user_message(self):
        builder = PromptBuilder()
        results = [_make_result("c1", "Some context text.", 0.9)]
        prompt = builder.build("Test question?", results)
        assert "[SOURCE 1]" in prompt.user_message

    def test_context_window_budget(self):
        builder = PromptBuilder(max_context_chars=100)
        # 10 passages that would exceed the budget together
        results = [_make_result(f"c{i}", "x" * 50, 0.9 - i * 0.01) for i in range(10)]
        prompt = builder.build("query", results)
        # Should trim to fit within budget
        assert len(prompt.citations) < 10

    def test_empty_results_returns_empty_citations(self):
        builder = PromptBuilder()
        prompt = builder.build("anything", [])
        assert prompt.citations == []
        assert "Question" in prompt.user_message


# ── HybridRetriever — RRF fusion ─────────────────────────────────────────────

class TestHybridRetrieverFusion:
    def test_rrf_scores_sum_contributions(self):
        r1 = [_make_result("a", "text a", 0.9), _make_result("b", "text b", 0.8)]
        r2 = [_make_result("b", "text b", 5.0), _make_result("c", "text c", 4.0)]

        from rag.retriever.hybrid_retriever import HybridRetriever
        h = HybridRetriever.__new__(HybridRetriever)
        h.top_k = 3
        h.dense_weight = 0.6
        h.bm25_weight = 0.4

        fused = h._fuse(r1, r2)
        ids = [r.chunk_id for r in fused]

        # 'b' appears in both lists → should rank first
        assert ids[0] == "b"

    def test_fuse_respects_top_k(self):
        from rag.retriever.hybrid_retriever import HybridRetriever
        h = HybridRetriever.__new__(HybridRetriever)
        h.top_k = 2
        h.dense_weight = 0.6
        h.bm25_weight = 0.4

        r = [_make_result(f"c{i}", f"text {i}", float(i)) for i in range(5)]
        fused = h._fuse(r, [])
        assert len(fused) <= 2


# ── CrossEncoderReranker ─────────────────────────────────────────────────────

class TestCrossEncoderReranker:
    def test_returns_top_k(self):
        reranker = CrossEncoderReranker(top_k=2)
        import numpy as np

        candidates = [_make_result(f"c{i}", f"passage {i}", 0.5) for i in range(5)]

        with patch.object(reranker, "_load_model"):
            reranker._model = MagicMock()
            reranker._model.predict.return_value = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
            result = reranker.rerank("query", candidates)

        assert len(result) == 2
        assert result[0].chunk_id == "c1"   # highest score 0.9
        assert result[1].chunk_id == "c3"   # second 0.7

    def test_empty_candidates_returns_empty(self):
        reranker = CrossEncoderReranker(top_k=5)
        assert reranker.rerank("query", []) == []

    def test_cross_encoder_score_in_metadata(self):
        reranker = CrossEncoderReranker(top_k=1)
        import numpy as np

        candidates = [_make_result("c1", "text", 0.5)]
        with patch.object(reranker, "_load_model"):
            reranker._model = MagicMock()
            reranker._model.predict.return_value = np.array([0.85])
            result = reranker.rerank("q", candidates)

        assert "cross_encoder_score" in result[0].metadata
        assert abs(result[0].metadata["cross_encoder_score"] - 0.85) < 1e-4


# ── RAGPipeline integration (mocked LLM) ────────────────────────────────────

class TestRAGPipelineIntegration:
    def test_query_returns_rag_response(self):
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline.__new__(RAGPipeline)

        # Mock sub-components
        pipeline._retriever = MagicMock()
        pipeline._retriever.retrieve.return_value = [
            _make_result("c1", "Python is a programming language.", 0.9)
        ]
        pipeline._reranker = MagicMock()
        pipeline._reranker.rerank.return_value = [
            _make_result("c1", "Python is a programming language.", 0.85)
        ]
        pipeline._prompt_builder = PromptBuilder()
        pipeline._llm_model = "gpt-4o-mini"
        pipeline._temperature = 0.2
        pipeline._llm_client = None
        pipeline._api_key = "fake"

        with patch.object(pipeline, "_call_llm", return_value="Python is a language [SOURCE 1]."):
            response = pipeline.query("What is Python?")

        assert "Python" in response.answer
        assert response.query == "What is Python?"
        assert len(response.citations) >= 1

    def test_no_candidates_returns_fallback(self):
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._retriever = MagicMock()
        pipeline._retriever.retrieve.return_value = []
        pipeline._reranker = MagicMock()
        pipeline._prompt_builder = PromptBuilder()
        pipeline._llm_model = "gpt-4o-mini"
        pipeline._llm_client = None
        pipeline._api_key = "fake"

        response = pipeline.query("unknown topic")
        assert "could not find" in response.answer.lower()
        assert response.citations == []
