"""tests/api/test_api.py"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from rag.pipeline import RAGResponse
from rag.prompt.prompt_builder import Citation


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _mock_response(query: str = "test") -> RAGResponse:
    return RAGResponse(
        query=query,
        answer="The answer is 42 [SOURCE 1].",
        citations=[
            Citation(
                index=1,
                chunk_id="abc123",
                doc_title="Test Doc",
                doc_source="pdf",
                text_snippet="The answer is 42.",
                url="https://example.com/doc",
                file_path=None,
                score=0.92,
            )
        ],
        model="gpt-4o-mini",
        metadata={"num_candidates": 5, "num_reranked": 1},
    )


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


# ── Chat JSON ─────────────────────────────────────────────────────────────────

class TestChatEndpoint:
    def test_chat_returns_200(self, client):
        with patch("api.routers.chat.AppState.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.query.return_value = _mock_response("What is the answer?")
            mock_get.return_value = mock_pipeline

            resp = client.post("/chat", json={"query": "What is the answer?"})
        assert resp.status_code == 200

    def test_chat_response_schema(self, client):
        with patch("api.routers.chat.AppState.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.query.return_value = _mock_response("Test?")
            mock_get.return_value = mock_pipeline

            data = client.post("/chat", json={"query": "Test?"}).json()

        assert "answer" in data
        assert "citations" in data
        assert "model" in data
        assert isinstance(data["citations"], list)

    def test_chat_empty_query_rejected(self, client):
        resp = client.post("/chat", json={"query": ""})
        assert resp.status_code == 422    # Pydantic min_length=1

    def test_chat_citations_have_index(self, client):
        with patch("api.routers.chat.AppState.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.query.return_value = _mock_response("q")
            mock_get.return_value = mock_pipeline

            data = client.post("/chat", json={"query": "q"}).json()

        for c in data["citations"]:
            assert "index" in c
            assert "doc_title" in c
            assert "score" in c


# ── SSE Streaming ────────────────────────────────────────────────────────────

class TestStreamEndpoint:
    def test_stream_returns_200(self, client):
        async def _fake_generator():
            import json
            yield f"data: {json.dumps({'type': 'token', 'content': 'Hello'})}\n\n"
            yield f"data: {json.dumps({'type': 'done',  'content': ''})}\n\n"

        with patch("api.routers.chat.HybridRetriever") as MockHybrid, \
             patch("api.routers.chat.CrossEncoderReranker") as MockReranker, \
             patch("api.routers.chat.AsyncOpenAI"):

            MockHybrid.return_value.retrieve = MagicMock(return_value=[])

            resp = client.post(
                "/chat/stream",
                json={"query": "hello?"},
                headers={"Accept": "text/event-stream"},
            )
        # Even with empty candidates the endpoint must return 200
        assert resp.status_code == 200


# ── Sources ───────────────────────────────────────────────────────────────────

class TestSourcesEndpoint:
    def test_sources_returns_200(self, client):
        with patch("api.routers.sources.LocalStorage") as MockStorage:
            MockStorage.return_value.load_all_chunks.return_value = []
            resp = client.get("/sources")
        assert resp.status_code == 200

    def test_sources_schema(self, client):
        with patch("api.routers.sources.LocalStorage") as MockStorage:
            MockStorage.return_value.load_all_chunks.return_value = []
            data = client.get("/sources").json()
        assert "sources" in data
        assert "total_chunks" in data
