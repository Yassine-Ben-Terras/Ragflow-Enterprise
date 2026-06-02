"""tests/monitoring/test_monitoring.py"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from monitoring.feedback.feedback_store import FeedbackEntry, FeedbackStore
from monitoring.ragas.evaluator import EvalResult, RAGAsEvaluator


# ── FeedbackStore ─────────────────────────────────────────────────────────────

class TestFeedbackStore:
    def test_submit_persists_to_jsonl(self, tmp_path):
        store = FeedbackStore(data_dir=str(tmp_path))
        entry = FeedbackEntry(
            query="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            rating="thumbs_up",
        )
        store.submit(entry)

        feedback_file = tmp_path / "feedback" / "feedback.jsonl"
        assert feedback_file.exists()
        lines = feedback_file.read_text().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["query"] == "What is RAG?"
        assert data["rating"] == "thumbs_up"

    def test_load_negative_filters_correctly(self, tmp_path):
        store = FeedbackStore(data_dir=str(tmp_path))
        store.submit(FeedbackEntry(query="q1", answer="a1", rating="thumbs_up"))
        store.submit(FeedbackEntry(query="q2", answer="a2", rating="thumbs_down"))
        store.submit(FeedbackEntry(query="q3", answer="a3", rating="thumbs_down"))

        negative = store.load_negative()
        assert len(negative) == 2
        assert all(e.rating == "thumbs_down" for e in negative)

    def test_summary_counts_correctly(self, tmp_path):
        store = FeedbackStore(data_dir=str(tmp_path))
        store.submit(FeedbackEntry(query="q1", answer="a", rating="thumbs_up"))
        store.submit(FeedbackEntry(query="q2", answer="a", rating="thumbs_up"))
        store.submit(FeedbackEntry(query="q3", answer="a", rating="thumbs_down"))

        summary = store.summary()
        assert summary["total"] == 3
        assert summary["thumbs_up"] == 2
        assert summary["thumbs_down"] == 1
        assert abs(summary["satisfaction_rate"] - 0.667) < 0.01

    def test_empty_store_summary(self, tmp_path):
        store = FeedbackStore(data_dir=str(tmp_path))
        summary = store.summary()
        assert summary["total"] == 0
        assert summary["satisfaction_rate"] is None


# ── RAGAsEvaluator ────────────────────────────────────────────────────────────

class TestRAGAsEvaluator:
    def _make_evaluator(self, tmp_path) -> RAGAsEvaluator:
        return RAGAsEvaluator(
            api_key="fake",
            model="gpt-4o-mini",
            results_dir=str(tmp_path / "ragas"),
        )

    def test_evaluate_returns_scores_in_range(self, tmp_path):
        evaluator = self._make_evaluator(tmp_path)

        with patch.object(evaluator, "_judge", return_value=(0.85, "All claims supported.")):
            result = evaluator.evaluate(
                query="What is Python?",
                answer="Python is a programming language [SOURCE 1].",
                context_passages=["Python is a high-level programming language."],
            )

        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.answer_relevancy <= 1.0

    def test_evaluate_persists_to_jsonl(self, tmp_path):
        evaluator = self._make_evaluator(tmp_path)

        with patch.object(evaluator, "_judge", return_value=(0.9, "Good.")):
            evaluator.evaluate(
                query="q", answer="a", context_passages=["context"]
            )

        log_file = tmp_path / "ragas" / "evals.jsonl"
        assert log_file.exists()
        data = json.loads(log_file.read_text().splitlines()[0])
        assert "faithfulness" in data
        assert "answer_relevancy" in data

    def test_summary_with_no_history(self, tmp_path):
        evaluator = self._make_evaluator(tmp_path)
        summary = evaluator.summary()
        assert summary["count"] == 0
        assert summary["mean_faithfulness"] is None

    def test_summary_with_history(self, tmp_path):
        evaluator = self._make_evaluator(tmp_path)

        with patch.object(evaluator, "_judge", side_effect=[
            (0.8, "r1"), (0.9, "r2"),   # first eval: faith=0.8, rel=0.9
            (0.6, "r3"), (0.7, "r4"),   # second eval: faith=0.6, rel=0.7
        ]):
            evaluator.evaluate("q1", "a1", ["c1"])
            evaluator.evaluate("q2", "a2", ["c2"])

        summary = evaluator.summary()
        assert summary["count"] == 2
        assert abs(summary["mean_faithfulness"] - 0.7) < 0.01
        assert abs(summary["mean_relevancy"]    - 0.8) < 0.01

    def test_judge_handles_malformed_response(self, tmp_path):
        evaluator = self._make_evaluator(tmp_path)
        evaluator._client = MagicMock()
        evaluator._client.chat.completions.create.return_value.choices[0].message.content = (
            "NOT VALID JSON"
        )
        score, reason = evaluator._judge("any prompt")
        assert score == 0.0
        assert "error" in reason


# ── Monitoring API endpoints ──────────────────────────────────────────────────

class TestMonitoringEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        from api.main import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    def test_metrics_endpoint_returns_200(self, client):
        resp = client.get("/monitoring/metrics")
        assert resp.status_code == 200
        assert "ragflow_requests_total" in resp.text

    def test_feedback_endpoint_accepts_thumbs_up(self, client):
        resp = client.post("/monitoring/feedback", json={
            "query": "test q",
            "answer": "test a",
            "rating": "thumbs_up",
        })
        assert resp.status_code == 204

    def test_feedback_rejects_invalid_rating(self, client):
        resp = client.post("/monitoring/feedback", json={
            "query": "test q",
            "answer": "test a",
            "rating": "neutral",   # not a valid Literal
        })
        assert resp.status_code == 422

    def test_feedback_summary_endpoint(self, client):
        data = client.get("/monitoring/feedback/summary").json()
        assert "total" in data
        assert "satisfaction_rate" in data

    def test_eval_summary_endpoint(self, client):
        data = client.get("/monitoring/eval/summary").json()
        assert "count" in data
