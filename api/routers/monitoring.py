"""
api/routers/monitoring.py
Exposes monitoring endpoints:

  GET  /metrics          → Prometheus text exposition format
  POST /feedback         → Submit user rating for a response
  POST /eval             → Run RAGAs evaluation on a single response
  GET  /eval/summary     → Aggregated RAGAs scores
  GET  /feedback/summary → Feedback stats (satisfaction rate, counts)
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from config.settings import settings
from monitoring.feedback.feedback_store import FeedbackEntry, FeedbackStore
from monitoring.ragas.evaluator import RAGAsEvaluator

logger = logging.getLogger(__name__)
router = APIRouter()

# Singletons
_feedback_store = FeedbackStore(data_dir=settings.data_dir)
_evaluator      = RAGAsEvaluator(
    api_key=settings.openai_api_key,
    model=settings.llm_model,
    results_dir=f"{settings.data_dir}/ragas",
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: Literal["thumbs_up", "thumbs_down"]
    comment: Optional[str] = None
    session_id: Optional[str] = None
    chunk_ids: List[str] = []


class EvalRequest(BaseModel):
    query: str
    answer: str
    context_passages: List[str]


class EvalResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    faithfulness_reason: str
    relevancy_reason: str
    sources_used: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> PlainTextResponse:
    """Prometheus scrape endpoint — returns all registered metrics."""
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@router.post("/feedback", status_code=204)
async def submit_feedback(request: FeedbackRequest) -> None:
    """Store user rating and optional comment for a RAG response."""
    entry = FeedbackEntry(
        query=request.query,
        answer=request.answer,
        rating=request.rating,
        comment=request.comment,
        session_id=request.session_id,
        chunk_ids=request.chunk_ids,
    )
    _feedback_store.submit(entry)


@router.get("/feedback/summary")
async def feedback_summary() -> dict:
    """Return aggregated feedback statistics."""
    return _feedback_store.summary()


@router.post("/eval", response_model=EvalResponse)
async def evaluate_response(request: EvalRequest) -> EvalResponse:
    """Run RAGAs evaluation (faithfulness + relevancy) on a single response."""
    result = _evaluator.evaluate(
        query=request.query,
        answer=request.answer,
        context_passages=request.context_passages,
    )
    return EvalResponse(
        faithfulness=result.faithfulness,
        answer_relevancy=result.answer_relevancy,
        faithfulness_reason=result.faithfulness_reason,
        relevancy_reason=result.relevancy_reason,
        sources_used=result.sources_used,
    )


@router.get("/eval/summary")
async def eval_summary() -> dict:
    """Return mean RAGAs scores across all persisted evaluations."""
    return _evaluator.summary()
