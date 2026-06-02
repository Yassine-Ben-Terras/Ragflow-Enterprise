"""
monitoring/metrics.py
Prometheus metrics for the RAG pipeline.

Metrics exposed at GET /metrics (added to FastAPI in Phase 5):

  ragflow_requests_total          Counter   — total queries by status
  ragflow_request_latency_seconds Histogram — end-to-end query latency
  ragflow_retrieval_latency_seconds Histogram — hybrid retrieval latency
  ragflow_rerank_latency_seconds  Histogram — cross-encoder rerank latency
  ragflow_llm_latency_seconds     Histogram — LLM call latency
  ragflow_candidates_retrieved    Histogram — number of hybrid candidates
  ragflow_context_chars           Histogram — chars sent to LLM
  ragflow_feedback_total          Counter   — user feedback by rating

All metrics use the label `pipeline_stage` where applicable.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, CollectorRegistry, REGISTRY

# ── Registry ─────────────────────────────────────────────────────────────────
# Use the default global registry so /metrics picks everything up automatically.

# ── Counters ──────────────────────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    "ragflow_requests_total",
    "Total number of RAG queries.",
    ["status"],          # "success" | "error"
)

FEEDBACK_TOTAL = Counter(
    "ragflow_feedback_total",
    "User feedback submitted.",
    ["rating"],          # "thumbs_up" | "thumbs_down"
)

# ── Histograms ────────────────────────────────────────────────────────────────

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

REQUEST_LATENCY = Histogram(
    "ragflow_request_latency_seconds",
    "End-to-end query latency.",
    buckets=_LATENCY_BUCKETS,
)

RETRIEVAL_LATENCY = Histogram(
    "ragflow_retrieval_latency_seconds",
    "Hybrid retrieval latency.",
    buckets=_LATENCY_BUCKETS,
)

RERANK_LATENCY = Histogram(
    "ragflow_rerank_latency_seconds",
    "Cross-encoder rerank latency.",
    buckets=_LATENCY_BUCKETS,
)

LLM_LATENCY = Histogram(
    "ragflow_llm_latency_seconds",
    "LLM call latency.",
    buckets=_LATENCY_BUCKETS,
)

CANDIDATES_RETRIEVED = Histogram(
    "ragflow_candidates_retrieved",
    "Number of hybrid retrieval candidates before reranking.",
    buckets=(1, 5, 10, 15, 20, 30, 50),
)

CONTEXT_CHARS = Histogram(
    "ragflow_context_chars",
    "Total characters of context sent to the LLM.",
    buckets=(500, 1000, 2000, 4000, 8000, 12000, 16000),
)

RAGAS_FAITHFULNESS = Histogram(
    "ragflow_ragas_faithfulness",
    "RAGAs faithfulness score (0–1).",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

RAGAS_RELEVANCY = Histogram(
    "ragflow_ragas_answer_relevancy",
    "RAGAs answer relevancy score (0–1).",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
