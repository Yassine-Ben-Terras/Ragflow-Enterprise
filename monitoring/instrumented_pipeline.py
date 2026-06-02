"""
monitoring/instrumented_pipeline.py
Wraps the Phase-3 RAGPipeline and records Prometheus metrics at each stage.

Replaces the plain RAGPipeline in AppState for Phase 5 onwards.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List

from monitoring.metrics import (
    CANDIDATES_RETRIEVED,
    CONTEXT_CHARS,
    LLM_LATENCY,

    REQUESTS_TOTAL,
    REQUEST_LATENCY,
    RETRIEVAL_LATENCY,
    RERANK_LATENCY,
)
from rag.pipeline import RAGPipeline, RAGResponse
from rag.prompt.prompt_builder import PromptBuilder
from rag.reranker.cross_encoder_reranker import CrossEncoderReranker
from rag.retriever.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class InstrumentedPipeline:
    """
    Drop-in replacement for RAGPipeline that records latency and
    count metrics for every pipeline stage.

    Args: same as RAGPipeline.
    """

    def __init__(
        self,
        hybrid_top_k: int = 20,
        rerank_top_k: int = 5,
        data_dir: str = "data",
        openai_api_key: str = "",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        self._retriever    = HybridRetriever(top_k=hybrid_top_k, data_dir=data_dir)
        self._reranker     = CrossEncoderReranker(top_k=rerank_top_k)
        self._prompt_builder = PromptBuilder()
        self._inner        = RAGPipeline(
            hybrid_top_k=hybrid_top_k,
            rerank_top_k=rerank_top_k,
            data_dir=data_dir,
            openai_api_key=openai_api_key,
            llm_model=llm_model,
            temperature=temperature,
        )

    def query(self, question: str) -> RAGResponse:
        t_start = time.perf_counter()
        status  = "success"

        try:
            # ── Stage 1: Hybrid retrieval ────────────────────────────────
            t0 = time.perf_counter()
            candidates = self._retriever.retrieve(question)
            RETRIEVAL_LATENCY.observe(time.perf_counter() - t0)
            CANDIDATES_RETRIEVED.observe(len(candidates))

            if not candidates:
                REQUESTS_TOTAL.labels(status="success").inc()
                return RAGResponse(
                    query=question,
                    answer="I could not find any relevant information in the knowledge base.",
                    citations=[],
                    model=self._inner._llm_model,
                )

            # ── Stage 2: Cross-encoder reranking ─────────────────────────
            t0 = time.perf_counter()
            reranked = self._reranker.rerank(question, candidates)
            RERANK_LATENCY.observe(time.perf_counter() - t0)

            # ── Stage 3: Prompt building ──────────────────────────────────
            prompt = self._prompt_builder.build(question, reranked)
            CONTEXT_CHARS.observe(len(prompt.user_message))

            # ── Stage 4: LLM call ─────────────────────────────────────────
            t0 = time.perf_counter()
            answer = self._inner._call_llm(prompt)
            LLM_LATENCY.observe(time.perf_counter() - t0)

            return RAGResponse(
                query=question,
                answer=answer,
                citations=prompt.citations,
                model=self._inner._llm_model,
                retrieval_scores=[r.score for r in reranked],
                metadata={
                    "num_candidates": len(candidates),
                    "num_reranked":   len(reranked),
                },
            )

        except Exception:
            status = "error"
            logger.exception("InstrumentedPipeline error")
            raise

        finally:
            REQUEST_LATENCY.observe(time.perf_counter() - t_start)
            REQUESTS_TOTAL.labels(status=status).inc()
