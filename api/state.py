"""
api/state.py
Application-level singleton state.
Centralises the RAGPipeline so all routers share one instance.
"""

from __future__ import annotations

import logging

from config.settings import settings
from monitoring.instrumented_pipeline import InstrumentedPipeline

logger = logging.getLogger(__name__)


class AppState:
    pipeline: InstrumentedPipeline | None = None

    @classmethod
    def init(cls) -> None:
        cls.pipeline = InstrumentedPipeline(
            hybrid_top_k=settings.hybrid_top_k,
            rerank_top_k=settings.rerank_top_k,
            data_dir=settings.data_dir,
            openai_api_key=settings.openai_api_key,
            llm_model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
        logger.info(
            "RAGPipeline initialised (llm=%s, hybrid_top_k=%d, rerank_top_k=%d)",
            settings.llm_model,
            settings.hybrid_top_k,
            settings.rerank_top_k,
        )

    @classmethod
    def get_pipeline(cls) -> RAGPipeline:
        if cls.pipeline is None:
            cls.init()
        return cls.pipeline
