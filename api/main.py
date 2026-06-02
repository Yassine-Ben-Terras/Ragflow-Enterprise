"""
api/main.py
FastAPI application entry point.

Features:
  - /chat/stream   : SSE streaming endpoint (Phase 4 core)
  - /chat          : Standard JSON endpoint
  - /health        : Liveness probe
  - /sources       : List indexed sources
  - CORS configured for local Streamlit dev (localhost:8501)
  - Singleton RAGPipeline loaded once at startup via lifespan
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, health, monitoring, sources
from api.state import AppState

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load heavy resources once at startup, clean up on shutdown."""
    logger.info("Starting ragflow-enterprise API …")
    AppState.init()
    logger.info("RAG pipeline ready.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ragflow-enterprise",
        description="Production-grade RAG API with SSE streaming.",
        version="0.4.0",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",   # Streamlit dev
            "http://localhost:3000",   # Next.js dev (future)
            "http://localhost:8000",   # self
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router,   prefix="/chat",   tags=["Chat"])
    app.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])
    app.include_router(sources.router, prefix="/sources", tags=["Sources"])

    return app


app = create_app()
