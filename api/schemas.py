"""
api/schemas.py
Pydantic v2 request / response models for the API.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question.")
    hybrid_top_k: Optional[int] = Field(None, ge=1, le=100, description="Override hybrid retrieval top-k.")
    rerank_top_k: Optional[int] = Field(None, ge=1, le=20,  description="Override reranker top-k.")
    stream: bool = Field(False, description="Use SSE streaming (ignored for /chat/stream endpoint).")


# ── Response ──────────────────────────────────────────────────────────────────

class CitationSchema(BaseModel):
    index: int
    chunk_id: str
    doc_title: str
    doc_source: str
    text_snippet: str
    url: Optional[str]
    file_path: Optional[str]
    score: float


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationSchema]
    model: str
    metadata: dict = Field(default_factory=dict)


# ── SSE token schema (for documentation only — SSE is plain text) ─────────────

class SSETokenEvent(BaseModel):
    """Describes the shape of each SSE data line during streaming."""
    type: str = Field(description="'token' | 'citation' | 'done' | 'error'")
    content: str = Field(description="Token text, JSON citation, or error message.")


# ── Sources ───────────────────────────────────────────────────────────────────

class SourceSummary(BaseModel):
    source: str
    total_chunks: int
    doc_count: int


class SourcesResponse(BaseModel):
    sources: List[SourceSummary]
    total_chunks: int
