"""
api/routers/chat.py
Two endpoints:

  POST /chat          → standard JSON response (RAGResponse)
  POST /chat/stream   → Server-Sent Events (SSE) streaming response

SSE stream protocol
───────────────────
Each SSE message has the format:

    data: <JSON>\n\n

Event types:
  { "type": "token",    "content": "<text fragment>" }
  { "type": "citation", "content": "<CitationSchema JSON>" }
  { "type": "done",     "content": "" }
  { "type": "error",    "content": "<error message>" }

The client assembles tokens into a full answer, then receives citations
once the LLM stream completes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationSchema,
)
from api.state import AppState
from config.settings import settings
from rag.prompt.prompt_builder import BuiltPrompt, PromptBuilder
from rag.reranker.cross_encoder_reranker import CrossEncoderReranker
from rag.retriever.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)
router = APIRouter()


# ── helpers ──────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


def _citation_schemas(citations) -> list[CitationSchema]:
    return [
        CitationSchema(
            index=c.index,
            chunk_id=c.chunk_id,
            doc_title=c.doc_title,
            doc_source=c.doc_source,
            text_snippet=c.text_snippet,
            url=c.url,
            file_path=c.file_path,
            score=c.score,
        )
        for c in citations
    ]


# ── standard JSON endpoint ───────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    pipeline = AppState.get_pipeline()
    try:
        response = await asyncio.to_thread(pipeline.query, request.query)
    except Exception as exc:
        logger.exception("RAG pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        query=response.query,
        answer=response.answer,
        citations=_citation_schemas(response.citations),
        model=response.model,
        metadata=response.metadata,
    )


# ── SSE streaming endpoint ───────────────────────────────────────────────────

@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Streams the LLM answer token-by-token via SSE, then sends citations.
    Uses OpenAI's async streaming API.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # ── Build retrieval + reranking synchronously in a thread ────
            retriever = HybridRetriever(
                top_k=request.hybrid_top_k or settings.hybrid_top_k,
                data_dir=settings.data_dir,
            )
            reranker = CrossEncoderReranker(top_k=request.rerank_top_k or settings.rerank_top_k)

            candidates = await asyncio.to_thread(retriever.retrieve, request.query)

            if not candidates:
                yield _sse({"type": "token", "content": "I could not find relevant information."})
                yield _sse({"type": "done",  "content": ""})
                return

            reranked = await asyncio.to_thread(reranker.rerank, request.query, candidates)

            builder   = PromptBuilder()
            prompt: BuiltPrompt = builder.build(request.query, reranked)

            # ── Stream LLM tokens ────────────────────────────────────────
            async_client = AsyncOpenAI(api_key=settings.openai_api_key or None)

            stream = await async_client.chat.completions.create(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                stream=True,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user",   "content": prompt.user_message},
                ],
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield _sse({"type": "token", "content": delta})

            # ── Send citations after stream completes ────────────────────
            for citation in _citation_schemas(prompt.citations):
                yield _sse({"type": "citation", "content": citation.model_dump_json()})

            yield _sse({"type": "done", "content": ""})

        except Exception as exc:
            logger.exception("SSE stream error")
            yield _sse({"type": "error", "content": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )
