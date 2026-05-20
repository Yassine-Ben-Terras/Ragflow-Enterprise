"""
rag/pipeline.py
Phase-3 RAG pipeline orchestrator.

Flow:
  query
    → HybridRetriever  (dense + BM25 → RRF fusion)
    → CrossEncoderReranker
    → PromptBuilder    (context block with [SOURCE N] labels)
    → LLM call         (OpenAI chat completion)
    → RAGResponse      (answer + citations)

Usage:
    python -m rag.pipeline --query "What is the deployment process?"
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import settings
from rag.prompt.prompt_builder import BuiltPrompt, Citation, PromptBuilder
from rag.reranker.cross_encoder_reranker import CrossEncoderReranker
from rag.retriever.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete response returned by the RAG pipeline."""
    query: str
    answer: str
    citations: List[Citation]
    model: str
    retrieval_scores: List[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def pretty(self) -> str:
        """Human-readable string for CLI use."""
        lines = [
            f"Q: {self.query}",
            "",
            f"A: {self.answer}",
            "",
            "── Sources ──────────────────────────────────────",
        ]
        for c in self.citations:
            ref = c.url or c.file_path or c.doc_source
            lines.append(f"[SOURCE {c.index}] {c.doc_title}  ({ref})  score={c.score:.3f}")
        return "\n".join(lines)


class RAGPipeline:
    """
    Args:
        hybrid_top_k:   Candidates from hybrid retrieval before reranking.
        rerank_top_k:   Final passages after cross-encoder reranking.
        data_dir:       Local storage root (for BM25).
        openai_api_key: OpenAI key (falls back to OPENAI_API_KEY env var).
        llm_model:      Chat model for answer generation.
        temperature:    LLM temperature.
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
        self._retriever = HybridRetriever(
            top_k=hybrid_top_k,
            data_dir=data_dir,
        )
        self._reranker = CrossEncoderReranker(top_k=rerank_top_k)
        self._prompt_builder = PromptBuilder()
        self._llm_model = llm_model
        self._temperature = temperature
        self._api_key = openai_api_key or settings.openai_api_key
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI(api_key=self._api_key or None)
        return self._llm_client

    def _call_llm(self, prompt: BuiltPrompt) -> str:
        client = self._get_llm_client()
        response = client.chat.completions.create(
            model=self._llm_model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user",   "content": prompt.user_message},
            ],
        )
        return response.choices[0].message.content.strip()

    # ── public API ───────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """
        Run the full RAG pipeline for a single question.

        Args:
            question: Natural language query.

        Returns:
            RAGResponse with answer text and source citations.
        """
        logger.info("RAG query: %r", question)

        # 1 — Hybrid retrieval
        candidates = self._retriever.retrieve(question)
        if not candidates:
            return RAGResponse(
                query=question,
                answer="I could not find any relevant information in the knowledge base.",
                citations=[],
                model=self._llm_model,
            )

        # 2 — Cross-encoder reranking
        reranked = self._reranker.rerank(question, candidates)

        # 3 — Build prompt with citations
        prompt = self._prompt_builder.build(question, reranked)

        # 4 — LLM call
        answer = self._call_llm(prompt)

        logger.info(
            "RAG complete: %d candidates → %d reranked → answer (%d chars)",
            len(candidates), len(reranked), len(answer),
        )

        return RAGResponse(
            query=question,
            answer=answer,
            citations=prompt.citations,
            model=self._llm_model,
            retrieval_scores=[r.score for r in reranked],
            metadata={
                "num_candidates": len(candidates),
                "num_reranked": len(reranked),
            },
        )


# ── CLI entry point ──────────────────────────────────────────────────────────

def _cli() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run a RAG query.")
    parser.add_argument("--query", "-q", required=True, help="Question to answer.")
    parser.add_argument("--hybrid-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        hybrid_top_k=args.hybrid_top_k,
        rerank_top_k=args.rerank_top_k,
        llm_model=args.model,
    )
    response = pipeline.query(args.query)
    print(response.pretty())


if __name__ == "__main__":
    _cli()
