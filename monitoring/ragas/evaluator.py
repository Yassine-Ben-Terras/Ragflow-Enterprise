"""
monitoring/ragas/evaluator.py
RAGAs-style evaluation of RAG pipeline responses.

Implements two core metrics without requiring the full ragas library:
  - Faithfulness   : Are claims in the answer supported by the retrieved context?
  - Answer Relevancy : Does the answer address the question?

Both metrics call the LLM as a judge, following the RAGAs paper pattern.
Results are stored locally as JSONL and pushed to Prometheus histograms.

Reference: "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
           Es et al., 2023  (arXiv:2309.15217)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from monitoring.metrics import RAGAS_FAITHFULNESS, RAGAS_RELEVANCY

logger = logging.getLogger(__name__)

_FAITHFULNESS_PROMPT = """\
You are an evaluation assistant. Given a question, an answer, and a set of \
source passages, determine whether every factual claim in the answer is \
supported by the source passages.

Score from 0.0 (no claims supported) to 1.0 (all claims fully supported).
Respond ONLY with a JSON object: {{"score": <float>, "reason": "<one sentence>"}}

Question: {question}
Answer: {answer}
Sources:
{sources}
"""

_RELEVANCY_PROMPT = """\
You are an evaluation assistant. Given a question and an answer, determine \
how well the answer addresses the question.

Score from 0.0 (completely off-topic) to 1.0 (perfectly answers the question).
Respond ONLY with a JSON object: {{"score": <float>, "reason": "<one sentence>"}}

Question: {question}
Answer: {answer}
"""


@dataclass
class EvalResult:
    query: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    faithfulness_reason: str = ""
    relevancy_reason: str = ""
    evaluated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    sources_used: int = 0


class RAGAsEvaluator:
    """
    Args:
        api_key:      OpenAI API key.
        model:        Judge LLM model.
        results_dir:  Directory for persisting evaluation JSONL.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        results_dir: str = "data/ragas",
    ) -> None:
        self._client = OpenAI(api_key=api_key or None)
        self._model  = model
        self._results_path = Path(results_dir)
        self._results_path.mkdir(parents=True, exist_ok=True)

    # ── LLM judge calls ──────────────────────────────────────────────────────

    def _judge(self, prompt: str) -> tuple[float, str]:
        """Call the LLM judge and parse the JSON score response."""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = raw.strip("```json").strip("```").strip()
            parsed = json.loads(raw)
            score  = float(parsed.get("score", 0.0))
            reason = str(parsed.get("reason", ""))
            return max(0.0, min(1.0, score)), reason
        except Exception as exc:
            logger.warning("RAGAs judge call failed: %s", exc)
            return 0.0, f"evaluation error: {exc}"

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        answer: str,
        context_passages: List[str],
    ) -> EvalResult:
        """
        Evaluate a single RAG response.

        Args:
            query:            Original user question.
            answer:           Generated answer.
            context_passages: List of retrieved passage texts.

        Returns:
            EvalResult with faithfulness and answer_relevancy scores.
        """
        sources_block = "\n---\n".join(
            f"[{i+1}] {p[:600]}" for i, p in enumerate(context_passages)
        )

        # ── Faithfulness ──────────────────────────────────────────────────
        faith_prompt = _FAITHFULNESS_PROMPT.format(
            question=query,
            answer=answer,
            sources=sources_block,
        )
        faithfulness, faith_reason = self._judge(faith_prompt)

        # ── Answer Relevancy ──────────────────────────────────────────────
        rel_prompt = _RELEVANCY_PROMPT.format(question=query, answer=answer)
        relevancy, rel_reason = self._judge(rel_prompt)

        result = EvalResult(
            query=query,
            answer=answer,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            faithfulness_reason=faith_reason,
            relevancy_reason=rel_reason,
            sources_used=len(context_passages),
        )

        # ── Push to Prometheus ────────────────────────────────────────────
        RAGAS_FAITHFULNESS.observe(faithfulness)
        RAGAS_RELEVANCY.observe(relevancy)

        # ── Persist locally ───────────────────────────────────────────────
        self._persist(result)

        logger.info(
            "RAGAs eval — faithfulness=%.3f  relevancy=%.3f  query=%r",
            faithfulness, relevancy, query[:60],
        )
        return result

    def _persist(self, result: EvalResult) -> None:
        log_file = self._results_path / "evals.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    def load_history(self) -> List[EvalResult]:
        """Load all persisted evaluation results."""
        log_file = self._results_path / "evals.jsonl"
        if not log_file.exists():
            return []
        results = []
        for line in log_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(EvalResult(**json.loads(line)))
        return results

    def summary(self) -> dict:
        """Compute mean scores across all stored evaluations."""
        history = self.load_history()
        if not history:
            return {"count": 0, "mean_faithfulness": None, "mean_relevancy": None}
        n = len(history)
        return {
            "count": n,
            "mean_faithfulness": sum(r.faithfulness for r in history) / n,
            "mean_relevancy":    sum(r.answer_relevancy for r in history) / n,
        }
