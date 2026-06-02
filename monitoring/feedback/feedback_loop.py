"""
monitoring/feedback/feedback_loop.py
Continuous improvement feedback loop.

Workflow:
  1. Load all thumbs_down feedback entries.
  2. For each unseen entry, run RAGAs evaluation (faithfulness + relevancy).
  3. Log entries whose faithfulness < threshold as "retrieval failures".
  4. Log entries whose relevancy < threshold as "prompt/LLM failures".
  5. Write a structured improvement report to data/feedback/improvement_report.jsonl.

This report can be reviewed by a human or used to:
  - Tune chunking parameters (if faithfulness is low → context is noisy)
  - Tune the prompt template (if relevancy is low → LLM ignores sources)
  - Flag documents for re-ingestion

Run manually or schedule via Airflow:
    python -m monitoring.feedback.feedback_loop
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List

from config.settings import settings
from monitoring.feedback.feedback_store import FeedbackEntry, FeedbackStore
from monitoring.ragas.evaluator import RAGAsEvaluator

logger = logging.getLogger(__name__)

_FAITHFULNESS_THRESHOLD = 0.6
_RELEVANCY_THRESHOLD    = 0.6


def run_feedback_loop(data_dir: str = "data") -> dict:
    """
    Process all unreviewed negative feedback and produce an improvement report.

    Returns a summary dict.
    """
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    store     = FeedbackStore(data_dir=data_dir)
    evaluator = RAGAsEvaluator(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        results_dir=f"{data_dir}/ragas",
    )

    report_path = Path(data_dir) / "feedback" / "improvement_report.jsonl"

    # Load already-reviewed entries to skip re-evaluation
    reviewed: set[str] = set()
    if report_path.exists():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                reviewed.add(entry.get("query", "") + entry.get("submitted_at", ""))

    negative: List[FeedbackEntry] = store.load_negative()
    new_entries = [
        e for e in negative
        if (e.query + e.submitted_at) not in reviewed
    ]

    logger.info(
        "Feedback loop: %d negative entries total, %d new to evaluate.",
        len(negative), len(new_entries),
    )

    retrieval_failures = 0
    prompt_failures    = 0
    processed          = 0

    for entry in new_entries:
        # RAGAs evaluation — use the stored answer as context proxy
        # In production, store the retrieved passages alongside feedback
        eval_result = evaluator.evaluate(
            query=entry.query,
            answer=entry.answer,
            context_passages=[entry.answer],  # fallback: use answer as context
        )

        failure_type = []
        if eval_result.faithfulness < _FAITHFULNESS_THRESHOLD:
            failure_type.append("retrieval")
            retrieval_failures += 1
        if eval_result.answer_relevancy < _RELEVANCY_THRESHOLD:
            failure_type.append("prompt_or_llm")
            prompt_failures += 1

        report_entry = {
            "query":             entry.query,
            "rating":            entry.rating,
            "comment":           entry.comment,
            "submitted_at":      entry.submitted_at,
            "faithfulness":      eval_result.faithfulness,
            "answer_relevancy":  eval_result.answer_relevancy,
            "failure_type":      failure_type,
            "processed_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report_entry, ensure_ascii=False) + "\n")

        processed += 1
        logger.info(
            "Processed [%s] faithfulness=%.3f relevancy=%.3f failures=%s",
            entry.rating,
            eval_result.faithfulness,
            eval_result.answer_relevancy,
            failure_type or "none",
        )

    summary = {
        "processed":          processed,
        "retrieval_failures": retrieval_failures,
        "prompt_failures":    prompt_failures,
        "report_path":        str(report_path),
    }
    logger.info("Feedback loop complete: %s", summary)
    return summary


if __name__ == "__main__":
    run_feedback_loop()
