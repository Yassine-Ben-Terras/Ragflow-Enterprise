"""
monitoring/feedback/feedback_store.py
Stores user feedback (thumbs up/down + optional comment) as local JSONL.

Feedback is:
  1. Persisted to data/feedback/feedback.jsonl
  2. Counted in Prometheus (ragflow_feedback_total)
  3. Optionally used to trigger re-evaluation via the feedback loop

The feedback loop identifies low-quality responses (thumbs_down) and
schedules them for RAGAs evaluation, enabling continuous improvement.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from monitoring.metrics import FEEDBACK_TOTAL

logger = logging.getLogger(__name__)

Rating = Literal["thumbs_up", "thumbs_down"]


@dataclass
class FeedbackEntry:
    query: str
    answer: str
    rating: Rating
    comment: Optional[str] = None
    session_id: Optional[str] = None
    submitted_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    chunk_ids: List[str] = field(default_factory=list)


class FeedbackStore:
    """
    Args:
        data_dir: Root of local storage — feedback written to <data_dir>/feedback/.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self._feedback_dir  = Path(data_dir) / "feedback"
        self._feedback_dir.mkdir(parents=True, exist_ok=True)
        self._feedback_file = self._feedback_dir / "feedback.jsonl"

    # ── write ─────────────────────────────────────────────────────────────────

    def submit(self, entry: FeedbackEntry) -> None:
        """Persist feedback and increment Prometheus counter."""
        with self._feedback_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

        FEEDBACK_TOTAL.labels(rating=entry.rating).inc()
        logger.info(
            "Feedback [%s] for query: %r", entry.rating, entry.query[:60]
        )

    # ── read ──────────────────────────────────────────────────────────────────

    def load_all(self) -> List[FeedbackEntry]:
        if not self._feedback_file.exists():
            return []
        entries = []
        for line in self._feedback_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(FeedbackEntry(**json.loads(line)))
        return entries

    def load_negative(self) -> List[FeedbackEntry]:
        """Return only thumbs_down entries — used by the feedback loop."""
        return [e for e in self.load_all() if e.rating == "thumbs_down"]

    def summary(self) -> dict:
        all_entries = self.load_all()
        up   = sum(1 for e in all_entries if e.rating == "thumbs_up")
        down = sum(1 for e in all_entries if e.rating == "thumbs_down")
        return {
            "total": len(all_entries),
            "thumbs_up":   up,
            "thumbs_down": down,
            "satisfaction_rate": round(up / len(all_entries), 3) if all_entries else None,
        }
