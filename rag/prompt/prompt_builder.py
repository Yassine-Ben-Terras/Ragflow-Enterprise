"""
rag/prompt/prompt_builder.py
Builds the final LLM prompt from the re-ranked context passages.

Design decisions:
  - Each passage is labelled [SOURCE N] so the LLM can cite by number.
  - The system prompt instructs the model to always include citations in its answer.
  - A structured Citation dataclass is returned alongside the rendered prompt
    so the API layer can attach source metadata to the response.
  - Context window budget: passages are trimmed to fit max_context_chars.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import List, Optional

from embeddings.stores.base import SearchResult

_DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, precise assistant. Answer the user's question using ONLY the \
provided source passages. Every factual claim in your answer MUST include a \
citation in the format [SOURCE N]. If the answer cannot be found in the passages, \
say so clearly. Do not invent information."""

_MAX_CONTEXT_CHARS = 12_000   # ~3 000 tokens at 4 chars/token — safe for most models


@dataclass
class Citation:
    """Metadata for one source passage used in the answer."""
    index: int            # [SOURCE N] number
    chunk_id: str
    doc_title: str
    doc_source: str
    text_snippet: str     # first 200 chars of the passage
    url: Optional[str]
    file_path: Optional[str]
    score: float


@dataclass
class BuiltPrompt:
    system_prompt: str
    user_message: str        # question + context block
    citations: List[Citation]


class PromptBuilder:
    """
    Args:
        system_prompt:     Override the default system instruction.
        max_context_chars: Maximum total characters for all context passages.
        snippet_length:    Characters to include per passage in the context block.
    """

    def __init__(
        self,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_context_chars: int = _MAX_CONTEXT_CHARS,
        snippet_length: int = 800,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars
        self.snippet_length = snippet_length

    def build(self, query: str, results: List[SearchResult]) -> BuiltPrompt:
        """
        Construct the prompt and citation list from re-ranked results.

        Passages are included in order until max_context_chars is reached.
        """
        context_lines: List[str] = []
        citations: List[Citation] = []
        total_chars = 0

        for i, result in enumerate(results, start=1):
            snippet = result.text[: self.snippet_length].strip()
            block = f"[SOURCE {i}]\nTitle: {result.doc_title}\n{snippet}"

            if total_chars + len(block) > self.max_context_chars:
                break

            context_lines.append(block)
            total_chars += len(block)
            citations.append(
                Citation(
                    index=i,
                    chunk_id=result.chunk_id,
                    doc_title=result.doc_title,
                    doc_source=result.doc_source,
                    text_snippet=snippet[:200],
                    url=result.url,
                    file_path=result.file_path,
                    score=result.score,
                )
            )

        context_block = "\n\n---\n\n".join(context_lines)

        user_message = textwrap.dedent(f"""\
            ### Source Passages
            {context_block}

            ---

            ### Question
            {query}

            ### Answer (cite every fact as [SOURCE N])
        """).strip()

        return BuiltPrompt(
            system_prompt=self.system_prompt,
            user_message=user_message,
            citations=citations,
        )
