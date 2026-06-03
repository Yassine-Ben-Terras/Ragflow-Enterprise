"""
ui/streamlit/app.py
Streamlit chat interface for ragflow-enterprise.

Features:
  - Real-time streaming answer via SSE
  - Expandable source citations panel
  - Conversation history (session state)
  - Sidebar: API status, indexed sources stats

Run:
    streamlit run ui/streamlit/app.py
"""

from __future__ import annotations

import json
import time
from typing import Generator

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
STREAM_ENDPOINT = f"{API_BASE}/chat/stream"
CHAT_ENDPOINT   = f"{API_BASE}/chat"
HEALTH_ENDPOINT = f"{API_BASE}/health"
SOURCES_ENDPOINT = f"{API_BASE}/sources"

st.set_page_config(
    page_title="ragflow-enterprise",
    page_icon="🔍",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": "user"|"assistant", "content": str, "citations": list}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ ragflow-enterprise")
    st.divider()

    # API health
    try:
        health = requests.get(HEALTH_ENDPOINT, timeout=2).json()
        ready = health.get("pipeline_ready", False)
        st.success("API online ✅") if ready else st.warning("API online — pipeline loading…")
    except Exception:
        st.error("API offline ❌")
        ready = False

    st.divider()

    # Indexed sources
    st.subheader("📚 Indexed sources")
    try:
        sources_data = requests.get(SOURCES_ENDPOINT, timeout=5).json()
        st.metric("Total chunks", sources_data.get("total_chunks", 0))
        for s in sources_data.get("sources", []):
            st.markdown(f"**{s['source']}** — {s['doc_count']} docs · {s['total_chunks']} chunks")
    except Exception:
        st.caption("Could not load source stats.")

    st.divider()

    # Settings
    st.subheader("🎛️ Settings")
    use_stream = st.toggle("Streaming mode (SSE)", value=True)
    hybrid_top_k = st.slider("Hybrid top-k", 5, 50, 20)
    rerank_top_k = st.slider("Rerank top-k", 1, 10, 5)

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────

st.title("🔍 ragflow-enterprise")
st.caption("Ask questions about your indexed documents. Every answer includes source citations.")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} source(s)"):
                for c in msg["citations"]:
                    ref = c.get("url") or c.get("file_path") or c.get("doc_source", "")
                    st.markdown(
                        f"**[SOURCE {c['index']}] {c['doc_title']}** — `{c['doc_source']}`  \n"
                        f"score: `{c['score']:.3f}` · [{ref}]({ref})  \n"
                        f"> {c['text_snippet'][:200]}…"
                    )

# ── Streaming helper ──────────────────────────────────────────────────────────

def _stream_sse(query: str, hybrid_top_k: int, rerank_top_k: int) -> Generator[tuple, None, None]:
    """
    Yields (token_so_far, citations_so_far) tuples as SSE events arrive.
    Final yield has the complete answer and all citations.
    """
    payload = {
        "query": query,
        "hybrid_top_k": hybrid_top_k,
        "rerank_top_k": rerank_top_k,
    }
    accumulated = ""
    citations = []

    with requests.post(STREAM_ENDPOINT, json=payload, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:].strip())
            event_type = data.get("type")

            if event_type == "token":
                accumulated += data.get("content", "")
                yield accumulated, citations

            elif event_type == "citation":
                citations.append(json.loads(data.get("content", "{}")))

            elif event_type in ("done", "error"):
                yield accumulated, citations
                break


# ── Input handling ────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents…", disabled=not ready):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        citations_list = []

        if use_stream:
            try:
                for partial_answer, partial_citations in _stream_sse(prompt, hybrid_top_k, rerank_top_k):
                    answer_placeholder.markdown(partial_answer + "▌")
                    citations_list = partial_citations
                answer_placeholder.markdown(partial_answer)
            except Exception as exc:
                answer_placeholder.error(f"Streaming error: {exc}")
                partial_answer = f"Error: {exc}"
        else:
            with st.spinner("Thinking…"):
                try:
                    resp = requests.post(
                        CHAT_ENDPOINT,
                        json={"query": prompt, "hybrid_top_k": hybrid_top_k, "rerank_top_k": rerank_top_k},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    partial_answer = data["answer"]
                    citations_list = data.get("citations", [])
                    answer_placeholder.markdown(partial_answer)
                except Exception as exc:
                    answer_placeholder.error(f"Error: {exc}")
                    partial_answer = f"Error: {exc}"

        if citations_list:
            with st.expander(f"📎 {len(citations_list)} source(s)"):
                for c in citations_list:
                    ref = c.get("url") or c.get("file_path") or c.get("doc_source", "")
                    st.markdown(
                        f"**[SOURCE {c['index']}] {c['doc_title']}** — `{c['doc_source']}`  \n"
                        f"score: `{c['score']:.3f}` · [{ref}]({ref})  \n"
                        f"> {c['text_snippet'][:200]}…"
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": partial_answer,
        "citations": citations_list,
    })
