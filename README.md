# ragflow-enterprise

> **Production-grade Retrieval-Augmented Generation (RAG) system** — multi-source document ingestion, semantic + lexical hybrid search, cross-encoder reranking, streaming API, and a full observability stack. Built locally-first, Dockerised for deployment.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Phase 1 — Ingestion](#phase-1--ingestion)
- [Phase 2 — Embeddings](#phase-2--embeddings)
- [Phase 3 — RAG Pipeline](#phase-3--rag-pipeline)
- [Phase 4 — API & UI](#phase-4--api--ui)
- [Phase 5 — Monitoring](#phase-5--monitoring)
- [Configuration Reference](#configuration-reference)
- [Running Tests](#running-tests)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)

---

## Overview

`ragflow-enterprise` is a complete, production-ready RAG pipeline designed to answer natural-language questions over private document collections — PDFs, Confluence wikis, and Git repositories.

**Key design principles:**

- **Locally-first** — every component runs on a laptop with no cloud account required
- **Incrementally built** — five independent phases, each runnable and testable on its own
- **Production-ready** — typed interfaces, error handling, retry logic, Docker, CI-ready tests
- **Observable** — every pipeline stage is instrumented with Prometheus metrics and a Grafana dashboard
- **Continuously improving** — a feedback loop captures user ratings and runs automated RAGAs evaluation to surface retrieval and prompt failures

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ragflow-enterprise                           │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │   Phase 1    │    │   Phase 2    │    │     Phase 3        │    │
│  │  Ingestion   │───▶│  Embeddings  │───▶│   RAG Pipeline     │    │
│  │              │    │              │    │                    │    │
│  │ PDF          │    │ OpenAI /     │    │ HybridRetriever    │    │
│  │ Confluence   │    │ BGE-M3       │    │  Dense (vector)    │    │
│  │ Git repos    │    │              │    │  + BM25 (lexical)  │    │
│  │              │    │ Qdrant /     │    │  → RRF fusion      │    │
│  │ SmartChunker │    │ PGVector     │    │                    │    │
│  │ LocalStorage │    │ (HNSW index) │    │ CrossEncoder       │    │
│  └──────────────┘    └──────────────┘    │ reranking          │    │
│                                          │                    │    │
│                                          │ PromptBuilder      │    │
│                                          │ [SOURCE N] cites   │    │
│                                          └────────┬───────────┘    │
│                                                   │                │
│  ┌──────────────────────────┐    ┌────────────────▼───────────┐   │
│  │        Phase 5           │    │         Phase 4            │   │
│  │       Monitoring         │    │        API + UI            │   │
│  │                          │    │                            │   │
│  │ Prometheus metrics       │    │ FastAPI                    │   │
│  │ Grafana dashboard        │◀───│  POST /chat  (JSON)        │   │
│  │ RAGAs evaluation         │    │  POST /chat/stream (SSE)   │   │
│  │ Feedback loop            │    │                            │   │
│  │ Airflow DAGs             │    │ Streamlit chat UI          │   │
│  └──────────────────────────┘    └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow (query time):**

```
User question
  → HybridRetriever  (dense top-20 + BM25 top-20 → RRF fusion)
  → CrossEncoderReranker  (ms-marco-MiniLM → top-5)
  → PromptBuilder  (context block with [SOURCE N] labels, 12k char budget)
  → OpenAI GPT-4o-mini  (streaming or batch)
  → RAGResponse  { answer, citations[] }
  → User feedback  → RAGAs evaluation  → improvement report
```

---

## Project Structure

```
ragflow-enterprise/
│
├── ingestion/                     # Phase 1 — Document ingestion
│   ├── connectors/
│   │   ├── base.py                  Abstract BaseConnector + Document dataclass
│   │   ├── pdf_connector.py         Local PDF ingestion (pdfplumber)
│   │   ├── confluence.py            Confluence Cloud REST API v2, paginated
│   │   └── git_connector.py         Shallow Git clone, multi-extension walker
│   ├── chunking/
│   │   └── smart_chunker.py         Recursive / sentence / semantic strategies + overlap
│   ├── storage/
│   │   └── local_storage.py         JSON + JSONL filesystem persistence (data/)
│   └── run.py                       Ingestion orchestrator CLI
│
├── embeddings/                    # Phase 2 — Vector embeddings
│   ├── providers/
│   │   ├── base.py                  BaseEmbedder ABC
│   │   ├── openai_embedder.py       OpenAI text-embedding-3-small/large, batched
│   │   └── bge_embedder.py          BAAI/bge-m3 local model (sentence-transformers)
│   ├── stores/
│   │   ├── base.py                  BaseVectorStore ABC + VectorRecord / SearchResult
│   │   ├── qdrant_store.py          Qdrant + HNSW (m=16, ef=200), cosine similarity
│   │   └── pgvector_store.py        PostgreSQL pgvector + HNSW, ON CONFLICT upsert
│   └── pipeline.py                  Embedding pipeline orchestrator CLI
│
├── rag/                           # Phase 3 — RAG pipeline
│   ├── retriever/
│   │   ├── dense_retriever.py       Semantic vector search
│   │   ├── bm25_retriever.py        BM25 sparse retrieval, lazy in-memory index
│   │   └── hybrid_retriever.py      RRF fusion (dense 60% + BM25 40%)
│   ├── reranker/
│   │   └── cross_encoder_reranker.py  ms-marco-MiniLM-L-6-v2, lazy loaded
│   ├── prompt/
│   │   └── prompt_builder.py        [SOURCE N] citation blocks, context budget
│   └── pipeline.py                  Full RAG orchestrator + CLI
│
├── api/                           # Phase 4 — REST API
│   ├── main.py                      FastAPI app factory, lifespan, CORS
│   ├── state.py                     Singleton InstrumentedPipeline
│   ├── schemas.py                   Pydantic v2 request/response models
│   └── routers/
│       ├── health.py                GET  /health
│       ├── chat.py                  POST /chat  +  POST /chat/stream (SSE)
│       ├── sources.py               GET  /sources
│       └── monitoring.py            GET  /monitoring/metrics  +  feedback + eval
│
├── ui/streamlit/
│   └── app.py                     # Streamlit chat UI — SSE streaming, citations
│
├── monitoring/                    # Phase 5 — Observability
│   ├── metrics.py                   Prometheus counters + histograms
│   ├── instrumented_pipeline.py     Per-stage timing wrapper for RAGPipeline
│   ├── ragas/
│   │   └── evaluator.py             LLM-as-judge: faithfulness + answer relevancy
│   ├── feedback/
│   │   ├── feedback_store.py        Thumbs up/down JSONL + Prometheus counter
│   │   └── feedback_loop.py         Batch: negative feedback → RAGAs → report
│   ├── prometheus/
│   │   └── prometheus.yml           Scrape config (15s interval)
│   └── grafana/
│       ├── dashboards/ragflow_overview.json   Pre-built dashboard
│       └── provisioning/            Auto-wired datasource + dashboard
│
├── airflow/dags/
│   ├── ingestion_dag.py           Daily ingestion DAG (02:00 UTC)
│   └── feedback_loop_dag.py       Daily feedback loop DAG (03:00 UTC)
│
├── tests/                         # Pytest test suites (one per phase)
│   ├── ingestion/
│   ├── embeddings/
│   ├── rag/
│   ├── api/
│   └── monitoring/
│
├── config/
│   └── settings.py                Pydantic-settings — all config from .env
│
├── Dockerfile.api                 FastAPI container (python:3.11-slim)
├── Dockerfile.ui                  Streamlit container
├── docker-compose.yml             All 6 services: api, ui, qdrant, postgres, prometheus, grafana
├── requirements.txt
└── .env.example                   All environment variables documented
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | Required |
| Docker + Docker Compose | ≥ 24 | For vector stores, monitoring stack |
| OpenAI API key | — | Required for embeddings (OpenAI provider) and LLM calls |
| Git | any | Required for GitConnector |

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/Yassine-Ben-Terras/Ragflow-Enterprise.git
cd ragflow-enterprise
cp .env.example .env
# → Edit .env: set OPENAI_API_KEY and PDF_SOURCE_DIR at minimum

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start vector store (Qdrant, default)
docker compose up qdrant -d

# 4. Drop your PDFs into data/pdfs/
mkdir -p data/pdfs
cp /path/to/your/documents/*.pdf data/pdfs/

# 5. Ingest documents
python -m ingestion.run

# 6. Generate and index embeddings
python -m embeddings.pipeline

# 7. Ask a question
python -m rag.pipeline --query "What is the deployment process?"

# 8. Start the full API + UI
uvicorn api.main:app --reload --port 8000 &
streamlit run ui/streamlit/app.py
```

Open **http://localhost:8501** to use the chat interface.

---

## Phase 1 — Ingestion

Ingests documents from three source types, splits them into chunks, and persists to the local filesystem.

```bash
# Ingest all configured sources
python -m ingestion.run

# Output layout
data/
├── raw/pdf/<source_id>.json         ← raw document JSON
├── raw/confluence/<source_id>.json
├── raw/git/<source_id>.json
└── chunks/<source>/<source_id>.jsonl ← chunk JSONL, one line per chunk
```

**Connectors:**

| Connector | Config keys | Description |
|---|---|---|
| `PDFConnector` | `PDF_SOURCE_DIR` | Recursively walks a local directory for `.pdf` files |
| `ConfluenceConnector` | `CONFLUENCE_URL`, `CONFLUENCE_API_TOKEN`, `CONFLUENCE_SPACES` | Paginates through Confluence spaces via REST API v2 |
| `GitConnector` | `GIT_REPOS`, `GIT_BRANCH`, `GIT_FILE_EXTENSIONS` | Shallow-clones repos and walks text files |

**Chunking strategies** (set via `CHUNKING_STRATEGY`):

| Strategy | Description |
|---|---|
| `recursive` | LangChain-style recursive character splitting — default, fastest |
| `sentence` | Splits on sentence boundaries, then merges up to `CHUNK_SIZE` |
| `semantic` | Embedding-aware splitting (stub, promoted in future phases) |

---

## Phase 2 — Embeddings

Loads all chunks from local storage, generates vector embeddings, and upserts them into a vector store with HNSW indexing for sub-100ms similarity search.

```bash
python -m embeddings.pipeline
```

**Embedding providers** (set via `EMBEDDING_PROVIDER`):

| Provider | Model | Dimension | Notes |
|---|---|---|---|
| `openai` | `text-embedding-3-small` | 1536 | Default — fast, cheap |
| `openai` | `text-embedding-3-large` | 3072 | Higher quality |
| `bge` | `BAAI/bge-m3` | 1024 | Fully local, ~2.3 GB download |

**Vector stores** (set via `VECTOR_STORE`):

| Store | HNSW params | Start command |
|---|---|---|
| `qdrant` | m=16, ef=200 | `docker compose up qdrant` |
| `pgvector` | m=16, ef=200 | `docker compose up postgres` |

---

## Phase 3 — RAG Pipeline

A three-stage retrieval pipeline that combines semantic and lexical search, reranks results with a cross-encoder, and generates cited answers.

```bash
# CLI usage
python -m rag.pipeline --query "How does authentication work?" --rerank-top-k 5

# Python API
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
response = pipeline.query("What are the deployment requirements?")
print(response.answer)
for citation in response.citations:
    print(f"[{citation.index}] {citation.doc_title} — {citation.url}")
```

**Retrieval stages:**

```
HybridRetriever
  ├── DenseRetriever   → top-20 results from vector store
  ├── BM25Retriever    → top-20 results from in-memory index
  └── RRF fusion       → score = Σ weight / (60 + rank)  → top-20 fused

CrossEncoderReranker
  └── ms-marco-MiniLM-L-6-v2  → rescores all 20 pairs → top-5

PromptBuilder
  └── [SOURCE N] labelled context blocks (12k char budget)
```

---

## Phase 4 — API & UI

A production FastAPI backend with SSE streaming and a Streamlit chat interface.

```bash
# Local development
uvicorn api.main:app --reload --port 8000
streamlit run ui/streamlit/app.py

# Or via Docker
docker compose up api ui --build
```

**API endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe + pipeline readiness |
| `POST` | `/chat` | Synchronous JSON response |
| `POST` | `/chat/stream` | SSE token-by-token streaming |
| `GET` | `/sources` | Indexed document stats |
| `GET` | `/monitoring/metrics` | Prometheus scrape endpoint |
| `POST` | `/monitoring/feedback` | Submit thumbs up / down |
| `POST` | `/monitoring/eval` | Run RAGAs on a single response |
| `GET` | `/monitoring/eval/summary` | Aggregated RAGAs scores |
| `GET` | `/monitoring/feedback/summary` | Satisfaction rate + counts |

**Interactive API docs:** http://localhost:8000/docs

**SSE stream protocol:**

```
POST /chat/stream → text/event-stream

data: {"type": "token",    "content": "The deployment"}
data: {"type": "token",    "content": " process requires..."}
data: {"type": "citation", "content": "{...CitationSchema...}"}
data: {"type": "done",     "content": ""}
```

---

## Phase 5 — Monitoring

Full observability stack: Prometheus metrics, a pre-built Grafana dashboard, RAGAs quality evaluation, and a daily feedback loop.

```bash
docker compose up prometheus grafana --build
# Grafana  → http://localhost:3000  (admin / ragflow)
# Prometheus → http://localhost:9090
```

**Prometheus metrics:**

| Metric | Type | Description |
|---|---|---|
| `ragflow_requests_total` | Counter | Total queries by status (success/error) |
| `ragflow_request_latency_seconds` | Histogram | End-to-end latency |
| `ragflow_retrieval_latency_seconds` | Histogram | Hybrid retrieval latency |
| `ragflow_rerank_latency_seconds` | Histogram | Cross-encoder rerank latency |
| `ragflow_llm_latency_seconds` | Histogram | LLM call latency |
| `ragflow_candidates_retrieved` | Histogram | Hybrid candidates before reranking |
| `ragflow_context_chars` | Histogram | Characters sent to LLM |
| `ragflow_ragas_faithfulness` | Histogram | RAGAs faithfulness score (0–1) |
| `ragflow_ragas_answer_relevancy` | Histogram | RAGAs answer relevancy score (0–1) |
| `ragflow_feedback_total` | Counter | User feedback by rating |

**RAGAs quality targets:**

| Metric | Target | Below threshold = |
|---|---|---|
| Faithfulness | ≥ 0.8 | Retrieval failure — re-tune chunking |
| Answer Relevancy | ≥ 0.8 | Prompt failure — revise system prompt |

**Feedback loop** (Airflow DAG at 03:00 UTC):

```
thumbs_down feedback
  → RAGAs evaluation (faithfulness + relevancy)
  → classify: "retrieval" vs "prompt_or_llm" failure
  → write data/feedback/improvement_report.jsonl
```

---

## Configuration Reference

Copy `.env.example` to `.env` and fill in your values.

| Variable | Default | Phase | Description |
|---|---|---|---|
| `DATA_DIR` | `data` | 1 | Local storage root directory |
| `PDF_SOURCE_DIR` | — | 1 | Folder containing PDF files to ingest |
| `CONFLUENCE_URL` | — | 1 | Confluence base URL |
| `CONFLUENCE_USERNAME` | — | 1 | Atlassian email |
| `CONFLUENCE_API_TOKEN` | — | 1 | Atlassian API token |
| `CONFLUENCE_SPACES` | — | 1 | Comma-separated space keys, e.g. `ENG,PROD` |
| `GIT_REPOS` | — | 1 | Comma-separated repository URLs |
| `GIT_BRANCH` | `main` | 1 | Branch to ingest |
| `GIT_FILE_EXTENSIONS` | `.py,.md,.rst,.txt` | 1 | File types to include |
| `CHUNK_SIZE` | `512` | 1 | Target characters per chunk |
| `CHUNK_OVERLAP` | `64` | 1 | Overlap between consecutive chunks |
| `CHUNKING_STRATEGY` | `recursive` | 1 | `recursive` / `sentence` / `semantic` |
| `EMBEDDING_PROVIDER` | `openai` | 2 | `openai` or `bge` |
| `OPENAI_API_KEY` | — | 2+ | OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 2 | Embedding model name |
| `BGE_MODEL_NAME` | `BAAI/bge-m3` | 2 | HuggingFace BGE model id |
| `EMBEDDING_BATCH_SIZE` | `64` | 2 | Texts per embedding API call |
| `VECTOR_STORE` | `qdrant` | 2 | `qdrant` or `pgvector` |
| `QDRANT_HOST` | `localhost` | 2 | Qdrant server host |
| `QDRANT_PORT` | `6333` | 2 | Qdrant server port |
| `QDRANT_COLLECTION` | `ragflow` | 2 | Qdrant collection name |
| `PGVECTOR_DSN` | `postgresql://...` | 2 | PostgreSQL connection string |
| `HYBRID_TOP_K` | `20` | 3 | Candidates from hybrid retrieval |
| `RERANK_TOP_K` | `5` | 3 | Passages after cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 3 | Cross-encoder model |
| `LLM_MODEL` | `gpt-4o-mini` | 3 | Chat completion model |
| `LLM_TEMPERATURE` | `0.2` | 3 | LLM sampling temperature |
| `LOG_LEVEL` | `INFO` | all | Python logging level |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# By phase
pytest tests/ingestion/   -v
pytest tests/embeddings/  -v
pytest tests/rag/         -v
pytest tests/api/         -v
pytest tests/monitoring/  -v

# Lint
ruff check .
```

---

## Docker Deployment

```bash
# Full stack (all 6 services)
docker compose up --build

# Service URLs
# Chat UI     → http://localhost:8501
# API         → http://localhost:8000
# API Docs    → http://localhost:8000/docs
# Grafana     → http://localhost:3000  (admin / ragflow)
# Prometheus  → http://localhost:9090
# Qdrant      → http://localhost:6333

# Scale the API horizontally
docker compose up --build --scale api=3
```

Individual services for development:

```bash
docker compose up qdrant -d          # vector store only
docker compose up qdrant api -d      # API without UI/monitoring
docker compose up prometheus grafana -d  # monitoring stack only
```

---

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feat/your-feature`
2. Follow the existing commit convention: `type(scope): description`
   - `feat` — new feature
   - `fix` — bug fix
   - `refactor` — code restructuring
   - `test` — test additions or fixes
   - `chore` — dependencies, tooling, config
   - `docs` — documentation only
3. Ensure all tests pass: `pytest tests/ -v`
4. Lint before committing: `ruff check .`
5. Open a pull request with a clear description of the change and its motivation

---

## License

MIT License — see `LICENSE` for details.
