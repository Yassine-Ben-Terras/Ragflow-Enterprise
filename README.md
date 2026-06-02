# ragflow-enterprise

> A production-grade Retrieval-Augmented Generation (RAG) pipeline with multi-source ingestion (PDF, Confluence, Git), hybrid retrieval, cross-encoder reranking, streaming API, and full observability — built incrementally across 5 phases.

## Architecture

```
Phase 1 — Ingestion      : PDF / Confluence / Git connectors, chunking, local storage
Phase 2 — Embeddings     : OpenAI / BGE-M3 vectors, PGVector / Qdrant (HNSW), <100ms search
Phase 3 — RAG Pipeline   : Hybrid retriever (dense + BM25), cross-encoder reranking, citations
Phase 4 — API + UI       : FastAPI SSE streaming, Streamlit / Next.js chat, Docker
Phase 5 — Monitoring     : Grafana dashboard, RAGAs evaluation, feedback loop
```

## Quick Start

```bash
cp .env.example .env

# 1 — install dependencies
pip install -r requirements.txt

# 2 — start vector store (pick one)
docker compose up qdrant       # default
docker compose up postgres     # for pgvector

# 3 — ingest documents
python -m ingestion.run

# 4 — embed and index
python -m embeddings.pipeline
```

## Phase 1 — Ingestion

```
ingestion/
├── connectors/
│   ├── base.py              AbstractConnector + Document dataclass
│   ├── pdf_connector.py     Local PDF ingestion (pdfplumber)
│   ├── confluence.py        Confluence REST API v2
│   └── git_connector.py     Git shallow clone + file walker
├── chunking/
│   └── smart_chunker.py     Recursive / sentence / semantic strategies
├── storage/
│   └── local_storage.py     JSON + JSONL on local filesystem (data/)
└── run.py                   Orchestrator
airflow/dags/ingestion_dag.py  Daily Airflow DAG
```

## Phase 2 — Embeddings

```
embeddings/
├── providers/
│   ├── base.py              BaseEmbedder ABC
│   ├── openai_embedder.py   OpenAI text-embedding-3-small / large
│   └── bge_embedder.py      BAAI/bge-m3 local model (sentence-transformers)
├── stores/
│   ├── base.py              BaseVectorStore ABC
│   ├── qdrant_store.py      Qdrant + HNSW (default)
│   └── pgvector_store.py    PostgreSQL + pgvector + HNSW
└── pipeline.py              Orchestrator: chunks → embed → upsert
docker-compose.yml           Local Qdrant + Postgres containers
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `data` | Local storage root |
| `PDF_SOURCE_DIR` | `` | Folder with PDF files |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `bge` |
| `OPENAI_API_KEY` | `` | Required when provider=openai |
| `VECTOR_STORE` | `qdrant` | `qdrant` or `pgvector` |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `PGVECTOR_DSN` | `...` | PostgreSQL connection string |

## Development

```bash
pytest tests/ -v
ruff check .
```

## Phase 3 — RAG Pipeline

```
rag/
├── retriever/
│   ├── dense_retriever.py       Semantic search via vector store (Phase 2)
│   ├── bm25_retriever.py        BM25 sparse retrieval over local chunks
│   └── hybrid_retriever.py      RRF fusion of dense + BM25, weighted
├── reranker/
│   └── cross_encoder_reranker.py  ms-marco-MiniLM cross-encoder, lazy loaded
├── prompt/
│   └── prompt_builder.py        [SOURCE N] citation blocks, context budget trim
└── pipeline.py                  Orchestrator + CLI  (python -m rag.pipeline -q "...")
```

### Usage

```bash
# Ask a question via CLI
python -m rag.pipeline --query "What is the deployment process?" --rerank-top-k 5

# Import in Python
from rag.pipeline import RAGPipeline
pipeline = RAGPipeline()
response = pipeline.query("How does authentication work?")
print(response.pretty())
```

### Pipeline flow

```
query
  ↓  HybridRetriever (dense top-20 + BM25 top-20 → RRF fusion → top-20)
  ↓  CrossEncoderReranker (ms-marco-MiniLM → top-5)
  ↓  PromptBuilder ([SOURCE N] labels, 12k char budget)
  ↓  OpenAI gpt-4o-mini
  ↓  RAGResponse { answer, citations[] }
```

## Phase 4 — API + UI

```
api/
├── main.py              FastAPI app factory, lifespan, CORS
├── state.py             Singleton RAGPipeline (loaded once at startup)
├── schemas.py           Pydantic v2 request/response models
└── routers/
    ├── health.py        GET  /health
    ├── chat.py          POST /chat  (JSON)  +  POST /chat/stream  (SSE)
    └── sources.py       GET  /sources
ui/streamlit/
└── app.py               Streamlit chat UI — streaming, citations panel, sidebar stats
Dockerfile.api           FastAPI container
Dockerfile.ui            Streamlit container
docker-compose.yml       All services: qdrant + postgres + api + ui
```

### Quick Start (local, no Docker)

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — UI
streamlit run ui/streamlit/app.py
```

### Quick Start (Docker)

```bash
docker compose up --build
# API  → http://localhost:8000
# UI   → http://localhost:8501
# Docs → http://localhost:8000/docs
```

### SSE Stream Protocol

```
POST /chat/stream  →  text/event-stream

data: {"type": "token",    "content": "The answer"}
data: {"type": "token",    "content": " is 42"}
data: {"type": "citation", "content": "{...CitationSchema...}"}
data: {"type": "done",     "content": ""}
```

## Phase 5 — Monitoring

```
monitoring/
├── metrics.py                     Prometheus counters + histograms (all pipeline stages)
├── instrumented_pipeline.py       Drop-in RAGPipeline wrapper with per-stage timing
├── ragas/
│   └── evaluator.py               LLM-as-judge: faithfulness + answer relevancy scores
├── feedback/
│   ├── feedback_store.py          JSONL persistence + Prometheus counter for thumbs up/down
│   └── feedback_loop.py           Daily batch: negative feedback → RAGAs eval → improvement report
├── prometheus/
│   └── prometheus.yml             Scrape config (scrapes /monitoring/metrics every 15s)
└── grafana/
    ├── dashboards/ragflow_overview.json   Pre-built dashboard (latency, RAGAs, feedback, errors)
    └── provisioning/               Auto-wired datasource + dashboard on first boot
api/routers/monitoring.py          GET /monitoring/metrics  POST /feedback  POST /eval  GET summaries
airflow/dags/feedback_loop_dag.py  Daily 03:00 UTC DAG — process feedback → improvement report
```

### All services

```bash
docker compose up --build

# Endpoints
# API        → http://localhost:8000
# UI         → http://localhost:8501
# Prometheus → http://localhost:9090
# Grafana    → http://localhost:3000  (admin / ragflow)
```

### Monitoring API

| Endpoint | Method | Description |
|---|---|---|
| `/monitoring/metrics` | GET | Prometheus scrape endpoint |
| `/monitoring/feedback` | POST | Submit thumbs_up / thumbs_down |
| `/monitoring/feedback/summary` | GET | Satisfaction rate + counts |
| `/monitoring/eval` | POST | Run RAGAs on a single response |
| `/monitoring/eval/summary` | GET | Mean faithfulness + relevancy |

### RAGAs Metrics

| Metric | Target | Description |
|---|---|---|
| Faithfulness | ≥ 0.8 | Claims in answer supported by context |
| Answer Relevancy | ≥ 0.8 | Answer actually addresses the question |

### Feedback Loop

Runs daily at 03:00 UTC via Airflow. Identifies low-scoring responses and classifies failures as either **retrieval** (noisy chunks → re-tune chunking) or **prompt/LLM** (answer ignores context → revise system prompt).
