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
