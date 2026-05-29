# ragflow-enterprise

> A production-grade Retrieval-Augmented Generation (RAG) pipeline with multi-source ingestion (PDF, Confluence, Git), hybrid retrieval, cross-encoder reranking, streaming API, and full observability — built incrementally across 5 phases.

## Architecture

```
Phase 1 — Ingestion      : PDF / Confluence / Git connectors, chunking, S3, Airflow DAG
Phase 2 — Embeddings     : OpenAI / BGE-M3 vectors, PGVector / Qdrant (HNSW), <100ms search
Phase 3 — RAG Pipeline   : Hybrid retriever (dense + BM25), cross-encoder reranking, citations
Phase 4 — API + UI       : FastAPI SSE streaming, Streamlit / Next.js chat, Docker
Phase 5 — Monitoring     : Grafana dashboard, RAGAs evaluation, feedback loop
```

## Phase 1 — Ingestion

### Structure
```
ingestion/
├── connectors/
│   ├── base.py            # Abstract base connector
│   ├── pdf_connector.py   # PDF ingestion (local + S3)
│   ├── confluence.py      # Confluence REST API connector
│   └── git_connector.py   # Git repository connector
├── chunking/
│   └── smart_chunker.py   # Semantic + recursive chunking strategies
├── storage/
│   └── s3_storage.py      # S3 upload / download / listing
airflow/
├── dags/
│   └── ingestion_dag.py   # Daily ingestion Airflow DAG
config/
└── settings.py            # Central settings (pydantic-settings)
```

### Quick Start

```bash
cp .env.example .env          # fill in your secrets
pip install -r requirements.txt
python -m ingestion.run       # one-shot ingestion run
```

### Environment Variables

| Variable | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `S3_BUCKET_NAME` | Target S3 bucket |
| `CONFLUENCE_URL` | Confluence base URL |
| `CONFLUENCE_USERNAME` | Confluence username / email |
| `CONFLUENCE_API_TOKEN` | Confluence API token |
| `GIT_REPOS` | Comma-separated list of repo URLs |

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check .
```
