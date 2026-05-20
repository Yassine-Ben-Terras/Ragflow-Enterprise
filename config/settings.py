"""
config/settings.py
Central configuration loaded from environment variables / .env file.
"""

from __future__ import annotations

from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Local paths ───────────────────────────────────────
    data_dir: str = Field(default="data", alias="DATA_DIR")
    pdf_source_dir: str = Field(default="", alias="PDF_SOURCE_DIR")

    # ── Confluence ────────────────────────────────────────
    confluence_url: str = Field(default="", alias="CONFLUENCE_URL")
    confluence_username: str = Field(default="", alias="CONFLUENCE_USERNAME")
    confluence_api_token: str = Field(default="", alias="CONFLUENCE_API_TOKEN")
    confluence_spaces: List[str] = Field(default_factory=list, alias="CONFLUENCE_SPACES")

    @field_validator("confluence_spaces", mode="before")
    @classmethod
    def split_confluence_spaces(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    # ── Git ───────────────────────────────────────────────
    git_repos: List[str] = Field(default_factory=list, alias="GIT_REPOS")
    git_branch: str = Field(default="main", alias="GIT_BRANCH")
    git_file_extensions: List[str] = Field(
        default_factory=lambda: [".py", ".md", ".rst", ".txt"],
        alias="GIT_FILE_EXTENSIONS",
    )

    @field_validator("git_repos", "git_file_extensions", mode="before")
    @classmethod
    def split_comma_list(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    # ── Chunking ──────────────────────────────────────────
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    chunking_strategy: str = Field(default="recursive", alias="CHUNKING_STRATEGY")

    # ── Embeddings (Phase 2) ──────────────────────────────
    embedding_provider: str = Field(default="openai", alias="EMBEDDING_PROVIDER")   # openai | bge
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    bge_model_name: str = Field(default="BAAI/bge-m3", alias="BGE_MODEL_NAME")
    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")

    # ── Vector store (Phase 2) ────────────────────────────
    vector_store: str = Field(default="qdrant", alias="VECTOR_STORE")               # qdrant | pgvector
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(default="ragflow", alias="QDRANT_COLLECTION")
    pgvector_dsn: str = Field(default="postgresql://ragflow:ragflow@localhost:5432/ragflow", alias="PGVECTOR_DSN")

    # ── Logging ───────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


settings = Settings()


# ── RAG Pipeline (Phase 3) ────────────────────────────────
hybrid_top_k: int = Field(default=20, alias="HYBRID_TOP_K")
rerank_top_k: int = Field(default=5,  alias="RERANK_TOP_K")
reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL")
llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
