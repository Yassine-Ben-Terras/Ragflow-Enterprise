"""
embeddings/stores/pgvector_store.py
PostgreSQL + pgvector store with HNSW indexing.

Run locally:
    docker run -e POSTGRES_PASSWORD=ragflow -e POSTGRES_USER=ragflow \
               -e POSTGRES_DB=ragflow -p 5432:5432 pgvector/pgvector:pg16
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from embeddings.stores.base import BaseVectorStore, SearchResult, VectorRecord

logger = logging.getLogger(__name__)


class PGVectorStore(BaseVectorStore):
    """
    Args:
        dsn:        PostgreSQL connection string.
        table_name: Table that holds vectors.
    """

    @property
    def name(self) -> str:
        return "pgvector"

    def __init__(
        self,
        dsn: str = "postgresql://ragflow:ragflow@localhost:5432/ragflow",
        table_name: str = "ragflow_chunks",
    ) -> None:
        self.dsn = dsn
        self.table_name = table_name
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            try:
                import psycopg2
                import psycopg2.extras
            except ImportError as exc:
                raise ImportError(
                    "psycopg2-binary is required. Install with: pip install psycopg2-binary"
                ) from exc
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
            logger.info("Connected to PostgreSQL via psycopg2.")
        return self._conn

    def create_collection(self, dimension: int, recreate: bool = False) -> None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            if recreate:
                cur.execute(f"DROP TABLE IF EXISTS {self.table_name};")
                logger.info("Dropped table '%s'.", self.table_name)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id          BIGSERIAL PRIMARY KEY,
                    chunk_id    TEXT UNIQUE NOT NULL,
                    doc_source_id TEXT NOT NULL,
                    doc_source  TEXT NOT NULL,
                    doc_title   TEXT,
                    text        TEXT NOT NULL,
                    url         TEXT,
                    file_path   TEXT,
                    metadata    JSONB DEFAULT '{{}}',
                    embedding   vector({dimension})
                );
            """)

            # HNSW index — much faster than IVFFlat for <1M vectors
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_hnsw_idx
                ON {self.table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200);
            """)

        logger.info(
            "PGVector table '%s' ready (dim=%d, HNSW index).", self.table_name, dimension
        )

    def upsert(self, records: List[VectorRecord]) -> None:
        import psycopg2.extras

        conn = self._get_conn()
        rows = [
            (
                r.chunk_id,
                r.doc_source_id,
                r.doc_source,
                r.doc_title,
                r.text,
                r.url,
                r.file_path,
                json.dumps(r.metadata),
                r.vector,
            )
            for r in records
        ]
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"""
                INSERT INTO {self.table_name}
                    (chunk_id, doc_source_id, doc_source, doc_title, text,
                     url, file_path, metadata, embedding)
                VALUES %s
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text      = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    metadata  = EXCLUDED.metadata;
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
            )
        logger.debug("Upserted %d vectors into '%s'.", len(records), self.table_name)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        vec_str = "[" + ",".join(str(v) for v in query_vector) + "]"
        sql = f"""
            SELECT chunk_id, doc_title, doc_source, text, url, file_path, metadata,
                   1 - (embedding <=> '{vec_str}'::vector) AS score
            FROM {self.table_name}
            ORDER BY embedding <=> '{vec_str}'::vector
            LIMIT %s;
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, (top_k,))
            rows = cur.fetchall()

        results = []
        for row in rows:
            chunk_id, doc_title, doc_source, text, url, file_path, metadata, score = row
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(SearchResult(
                chunk_id=chunk_id,
                doc_title=doc_title or "",
                doc_source=doc_source or "",
                text=text,
                score=score,
                url=url,
                file_path=file_path,
                metadata=metadata or {},
            ))
        return results

    def count(self) -> int:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
            return cur.fetchone()[0]
