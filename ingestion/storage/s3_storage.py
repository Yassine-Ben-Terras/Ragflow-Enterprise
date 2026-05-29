"""
ingestion/storage/s3_storage.py
Upload / download / list chunks and raw documents on S3.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

from ingestion.chunking.smart_chunker import Chunk
from ingestion.connectors.base import Document

logger = logging.getLogger(__name__)


class S3Storage:
    """
    Persists raw Documents and Chunks to S3 as newline-delimited JSON.

    Layout:
      <prefix>/raw/<source>/<source_id>.json       ← raw Document
      <prefix>/chunks/<source>/<doc_source_id>/chunks.jsonl  ← all chunks for a doc
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "ragflow/",
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    # ── internal helpers ─────────────────────────────────────────────────

    def _raw_key(self, doc: Document) -> str:
        return f"{self.prefix}/raw/{doc.source}/{doc.source_id}.json"

    def _chunks_key(self, doc_source: str, doc_source_id: str) -> str:
        return f"{self.prefix}/chunks/{doc_source}/{doc_source_id}/chunks.jsonl"

    def _put(self, key: str, body: str) -> None:
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )

    def _get(self, key: str) -> str:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    # ── public API ───────────────────────────────────────────────────────

    def save_document(self, doc: Document) -> str:
        """Persist a raw Document. Returns the S3 key."""
        key = self._raw_key(doc)
        self._put(key, json.dumps(doc.to_dict(), ensure_ascii=False))
        logger.debug("Saved document → s3://%s/%s", self.bucket, key)
        return key

    def save_chunks(self, chunks: List[Chunk]) -> str:
        """Persist all chunks for a document as JSONL. Returns the S3 key."""
        if not chunks:
            raise ValueError("chunks list is empty")

        key = self._chunks_key(chunks[0].doc_source, chunks[0].doc_source_id)
        body = "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks)
        self._put(key, body)
        logger.debug(
            "Saved %d chunk(s) → s3://%s/%s", len(chunks), self.bucket, key
        )
        return key

    def load_chunks(self, doc_source: str, doc_source_id: str) -> List[Chunk]:
        """Load persisted chunks from S3 back into Chunk objects."""
        key = self._chunks_key(doc_source, doc_source_id)
        try:
            body = self._get(key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise

        chunks: List[Chunk] = []
        for line in body.splitlines():
            d = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=d["chunk_id"],
                    doc_source_id=d["doc_source_id"],
                    doc_source=d["doc_source"],
                    doc_title=d["doc_title"],
                    text=d["text"],
                    chunk_index=d["chunk_index"],
                    total_chunks=d["total_chunks"],
                    url=d.get("url"),
                    file_path=d.get("file_path"),
                    metadata=d.get("metadata", {}),
                )
            )
        return chunks

    def list_raw_keys(self, source: Optional[str] = None) -> List[str]:
        """List all raw document keys, optionally filtered by source type."""
        prefix = f"{self.prefix}/raw/"
        if source:
            prefix += f"{source}/"

        paginator = self._client.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def document_exists(self, doc: Document) -> bool:
        """Return True if this document has already been stored (deduplication)."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._raw_key(doc))
            return True
        except ClientError:
            return False
