"""
ingestion/connectors/confluence.py
Fetches pages from one or more Confluence spaces using the REST API v2.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterator, List, Optional

import requests
from bs4 import BeautifulSoup

from ingestion.connectors.base import BaseConnector, Document

logger = logging.getLogger(__name__)

_PAGE_LIMIT = 50  # Confluence default max per request


class ConfluenceConnector(BaseConnector):
    """
    Streams pages from Confluence Cloud (or Server) using basic auth.

    Args:
        base_url:   e.g. "https://your-org.atlassian.net"
        username:   Atlassian account email
        api_token:  Atlassian API token (or password for Server)
        spaces:     List of space keys to ingest, e.g. ["ENG", "PROD"]
    """

    @property
    def name(self) -> str:
        return "confluence"

    def __init__(
        self,
        base_url: str,
        username: str,
        api_token: str,
        spaces: Optional[List[str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.spaces = spaces or []
        self._session = requests.Session()
        self._session.auth = (username, api_token)
        self._session.headers.update({"Accept": "application/json"})

    # ── helpers ─────────────────────────────────────────────────────────

    def _html_to_text(self, html: str) -> str:
        """Strip HTML tags and return clean plain text."""
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n").strip()

    def _get_pages_in_space(self, space_key: str) -> Iterator[dict]:
        """Paginate through all pages in a Confluence space."""
        url = f"{self.base_url}/wiki/rest/api/content"
        params = {
            "spaceKey": space_key,
            "type": "page",
            "status": "current",
            "expand": "body.storage,metadata.labels,version,space",
            "limit": _PAGE_LIMIT,
            "start": 0,
        }

        while True:
            try:
                resp = self._session.get(url, params=params, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.error("Confluence API error for space %s: %s", space_key, exc)
                break

            data = resp.json()
            results = data.get("results", [])

            for page in results:
                yield page

            # Pagination
            size = data.get("size", 0)
            total_size = data.get("totalSize", size)
            next_start = params["start"] + size  # type: ignore[operator]

            if next_start >= total_size or size == 0:
                break
            params["start"] = next_start  # type: ignore[assignment]

    def _page_to_document(self, page: dict) -> Optional[Document]:
        page_id: str = page.get("id", "")
        title: str = page.get("title", "")
        space_key: str = page.get("space", {}).get("key", "")

        # Body HTML
        body_html: str = page.get("body", {}).get("storage", {}).get("value", "")
        text = self._html_to_text(body_html)

        if not text:
            return None

        page_url = f"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}"
        labels = [lbl["name"] for lbl in page.get("metadata", {}).get("labels", {}).get("results", [])]
        version = page.get("version", {}).get("number", 1)
        source_id = hashlib.sha256(page_id.encode()).hexdigest()[:16]

        return Document(
            source="confluence",
            source_id=source_id,
            title=title,
            content=text,
            url=page_url,
            metadata={
                "page_id": page_id,
                "space_key": space_key,
                "labels": labels,
                "version": version,
                "char_count": len(text),
            },
        )

    # ── public API ──────────────────────────────────────────────────────

    def fetch(self) -> Iterator[Document]:
        if not self.spaces:
            logger.warning("No Confluence spaces configured — nothing to fetch.")
            return

        for space_key in self.spaces:
            logger.info("Fetching Confluence space: %s", space_key)
            page_count = 0

            for raw_page in self._get_pages_in_space(space_key):
                doc = self._page_to_document(raw_page)
                if doc:
                    page_count += 1
                    yield doc

            logger.info("Space %s: fetched %d page(s).", space_key, page_count)
