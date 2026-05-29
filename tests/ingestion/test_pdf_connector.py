"""tests/ingestion/test_pdf_connector.py"""
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.connectors.pdf_connector import PDFConnector


class TestPDFConnectorLocal:
    def test_raises_if_no_source(self):
        connector = PDFConnector()
        with pytest.raises(ValueError, match="requires either"):
            list(connector.fetch())

    def test_nonexistent_dir_raises(self):
        connector = PDFConnector(source_dir="/nonexistent/path")
        with pytest.raises(ValueError):
            list(connector.fetch())

    def test_empty_pdf_skipped(self, tmp_path):
        """A PDF that yields no text should be skipped gracefully."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")  # minimal invalid PDF bytes

        connector = PDFConnector(source_dir=str(tmp_path))
        # pdfplumber may raise or return empty text — connector should not crash
        docs = list(connector.fetch())
        # We just assert it doesn't raise; empty docs are acceptable
        assert isinstance(docs, list)

    def test_name_property(self):
        connector = PDFConnector()
        assert connector.name == "pdf"
