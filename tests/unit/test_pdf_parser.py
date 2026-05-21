"""Unit tests for the PDF parser."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from cognilink.extract.pdf import pdf_parser


class TestPdfParser:
    """Tests for pdf_parser function."""

    def test_returns_empty_string_when_pypdf2_not_installed(self):
        """When PyPDF2 is not available, return empty string."""
        with patch.dict("sys.modules", {"PyPDF2": None}):
            # Force reimport to trigger ImportError
            import importlib
            import cognilink.extract.pdf as pdf_mod

            importlib.reload(pdf_mod)
            result = pdf_mod.pdf_parser("/nonexistent.pdf")
            assert result == ""

    def test_returns_empty_string_for_nonexistent_file(self):
        """Non-existent file path returns empty string."""
        result = pdf_parser("/tmp/nonexistent_file_abc123.pdf")
        assert result == ""

    def test_returns_empty_string_for_malformed_file(self):
        """A file that is not a valid PDF returns empty string."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"This is not a PDF file at all")
            f.flush()
            result = pdf_parser(f.name)
            assert result == ""

    def test_extracts_text_from_valid_pdf(self):
        """Mock a valid PDF and verify text extraction."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"

        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page1, mock_page2]

        with patch("cognilink.extract.pdf.PdfReader", create=True) as mock_cls:
            # We need to patch the import inside the function
            mock_module = MagicMock()
            mock_module.PdfReader = MagicMock(return_value=mock_reader)

            with patch.dict("sys.modules", {"PyPDF2": mock_module}):
                import importlib
                import cognilink.extract.pdf as pdf_mod

                importlib.reload(pdf_mod)
                result = pdf_mod.pdf_parser("test.pdf")
                assert result == "Page 1 content\nPage 2 content"

    def test_returns_empty_string_for_encrypted_pdf(self):
        """Encrypted PDF returns empty string."""
        mock_reader = MagicMock()
        mock_reader.is_encrypted = True

        mock_module = MagicMock()
        mock_module.PdfReader = MagicMock(return_value=mock_reader)

        with patch.dict("sys.modules", {"PyPDF2": mock_module}):
            import importlib
            import cognilink.extract.pdf as pdf_mod

            importlib.reload(pdf_mod)
            result = pdf_mod.pdf_parser("encrypted.pdf")
            assert result == ""

    def test_handles_page_extraction_failure_gracefully(self):
        """If a page fails to extract, skip it and continue."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Good page"
        mock_page2 = MagicMock()
        mock_page2.extract_text.side_effect = Exception("Corrupt page")
        mock_page3 = MagicMock()
        mock_page3.extract_text.return_value = "Another good page"

        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page1, mock_page2, mock_page3]

        mock_module = MagicMock()
        mock_module.PdfReader = MagicMock(return_value=mock_reader)

        with patch.dict("sys.modules", {"PyPDF2": mock_module}):
            import importlib
            import cognilink.extract.pdf as pdf_mod

            importlib.reload(pdf_mod)
            result = pdf_mod.pdf_parser("partial.pdf")
            assert result == "Good page\nAnother good page"
