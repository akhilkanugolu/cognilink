"""PDF document parser for CogniLink extraction layer."""

import logging

logger = logging.getLogger(__name__)


def pdf_parser(path: str) -> str:
    """Extract text from a PDF file using PyPDF2.

    Opens the PDF at the given path, iterates over all pages, and
    concatenates extracted text with newline separators.

    Args:
        path: File path to the PDF document.

    Returns:
        Extracted text from all pages concatenated with newlines.
        Returns an empty string if the PDF is encrypted, malformed,
        or if PyPDF2 is not installed.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        logger.warning(
            "PyPDF2 is not installed. Install it with: pip install cognilink[pdf]"
        )
        return ""

    try:
        reader = PdfReader(path)
    except Exception as exc:
        logger.warning("Failed to open PDF '%s': %s", path, exc)
        return ""

    if reader.is_encrypted:
        logger.warning("PDF '%s' is encrypted and cannot be read.", path)
        return ""

    pages_text: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        except Exception as exc:
            logger.warning(
                "Failed to extract text from a page in '%s': %s", path, exc
            )

    return "\n".join(pages_text)
