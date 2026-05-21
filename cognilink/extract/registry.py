"""CogniLink ExtractorRegistry — plugin-based file parser registry with concurrent extraction.

Provides a registry for file extension parsers and concurrent multi-threaded extraction
of raw text from heterogeneous document formats with zero LLM token consumption.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """Plugin-based file parser registry enabling concurrent document extraction.

    The registry maps file extensions to parser callables. During extraction,
    files are processed in parallel using a ThreadPoolExecutor. Files with
    unregistered extensions fall back to plain UTF-8 text reading. Non-existent
    or unreadable files are skipped with a warning log.

    Zero LLM tokens are consumed during extraction — all parsing is programmatic.
    """

    def __init__(self) -> None:
        """Initialize the registry with an empty parser mapping."""
        self._parsers: Dict[str, Callable[[str], str]] = {}

    def register_parser(self, extension: str, parser_fn: Callable[[str], str]) -> None:
        """Register a parser function for a given file extension.

        Args:
            extension: File extension including the dot (e.g., '.pdf', '.xlsx').
            parser_fn: A callable that accepts a file path string and returns
                       the extracted text content as a string.
        """
        # Normalize extension to lowercase with leading dot
        ext = extension if extension.startswith(".") else f".{extension}"
        self._parsers[ext.lower()] = parser_fn

    def get_registered_extensions(self) -> List[str]:
        """Return a list of all registered file extensions."""
        return list(self._parsers.keys())

    def parse_workspace_concurrently(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Extract raw text from multiple files concurrently.

        Uses ThreadPoolExecutor for parallel extraction. Each file produces a
        result dict with deterministic node ID, extension, raw text, and path.
        Non-existent or unreadable files are skipped with a warning.

        Args:
            paths: List of file path strings to extract.

        Returns:
            List of dicts with keys: "id", "ext", "raw_text", "path".
            The "id" is in the format NODE_{UPPERCASE_FILENAME_WITHOUT_EXTENSION}
            with hyphens replaced by underscores.
        """
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._extract_single, p): p for p in paths}
            results: List[Dict[str, Any]] = []
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)
        return results

    def _extract_single(self, file_path: str) -> Dict[str, Any] | None:
        """Extract text from a single file.

        Args:
            file_path: Path to the file to extract.

        Returns:
            A dict with keys "id", "ext", "raw_text", "path", or None if the
            file cannot be read.
        """
        path = Path(file_path)

        # Check if file exists and is readable
        if not path.exists():
            logger.warning("Skipping non-existent file: %s", file_path)
            return None

        if not os.access(path, os.R_OK):
            logger.warning("Skipping unreadable file: %s", file_path)
            return None

        ext = path.suffix.lower()
        node_id = self._generate_node_id(path)

        try:
            if ext in self._parsers:
                raw_text = self._parsers[ext](file_path)
            else:
                # Fallback: read as plain UTF-8 text
                raw_text = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Skipping unreadable file: %s (error: %s)", file_path, e)
            return None

        return {
            "id": node_id,
            "ext": ext,
            "raw_text": raw_text,
            "path": file_path,
        }

    @staticmethod
    def _generate_node_id(path: Path) -> str:
        """Generate a deterministic node ID from a file path.

        Format: NODE_{UPPERCASE_FILENAME_WITHOUT_EXTENSION}
        Hyphens in the filename are replaced with underscores.

        Args:
            path: Path object for the file.

        Returns:
            Deterministic node ID string.
        """
        stem = path.stem  # filename without extension
        normalized = stem.replace("-", "_")
        return f"NODE_{normalized.upper()}"
