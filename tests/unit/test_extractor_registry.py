"""Unit tests for ExtractorRegistry."""

import logging
import tempfile
from pathlib import Path

import pytest

from cognilink.extract.registry import ExtractorRegistry


@pytest.fixture
def registry() -> ExtractorRegistry:
    """Create a fresh ExtractorRegistry instance."""
    return ExtractorRegistry()


@pytest.fixture
def sample_workspace(tmp_path: Path) -> dict:
    """Create a temporary workspace with sample files."""
    txt_file = tmp_path / "hello.txt"
    txt_file.write_text("Hello, world!", encoding="utf-8")

    py_file = tmp_path / "main.py"
    py_file.write_text("print('hello')", encoding="utf-8")

    md_file = tmp_path / "my-readme.md"
    md_file.write_text("# Title\nSome content", encoding="utf-8")

    return {
        "txt": str(txt_file),
        "py": str(py_file),
        "md": str(md_file),
        "dir": tmp_path,
    }


class TestRegisterParser:
    """Tests for register_parser method."""

    def test_register_parser_with_dot(self, registry: ExtractorRegistry) -> None:
        """Parser registered with dot prefix is stored correctly."""
        registry.register_parser(".pdf", lambda p: "pdf content")
        assert ".pdf" in registry.get_registered_extensions()

    def test_register_parser_without_dot(self, registry: ExtractorRegistry) -> None:
        """Parser registered without dot prefix is normalized."""
        registry.register_parser("pdf", lambda p: "pdf content")
        assert ".pdf" in registry.get_registered_extensions()

    def test_register_parser_case_insensitive(self, registry: ExtractorRegistry) -> None:
        """Extension is normalized to lowercase."""
        registry.register_parser(".PDF", lambda p: "pdf content")
        assert ".pdf" in registry.get_registered_extensions()

    def test_registered_parser_is_used(
        self, registry: ExtractorRegistry, sample_workspace: dict
    ) -> None:
        """Registered parser is dispatched for matching extension."""
        registry.register_parser(".txt", lambda p: "custom parsed")
        results = registry.parse_workspace_concurrently([sample_workspace["txt"]])
        assert len(results) == 1
        assert results[0]["raw_text"] == "custom parsed"


class TestParseWorkspaceConcurrently:
    """Tests for parse_workspace_concurrently method."""

    def test_returns_correct_keys(
        self, registry: ExtractorRegistry, sample_workspace: dict
    ) -> None:
        """Each result dict has the expected keys."""
        results = registry.parse_workspace_concurrently([sample_workspace["txt"]])
        assert len(results) == 1
        result = results[0]
        assert set(result.keys()) == {"id", "ext", "raw_text", "path"}

    def test_fallback_utf8_reader(
        self, registry: ExtractorRegistry, sample_workspace: dict
    ) -> None:
        """Unregistered extensions fall back to plain UTF-8 reading."""
        results = registry.parse_workspace_concurrently([sample_workspace["py"]])
        assert len(results) == 1
        assert results[0]["raw_text"] == "print('hello')"
        assert results[0]["ext"] == ".py"

    def test_multiple_files_concurrent(
        self, registry: ExtractorRegistry, sample_workspace: dict
    ) -> None:
        """Multiple files are extracted concurrently."""
        paths = [sample_workspace["txt"], sample_workspace["py"], sample_workspace["md"]]
        results = registry.parse_workspace_concurrently(paths)
        assert len(results) == 3

    def test_skips_nonexistent_file(
        self, registry: ExtractorRegistry, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-existent files are skipped with a warning."""
        with caplog.at_level(logging.WARNING):
            results = registry.parse_workspace_concurrently(["/nonexistent/file.txt"])
        assert len(results) == 0
        assert "Skipping non-existent file" in caplog.text

    def test_skips_unreadable_file_continues_others(
        self, registry: ExtractorRegistry, sample_workspace: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unreadable files are skipped but other files still process."""
        paths = ["/nonexistent/file.txt", sample_workspace["txt"]]
        with caplog.at_level(logging.WARNING):
            results = registry.parse_workspace_concurrently(paths)
        assert len(results) == 1
        assert results[0]["raw_text"] == "Hello, world!"

    def test_empty_paths_list(self, registry: ExtractorRegistry) -> None:
        """Empty paths list returns empty results."""
        results = registry.parse_workspace_concurrently([])
        assert results == []

    def test_parser_exception_skips_file(
        self, registry: ExtractorRegistry, sample_workspace: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If a parser raises an exception, the file is skipped with a warning."""

        def bad_parser(path: str) -> str:
            raise RuntimeError("Parser failed")

        registry.register_parser(".txt", bad_parser)
        with caplog.at_level(logging.WARNING):
            results = registry.parse_workspace_concurrently([sample_workspace["txt"]])
        assert len(results) == 0
        assert "Skipping unreadable file" in caplog.text


class TestNodeIdGeneration:
    """Tests for deterministic node ID generation."""

    def test_basic_filename(
        self, registry: ExtractorRegistry, sample_workspace: dict
    ) -> None:
        """Simple filename produces NODE_{UPPERCASE_NAME}."""
        results = registry.parse_workspace_concurrently([sample_workspace["txt"]])
        assert results[0]["id"] == "NODE_HELLO"

    def test_hyphen_replaced_with_underscore(
        self, registry: ExtractorRegistry, sample_workspace: dict
    ) -> None:
        """Hyphens in filename are replaced with underscores."""
        results = registry.parse_workspace_concurrently([sample_workspace["md"]])
        assert results[0]["id"] == "NODE_MY_README"

    def test_deterministic_across_calls(
        self, registry: ExtractorRegistry, sample_workspace: dict
    ) -> None:
        """Same file produces same ID across multiple calls."""
        results1 = registry.parse_workspace_concurrently([sample_workspace["txt"]])
        results2 = registry.parse_workspace_concurrently([sample_workspace["txt"]])
        assert results1[0]["id"] == results2[0]["id"]

    def test_uppercase_conversion(self, registry: ExtractorRegistry, tmp_path: Path) -> None:
        """Lowercase filenames are converted to uppercase in ID."""
        f = tmp_path / "my_module.py"
        f.write_text("code", encoding="utf-8")
        results = registry.parse_workspace_concurrently([str(f)])
        assert results[0]["id"] == "NODE_MY_MODULE"


class TestZeroLLMTokens:
    """Tests ensuring zero LLM token consumption."""

    def test_no_llm_calls_during_extraction(
        self, registry: ExtractorRegistry, sample_workspace: dict
    ) -> None:
        """Extraction uses only programmatic parsing, no LLM calls.

        This is verified by the fact that ExtractorRegistry has no LLM
        dependency in its constructor or methods — it's purely file I/O.
        """
        # The registry has no LLM provider attribute
        assert not hasattr(registry, "llm")
        assert not hasattr(registry, "llm_provider")

        # Extraction works without any LLM configuration
        results = registry.parse_workspace_concurrently([sample_workspace["txt"]])
        assert len(results) == 1
        assert results[0]["raw_text"] == "Hello, world!"
