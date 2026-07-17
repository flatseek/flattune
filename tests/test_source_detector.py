"""Tests for source detection."""

from __future__ import annotations

import pytest
from pathlib import Path

from flattune.build.source import SourceDetector
from flattune.build.registry import SourceDetectionResult
from flattune.teach.registry import SourceType


class TestSourceDetector:
    """Test SourceDetector class."""

    def test_detect_markdown_file(self, markdown_file: Path):
        """Test detecting markdown files."""
        detector = SourceDetector()
        result = detector.detect(str(markdown_file))

        assert result.source_type == SourceType.MARKDOWN
        assert result.confidence >= 0.8

    def test_detect_json_file(self, json_file: Path):
        """Test detecting JSON files."""
        detector = SourceDetector()
        result = detector.detect(str(json_file))

        assert result.source_type == SourceType.JSON
        assert result.confidence >= 0.8

    def test_detect_jsonl_file(self, jsonl_file: Path):
        """Test detecting JSONL files."""
        detector = SourceDetector()
        result = detector.detect(str(jsonl_file))

        assert result.source_type == SourceType.JSONL
        assert result.confidence >= 0.8

    def test_detect_openapi_file(self, openapi_file: Path):
        """Test detecting OpenAPI files."""
        detector = SourceDetector()
        result = detector.detect(str(openapi_file))

        # OpenAPI files may be detected as JSON or OPENAPI depending on content
        assert result.source_type in [SourceType.JSON, SourceType.OPENAPI]

    def test_detect_sql_file(self, sql_file: Path):
        """Test detecting SQL files by extension."""
        detector = SourceDetector()
        result = detector.detect(str(sql_file))

        # .sql extension maps to TEXT type
        assert result.source_type == SourceType.TEXT

    def test_detect_http_url(self):
        """Test detecting HTTP URLs."""
        detector = SourceDetector()
        result = detector.detect("https://api.example.com/openapi.json")

        assert result.confidence >= 0.8
        # May be detected as JSON if file exists and is parsed, or OPENAPI if URL-based
        assert result.source_type in [SourceType.JSON, SourceType.OPENAPI]

    def test_detect_github_url(self):
        """Test detecting GitHub URLs."""
        detector = SourceDetector()
        result = detector.detect("https://github.com/user/repo/blob/main/schema.json")

        assert result.source_type == SourceType.JSON
        assert result.confidence >= 0.7

    def test_detect_unknown_extension(self, temp_dir: Path):
        """Test detecting files with unknown extensions."""
        unknown_file = temp_dir / "file.unknown"
        unknown_file.write_text("some content")

        detector = SourceDetector()
        result = detector.detect(str(unknown_file))

        assert result.source_type == SourceType.TEXT
        assert result.confidence < 1.0

    def test_detect_nonexistent_file(self):
        """Test detecting non-existent file."""
        detector = SourceDetector()
        result = detector.detect("/nonexistent/path/file.md")

        # Should try extension-based detection
        assert result.source_type == SourceType.MARKDOWN

    def test_detect_multiple(self, markdown_file: Path, json_file: Path):
        """Test detecting multiple sources."""
        detector = SourceDetector()
        results = detector.detect_multiple([str(markdown_file), str(json_file)])

        assert len(results) == 2
        assert results[0].source_type == SourceType.MARKDOWN
        assert results[1].source_type == SourceType.JSON

    def test_detect_from_directory(self, temp_dir: Path, markdown_file: Path, json_file: Path):
        """Test detecting all files in a directory."""
        detector = SourceDetector()
        results = detector.detect_from_directory(str(temp_dir))

        # Should find both markdown and json files
        source_types = [r.source_type for r in results]
        assert SourceType.MARKDOWN in source_types
        assert SourceType.JSON in source_types

    def test_detect_from_directory_skips_hidden(self, temp_dir: Path):
        """Test that hidden files are skipped."""
        # Create hidden file
        hidden_file = temp_dir / ".hidden.md"
        hidden_file.write_text("# Hidden")

        # Create normal file
        normal_file = temp_dir / "visible.md"
        normal_file.write_text("# Visible")

        detector = SourceDetector()
        results = detector.detect_from_directory(str(temp_dir))

        source_types = [r.source_type for r in results]
        assert SourceType.MARKDOWN in source_types
        # Hidden file should be skipped
        assert len(results) == 1

    def test_detect_extension_map(self):
        """Test extension mapping."""
        detector = SourceDetector()

        # Test various extensions
        assert detector.EXTENSION_MAP[".md"] == SourceType.MARKDOWN
        assert detector.EXTENSION_MAP[".json"] == SourceType.JSON
        assert detector.EXTENSION_MAP[".jsonl"] == SourceType.JSONL
        assert detector.EXTENSION_MAP[".fsk"] == SourceType.FLATSEEK
        assert detector.EXTENSION_MAP[".yaml"] == SourceType.YAML
        assert detector.EXTENSION_MAP[".yml"] == SourceType.YAML


class TestSourceDetectionResult:
    """Test SourceDetectionResult dataclass."""

    def test_creation(self):
        """Test creating a detection result."""
        result = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.9,
            detected_format="markdown",
            metadata={"key": "value"},
        )

        assert result.source_type == SourceType.MARKDOWN
        assert result.confidence == 0.9
        assert result.detected_format == "markdown"
        assert result.metadata == {"key": "value"}

    def test_defaults(self):
        """Test default values."""
        result = SourceDetectionResult(
            source_type=SourceType.JSON,
            confidence=0.8,
        )

        assert result.detected_format is None
        assert result.metadata == {}
        assert result.sample_content is None
