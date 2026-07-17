"""Tests for content analysis."""

from __future__ import annotations

import pytest

from flattune.build.analyzer import ContentAnalyzer, AnalysisResult
from flattune.build.registry import SourceDetectionResult
from flattune.build.source import SourceDetector
from flattune.teach.registry import SourceType


class TestContentAnalyzer:
    """Test ContentAnalyzer class."""

    def test_analyze_markdown(self, sample_markdown_content: str):
        """Test analyzing markdown content."""
        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
            detected_format="markdown",
        )

        result = analyzer.analyze(detection, sample_markdown_content)

        assert isinstance(result, AnalysisResult)
        assert result.source_type == SourceType.MARKDOWN
        assert len(result.suggested_categories) > 0
        assert len(result.suggested_types) > 0

    def test_analyze_json(self, sample_json_content: str):
        """Test analyzing JSON content."""
        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.JSON,
            confidence=0.8,
            detected_format="json",
        )

        result = analyzer.analyze(detection, sample_json_content)

        assert isinstance(result, AnalysisResult)
        assert "extraction" in [c[0] for c in result.suggested_categories]

    def test_analyze_empty_content(self):
        """Test analyzing empty content."""
        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.TEXT,
            confidence=0.5,
        )

        result = analyzer.analyze(detection, "")

        assert isinstance(result, AnalysisResult)
        assert result.content_statistics["total_characters"] == 0

    def test_suggested_types_include_facts(self, sample_markdown_content: str):
        """Test that facts type is suggested for factual content."""
        # Content with more numbers should trigger facts
        content = sample_markdown_content + " Numbers: 1 2 3 4 5 6 7 8 9 10"
        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
        )

        result = analyzer.analyze(detection, content)

        type_names = [t[0] for t in result.suggested_types]
        assert "facts" in type_names

    def test_suggested_types_include_procedure(self, sample_markdown_content: str):
        """Test that procedure type is suggested for instructional content."""
        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
        )

        result = analyzer.analyze(detection, sample_markdown_content)

        type_names = [t[0] for t in result.suggested_types]
        assert "procedure" in type_names

    def test_content_statistics_calculation(self):
        """Test content statistics are calculated correctly."""
        content = """
        # Title

        This is a paragraph with **bold** text.

        1. First item
        2. Second item

        What is Python? It's a programming language.

        Key concept: This is defined as something important.
        """

        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
        )

        result = analyzer.analyze(detection, content)
        stats = result.content_statistics

        assert stats["total_characters"] > 0
        assert stats["has_code_blocks"] is False
        assert stats["has_questions"] is True
        assert stats["has_definitions"] is True
        assert stats["has_list_items"] is True

    def test_code_blocks_detection(self):
        """Test detection of code blocks."""
        content = """
        Here's some code:

        ```python
        def hello():
            print("Hello, world!")
        ```

        More text.
        """

        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
        )

        result = analyzer.analyze(detection, content)

        assert result.content_statistics["has_code_blocks"] is True

    def test_numbers_detection(self):
        """Test detection of numbers (potential facts)."""
        # More than 5 numbers needed for detection
        content = "Python 3.11 1 2 3 4 5 6 7 8 was released in October 2022. Version 3.12 came out in October 2023."

        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.TEXT,
            confidence=0.5,
        )

        result = analyzer.analyze(detection, content)

        assert result.content_statistics["has_numbers"] is True

    def test_structured_fields_detection(self):
        """Test detection of structured fields (JSON-like)."""
        content = '{"name": "John", "age": 30, "city": "NYC"}'

        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.JSON,
            confidence=0.8,
        )

        result = analyzer.analyze(detection, content)

        assert result.content_statistics["has_structured_fields"] is True
        assert "name" in result.content_statistics["field_names"]

    def test_summary_generation(self):
        """Test content summary generation."""
        content = "x" * 1500  # Long content

        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
        )

        result = analyzer.analyze(detection, content)

        assert "substantial" in result.content_summary.lower()
        assert "markdown" in result.content_summary.lower()

    def test_category_scoring(self):
        """Test category scoring."""
        # Content with questions should score high for QA
        content = """
        What is Python?

        How do I install it?

        Why should I use it?

        Here are some facts: Python is a language. It is popular.
        """

        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
        )

        result = analyzer.analyze(detection, content)

        category_names = [c[0] for c in result.suggested_categories]
        assert "qa" in category_names

    def test_type_scoring_sorted(self):
        """Test that types are sorted by score descending."""
        content = "Test content " * 50
        analyzer = ContentAnalyzer()
        detection = SourceDetectionResult(
            source_type=SourceType.MARKDOWN,
            confidence=0.8,
        )

        result = analyzer.analyze(detection, content)

        scores = [t[1] for t in result.suggested_types]
        assert scores == sorted(scores, reverse=True)


class TestAnalysisResult:
    """Test AnalysisResult dataclass."""

    def test_creation(self):
        """Test creating an analysis result."""
        result = AnalysisResult(
            source_type=SourceType.MARKDOWN,
            content_summary="Markdown document",
            suggested_categories=[("qa", 0.8), ("knowledge", 0.6)],
            suggested_types=[("context_qa", 0.7), ("facts", 0.5)],
            content_statistics={"total_characters": 100},
        )

        assert result.source_type == SourceType.MARKDOWN
        assert result.content_summary == "Markdown document"
        assert len(result.suggested_categories) == 2
        assert len(result.suggested_types) == 2
        assert result.content_statistics["total_characters"] == 100

    def test_statistics_default_values(self):
        """Test default statistics values."""
        result = AnalysisResult(
            source_type=SourceType.TEXT,
            content_summary="Text",
            suggested_categories=[],
            suggested_types=[],
            content_statistics={},
        )

        stats = result.content_statistics
        # Empty stats - the _calculate_statistics would set these
        # This test just verifies the field exists
        assert isinstance(stats, dict)
