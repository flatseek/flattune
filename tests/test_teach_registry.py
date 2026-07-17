"""Tests for teach registry and plugins."""

from __future__ import annotations

import pytest

from flattune.teach.registry import (
    SourceType,
    ParserRegistry,
    TeacherRegistry,
    GeneratorRegistry,
    BaseParser,
    BaseTeacher,
    BaseGenerator,
    GeneratedSample,
    ParseResult,
)


class TestSourceType:
    """Test SourceType enum."""

    def test_source_types_exist(self):
        """Test all expected source types exist."""
        assert SourceType.MARKDOWN.value == "markdown"
        assert SourceType.TEXT.value == "text"
        assert SourceType.JSON.value == "json"
        assert SourceType.JSONL.value == "jsonl"
        assert SourceType.CSV.value == "csv"
        assert SourceType.YAML.value == "yaml"
        assert SourceType.FLATSEEK.value == "flatseek"
        assert SourceType.OPENAPI.value == "openapi"
        assert SourceType.SQLITE.value == "sqlite"
        assert SourceType.POSTGRESQL.value == "postgresql"
        assert SourceType.MYSQL.value == "mysql"

    def test_source_type_count(self):
        """Test number of source types."""
        # Should have at least 20 source types
        assert len(SourceType) >= 20


class TestParserRegistry:
    """Test ParserRegistry."""

    def test_list_plugins(self):
        """Test listing registered parsers."""
        parsers = ParserRegistry.list_plugins()
        assert isinstance(parsers, list)

    def test_get_parser(self):
        """Test getting a parser by name."""
        # Built-in parsers should be available
        markdown_parser = ParserRegistry.get("markdown")
        assert markdown_parser is not None

    def test_register_parser(self):
        """Test registering a parser."""

        class TestParser(BaseParser):
            source_type = SourceType.TEXT

            def parse(self, source):
                yield ParseResult(
                    source=str(source),
                    source_type=self.source_type,
                    content="test",
                )

            def can_parse(self, source):
                return str(source).endswith(".txt")

        ParserRegistry.register("test_parser", TestParser)
        assert "test_parser" in ParserRegistry.list_plugins()

    def test_get_for_source(self):
        """Test getting parser for a source file."""
        parser_cls = ParserRegistry.get_for_source("/path/to/file.md")
        assert parser_cls is not None


class TestTeacherRegistry:
    """Test TeacherRegistry."""

    def test_list_plugins(self):
        """Test listing registered teachers."""
        teachers = TeacherRegistry.list_plugins()
        assert isinstance(teachers, list)

    def test_register_teacher(self):
        """Test registering a teacher."""

        class TestTeacher(BaseTeacher):
            name = "test_teacher"
            supports_local = True
            supports_remote = False

            def generate(self, knowledge, sample_type, **kwargs):
                return {}

            def generate_batch(self, knowledge_iter, sample_type, **kwargs):
                yield {}

        TeacherRegistry.register("test_teacher", TestTeacher)
        assert "test_teacher" in TeacherRegistry.list_plugins()

    def test_list_local_teachers(self):
        """Test listing local teachers."""
        local_teachers = TeacherRegistry.list_local_teachers()
        assert isinstance(local_teachers, list)

    def test_list_remote_teachers(self):
        """Test listing remote teachers."""
        remote_teachers = TeacherRegistry.list_remote_teachers()
        assert isinstance(remote_teachers, list)


class TestGeneratorRegistry:
    """Test GeneratorRegistry."""

    def test_list_plugins(self):
        """Test listing registered generators."""
        generators = GeneratorRegistry.list_plugins()
        assert isinstance(generators, list)

    def test_register_generator(self):
        """Test registering a generator."""

        class TestGenerator(BaseGenerator):
            name = "test_generator"
            supported_types = ["qa", "facts"]

            def generate(self, knowledge, **kwargs):
                yield GeneratedSample(
                    conversation=[{"role": "user", "content": "test"}],
                    sample_type="test",
                    source="test",
                )

        GeneratorRegistry.register("test_gen", TestGenerator)
        assert "test_gen" in GeneratorRegistry.list_plugins()

    def test_get_for_type(self):
        """Test getting generator for a sample type."""
        generator_cls = GeneratorRegistry.get_for_type("qa")
        assert generator_cls is not None


class TestParseResult:
    """Test ParseResult dataclass."""

    def test_creation(self):
        """Test creating a parse result."""
        result = ParseResult(
            source="/path/to/file.md",
            source_type=SourceType.MARKDOWN,
            content="# Title\n\nContent",
            metadata={"line_count": 2},
            entities=[{"type": "concept", "name": "Python"}],
            relationships=[("Python", "is", "language")],
        )

        assert result.source == "/path/to/file.md"
        assert result.source_type == SourceType.MARKDOWN
        assert "Title" in result.content
        assert result.metadata == {"line_count": 2}
        assert len(result.entities) == 1
        assert len(result.relationships) == 1

    def test_defaults(self):
        """Test default values."""
        result = ParseResult(
            source="/path",
            source_type=SourceType.TEXT,
            content="content",
        )

        assert result.metadata == {}
        assert result.entities == []
        assert result.relationships == []


class TestGeneratedSample:
    """Test GeneratedSample dataclass."""

    def test_creation(self):
        """Test creating a generated sample."""
        sample = GeneratedSample(
            conversation=[
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
            sample_type="qa",
            source="test.md",
            quality_score=0.9,
            metadata={"tokens": 50},
            warnings=[],
        )

        assert len(sample.conversation) == 2
        assert sample.sample_type == "qa"
        assert sample.quality_score == 0.9

    def test_defaults(self):
        """Test default values."""
        sample = GeneratedSample(
            conversation=[{"role": "user", "content": "test"}],
            sample_type="test",
            source="test",
        )

        assert sample.quality_score == 1.0
        assert sample.metadata == {}
        assert sample.warnings == []
