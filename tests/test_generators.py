"""Tests for dataset generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from flattune.dataset.generators.base import BaseGenerator, GeneratorOutput
from flattune.dataset.generators.facts import FactsGenerator
from flattune.dataset.generators.glossary import GlossaryGenerator
from flattune.dataset.generators.qa import QAGenerator
from flattune.dataset.generators.summary import SummaryGenerator
from flattune.dataset.generators.classification import ClassificationGenerator
from flattune.dataset.generators.conversation import ConversationGenerator
from flattune.dataset.generators.procedure import ProcedureGenerator
from flattune.dataset.generators.extraction import ExtractionGenerator


@dataclass
class MockConfig:
    """Mock config for testing generators."""
    min_length: int = 10
    max_length: int = 1000


class TestGeneratorOutput:
    """Test GeneratorOutput dataclass."""

    def test_creation(self):
        """Test creating generator output."""
        output = GeneratorOutput(
            instruction="Extract facts",
            input="Some text",
            output="Fact 1\nFact 2",
            metadata={"source": "doc.md"},
        )

        assert output.instruction == "Extract facts"
        assert output.input == "Some text"
        assert output.output == "Fact 1\nFact 2"
        assert output.metadata == {"source": "doc.md"}

    def test_to_dict(self):
        """Test converting to dictionary."""
        output = GeneratorOutput(
            instruction="Test",
            input="Input",
            output="Output",
        )

        d = output.to_dict()

        assert d["instruction"] == "Test"
        assert d["input"] == "Input"
        assert d["output"] == "Output"
        assert d["metadata"] == {}


class TestBaseGenerator:
    """Test BaseGenerator abstract class."""

    def test_supports_with_matching_type(self):
        """Test supports returns True for matching type."""
        class TestGenerator(BaseGenerator):
            name = "test"
            supported_types = ["qa", "facts"]

            def generate(self, document, config, instruction=None):
                return []

        gen = TestGenerator()
        assert gen.supports("qa") is True
        assert gen.supports("facts") is True

    def test_supports_with_empty_types(self):
        """Test supports returns True when no types specified."""
        class TestGenerator(BaseGenerator):
            name = "test"
            supported_types = []  # Empty list, not undefined

            def generate(self, document, config, instruction=None):
                return []

        gen = TestGenerator()
        assert gen.supports("anything") is True

    def test_filter_by_length(self):
        """Test text filtering by length."""
        class TestGenerator(BaseGenerator):
            name = "test"

            def generate(self, document, config, instruction=None):
                return []

        gen = TestGenerator()

        # Too short
        result = gen._filter_by_length("short", min_length=10, max_length=100)
        assert result == ""

        # Too long
        long_text = "x" * 150
        result = gen._filter_by_length(long_text, min_length=10, max_length=100)
        assert result == "x" * 100

        # Just right
        result = gen._filter_by_length("correct length", min_length=5, max_length=20)
        assert result == "correct length"

    def test_extract_field(self):
        """Test field extraction from document."""
        class TestGenerator(BaseGenerator):
            name = "test"

            def generate(self, document, config, instruction=None):
                return []

        gen = TestGenerator()

        doc = {
            "title": "Test Title",
            "body": "Test Body",
            "description": "Test Description",
        }

        # First matching field
        result = gen._extract_field(doc, ["title", "body"])
        assert result == "Test Title"

        # Fallback to second field
        result = gen._extract_field(doc, ["missing", "body"])
        assert result == "Test Body"

        # Returns empty for missing fields
        result = gen._extract_field(doc, ["missing"])
        assert result == ""

    def test_build_sample(self):
        """Test building a sample dictionary."""
        class TestGenerator(BaseGenerator):
            name = "test"

            def generate(self, document, config, instruction=None):
                return []

        gen = TestGenerator()

        sample = gen._build_sample(
            instruction="Test instruction",
            input_text="Test input",
            output="Test output",
            metadata={"key": "value"},
        )

        assert sample["instruction"] == "Test instruction"
        assert sample["input"] == "Test input"
        assert sample["output"] == "Test output"
        assert sample["metadata"] == {"key": "value"}


class TestFactsGenerator:
    """Test FactsGenerator."""

    def test_generate_from_document(self):
        """Test generating facts from a document."""
        config = MockConfig(min_length=10, max_length=1000)
        document = {
            "title": "Python asyncio",
            "body": "Python 3.11 was released in October 2022. "
                    "Python is a popular programming language. "
                    "asyncio provides concurrency tools.",
        }

        generator = FactsGenerator()
        results = generator.generate(document, config)

        assert isinstance(results, list)

    def test_generated_sample_format(self):
        """Test generated sample has correct format."""
        config = MockConfig(min_length=10, max_length=1000)
        document = {
            "body": "Python 3.11 was released in October 2022. "
                    "It includes performance improvements.",
        }

        generator = FactsGenerator()
        results = generator.generate(document, config)

        if results:
            sample = results[0]
            assert "instruction" in sample
            assert "input" in sample
            assert "output" in sample


class TestGlossaryGenerator:
    """Test GlossaryGenerator."""

    def test_generate_from_document(self):
        """Test generating glossary entries."""
        config = MockConfig(min_length=10, max_length=1000)
        document = {
            "body": "Python: A high-level programming language. "
                    "asyncio: A library for asynchronous I/O. "
                    "coroutine: A function that can suspend execution.",
        }

        generator = GlossaryGenerator()
        results = generator.generate(document, config)

        assert isinstance(results, list)


class TestQAGenerator:
    """Test QAGenerator."""

    def test_generate_qa_pairs(self):
        """Test generating QA pairs."""
        config = MockConfig(min_length=10, max_length=1000)
        document = {
            "title": "Python Tutorial",
            "body": "Python is a programming language created by Guido van Rossum. "
                    "It was first released in 1991.",
        }

        generator = QAGenerator()
        results = generator.generate(document, config)

        assert isinstance(results, list)


class TestSummaryGenerator:
    """Test SummaryGenerator."""

    def test_generate_summary(self):
        """Test generating summaries."""
        config = MockConfig(min_length=10, max_length=2000)
        document = {
            "title": "Annual Report",
            "body": "This is a long document. " * 50,
        }

        generator = SummaryGenerator()
        results = generator.generate(document, config)

        assert isinstance(results, list)


class TestClassificationGenerator:
    """Test ClassificationGenerator."""

    def test_generate_classification(self):
        """Test generating classification samples."""
        config = MockConfig(min_length=10, max_length=1000)
        document = {
            "title": "Tech Article",
            "body": "This article discusses Python programming and software development.",
        }

        generator = ClassificationGenerator()
        results = generator.generate(document, config)

        assert isinstance(results, list)


class TestConversationGenerator:
    """Test ConversationGenerator."""

    def test_generate_conversation(self):
        """Test generating conversation samples."""
        config = MockConfig(min_length=10, max_length=1000)
        document = {
            "title": "FAQ Document",
            "body": "Q: How do I install Python? A: Download from python.org. "
                    "Q: Is Python free? A: Yes, Python is open source.",
        }

        generator = ConversationGenerator()
        results = generator.generate(document, config)

        assert isinstance(results, list)


class TestProcedureGenerator:
    """Test ProcedureGenerator."""

    def test_generate_procedure(self):
        """Test generating procedure samples."""
        config = MockConfig(min_length=10, max_length=1000)
        document = {
            "title": "Installation Guide",
            "body": "1. Download Python from python.org\n"
                    "2. Run the installer\n"
                    "3. Add Python to PATH\n"
                    "4. Verify installation with python --version",
        }

        generator = ProcedureGenerator()
        results = generator.generate(document, config)

        assert isinstance(results, list)


class TestExtractionGenerator:
    """Test ExtractionGenerator."""

    def test_generate_extraction(self):
        """Test generating extraction samples."""
        config = MockConfig(min_length=10, max_length=1000)
        document = {
            "body": '{"name": "John", "age": 30, "city": "NYC"}',
        }

        generator = ExtractionGenerator()
        results = generator.generate(document, config)

        assert isinstance(results, list)
