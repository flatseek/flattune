"""Tests for build pipeline registries."""

from __future__ import annotations

import pytest

from flattune.build.registry import (
    CategoryRegistry,
    DatasetTypeRegistry,
    SourceRegistry,
    SourceDetectionResult,
    DatasetType,
    DatasetCategory,
    PluginRegistry,
)
from flattune.teach.registry import SourceType


class TestSourceRegistry:
    """Test SourceRegistry functionality."""

    def test_register_source(self):
        """Test registering a source detector."""
        from flattune.build.registry import BaseSourceDetector

        class MockSourceDetector(BaseSourceDetector):
            source_type = SourceType.MARKDOWN

            def detect(self, source):
                return SourceDetectionResult(source_type=self.source_type, confidence=0.9)

            def can_handle(self, source):
                return str(source).endswith(".md")

        SourceRegistry.register("test_source", MockSourceDetector)
        assert "test_source" in SourceRegistry.list_all()

    def test_get_source(self):
        """Test getting a registered source."""
        result = SourceRegistry.get("markdown")
        assert result is not None

    def test_list_all(self):
        """Test listing all registered sources."""
        sources = SourceRegistry.list_all()
        assert isinstance(sources, list)
        # Built-in sources should be registered
        assert "markdown" in sources or "json" in sources

    def test_get_for_source_by_extension(self):
        """Test getting detector by file extension."""
        detector_cls = SourceRegistry.get_for_source("/path/to/file.md")
        assert detector_cls is not None


class TestCategoryRegistry:
    """Test CategoryRegistry functionality."""

    def test_register_category(self):
        """Test registering a category."""
        category = DatasetCategory(
            name="test_category",
            description="Test category",
            dataset_types=["type1", "type2"],
            priority=10,
        )
        CategoryRegistry.register("test_cat", category)

        retrieved = CategoryRegistry.get("test_cat")
        assert retrieved is not None
        assert retrieved.name == "test_category"

    def test_get_category(self):
        """Test getting a registered category."""
        # Built-in categories should be registered
        knowledge = CategoryRegistry.get("knowledge")
        assert knowledge is not None
        assert knowledge.name == "knowledge"

    def test_list_all_categories(self):
        """Test listing all categories."""
        categories = CategoryRegistry.list_all_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0

    def test_get_types_for_category(self):
        """Test getting types for a category."""
        types = CategoryRegistry.get_types_for_category("knowledge")
        assert isinstance(types, list)
        assert "facts" in types
        assert "glossary" in types


class TestDatasetTypeRegistry:
    """Test DatasetTypeRegistry functionality."""

    def test_register_dataset_type(self):
        """Test registering a dataset type."""
        dtype = DatasetType(
            name="test_type",
            description="Test type",
            category="test_category",
            generator_name="test_generator",
            estimated_samples_per_doc=2.0,
        )
        DatasetTypeRegistry.register("test_dtype", dtype)

        retrieved = DatasetTypeRegistry.get("test_dtype")
        assert retrieved is not None
        assert retrieved.name == "test_type"

    def test_get_dataset_type(self):
        """Test getting a registered dataset type."""
        # Built-in types should be registered
        facts = DatasetTypeRegistry.get("facts")
        assert facts is not None
        assert facts.name == "facts"
        assert facts.category == "knowledge"

    def test_list_all_types(self):
        """Test listing all dataset types."""
        types = DatasetTypeRegistry.list_all_types()
        assert isinstance(types, list)
        assert len(types) >= 17  # 17 built-in types

    def test_list_by_category(self):
        """Test listing types by category."""
        qa_types = DatasetTypeRegistry.list_by_category("qa")
        assert isinstance(qa_types, list)
        assert "context_qa" in qa_types
        assert "direct_qa" in qa_types
        assert "yes_no" in qa_types

    def test_get_generator_for_type(self):
        """Test getting generator name for a type."""
        gen_name = DatasetTypeRegistry.get_generator_for_type("facts")
        assert gen_name == "facts"

        gen_name = DatasetTypeRegistry.get_generator_for_type("context_qa")
        assert gen_name == "qa"


class TestBuiltInTypes:
    """Test built-in dataset types."""

    def test_knowledge_types(self):
        """Test knowledge category types."""
        for type_name in ["facts", "glossary", "concept", "definition"]:
            dtype = DatasetTypeRegistry.get(type_name)
            assert dtype is not None
            assert dtype.category == "knowledge"

    def test_qa_types(self):
        """Test QA category types."""
        for type_name in ["context_qa", "direct_qa", "yes_no"]:
            dtype = DatasetTypeRegistry.get(type_name)
            assert dtype is not None
            assert dtype.category == "qa"

    def test_conversation_types(self):
        """Test conversation category types."""
        for type_name in ["chat", "dialogue"]:
            dtype = DatasetTypeRegistry.get(type_name)
            assert dtype is not None
            assert dtype.category == "conversation"

    def test_instruction_types(self):
        """Test instruction category types."""
        for type_name in ["procedure", "howto"]:
            dtype = DatasetTypeRegistry.get(type_name)
            assert dtype is not None
            assert dtype.category == "instruction"

    def test_classification_types(self):
        """Test classification category types."""
        for type_name in ["topic_classify", "sentiment", "entity_classify"]:
            dtype = DatasetTypeRegistry.get(type_name)
            assert dtype is not None
            assert dtype.category == "classification"

    def test_summary_types(self):
        """Test summary category types."""
        for type_name in ["executive_summary", "concise_summary"]:
            dtype = DatasetTypeRegistry.get(type_name)
            assert dtype is not None
            assert dtype.category == "summary"

    def test_extraction_types(self):
        """Test extraction category types."""
        dtype = DatasetTypeRegistry.get("key_value_extract")
        assert dtype is not None
        assert dtype.category == "extraction"

    def test_rag_types(self):
        """Test RAG category types."""
        dtype = DatasetTypeRegistry.get("citation_qa")
        assert dtype is not None
        assert dtype.category == "rag"

    def test_estimated_samples(self):
        """Test estimated samples per document."""
        facts = DatasetTypeRegistry.get("facts")
        assert facts.estimated_samples_per_doc == 3.0

        context_qa = DatasetTypeRegistry.get("context_qa")
        assert context_qa.estimated_samples_per_doc == 4.0

    def test_min_content_length(self):
        """Test minimum content length requirements."""
        definition = DatasetTypeRegistry.get("definition")
        assert definition.min_content_length == 30

        executive_summary = DatasetTypeRegistry.get("executive_summary")
        assert executive_summary.min_content_length == 300
