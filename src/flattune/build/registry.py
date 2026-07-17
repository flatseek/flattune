"""Plugin registries for the build pipeline.

This module provides registry-based architecture for:
- Source detection (file types, formats)
- Dataset categories (organizational groupings)
- Dataset types (concrete formats bound to generators)

Reference: flattune.teach.registry for the canonical PluginRegistry pattern.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flattune.teach.registry import SourceType

logger = logging.getLogger(__name__)


# ============================================================================
# Base Registry
# ============================================================================


class PluginRegistry(ABC):
    """Base registry for plugin management."""

    _plugins: dict[str, type] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, plugin_class: type, alias: str | None = None) -> None:
        """Register a plugin.

        Args:
            name: Plugin identifier.
            plugin_class: Plugin class to register.
            alias: Optional alias for the plugin.
        """
        cls._plugins[name] = plugin_class
        if alias:
            cls._aliases[alias] = name
        logger.debug(f"Registered {cls.__name__}: {name}")

    @classmethod
    def get(cls, name: str) -> type | None:
        """Get a plugin by name.

        Args:
            name: Plugin name or alias.

        Returns:
            Plugin class or None if not found.
        """
        name = cls._aliases.get(name, name)
        return cls._plugins.get(name)

    @classmethod
    def create(cls, name: str, **kwargs):
        """Create a plugin instance by name.

        Args:
            name: Plugin name.
            **kwargs: Arguments to pass to the plugin constructor.

        Returns:
            Plugin instance.

        Raises:
            ValueError: If plugin not found.
        """
        plugin_class = cls.get(name)
        if plugin_class is None:
            available = list(cls._plugins.keys())
            raise ValueError(f"Unknown plugin: {name}. Available: {available}")
        return plugin_class(**kwargs)

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered plugin names.

        Returns:
            List of plugin names.
        """
        return list(cls._plugins.keys())


# ============================================================================
# Source Detection
# ============================================================================


@dataclass
class SourceDetectionResult:
    """Result from detecting a source type."""
    source_type: SourceType
    confidence: float  # 0.0 to 1.0
    detected_format: str | None = None  # e.g., "jsonl", "csv", "markdown"
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_content: str | None = None  # First ~1KB for analysis


class BaseSourceDetector(ABC):
    """Detect and handle specific source types."""

    source_type: SourceType

    @abstractmethod
    def detect(self, source: str | Path) -> SourceDetectionResult:
        """Detect source type and metadata.

        Args:
            source: Path or URL to detect.

        Returns:
            SourceDetectionResult with detected type and confidence.
        """
        pass

    @abstractmethod
    def can_handle(self, source: str | Path) -> bool:
        """Check if this detector can handle the source.

        Args:
            source: Path or URL to check.

        Returns:
            True if this detector can handle the source.
        """
        pass


class SourceRegistry(PluginRegistry):
    """Registry for source detector plugins."""

    _plugins: dict[str, type] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def get_for_source(cls, source: str | Path) -> type[BaseSourceDetector] | None:
        """Get appropriate detector for a source.

        Args:
            source: Path or URL to find detector for.

        Returns:
            Detector class that can handle the source, or None.
        """
        source_str = str(source)

        # First try all detectors' can_handle() method
        candidates = []
        for name, detector_class in cls._plugins.items():
            try:
                instance = detector_class()
                if instance.can_handle(source_str):
                    priority = getattr(instance, 'priority', 50)
                    candidates.append((priority, name, detector_class))
            except Exception:
                pass

        # Return highest priority (lowest number) match
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][2]

        # Try by extension
        ext = Path(source_str).suffix.lower()
        ext_map = {
            ".md": "markdown",
            ".markdown": "markdown",
            ".txt": "text",
            ".pdf": "pdf",
            ".html": "html",
            ".htm": "html",
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "jsonl",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".fsk": "flatseek",
            ".openapi": "openapi",
            ".swagger": "swagger",
        }

        if ext in ext_map:
            return cls.get(ext_map[ext])

        return None


def register_source(name: str, alias: str | None = None):
    """Decorator to register a source detector plugin.

    Usage:
        @register_source("markdown")
        class MarkdownSourceDetector(BaseSourceDetector):
            ...
    """
    def decorator(cls: type[BaseSourceDetector]) -> type[BaseSourceDetector]:
        SourceRegistry.register(name, cls, alias)
        return cls
    return decorator


# ============================================================================
# Dataset Categories
# ============================================================================


@dataclass
class DatasetCategory:
    """A dataset category for organizational grouping.

    Categories are organizational only - they do not generate data.
    They group related dataset types together.
    """
    name: str
    description: str
    dataset_types: list[str] = field(default_factory=list)  # Names of associated DatasetType entries
    priority: int = 50  # Lower = higher priority for auto-detection


class CategoryRegistry(PluginRegistry):
    """Registry for dataset category plugins."""

    _plugins: dict[str, DatasetCategory] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, category: DatasetCategory) -> None:
        """Register a category."""
        # Store the instance directly
        cls._plugins[name] = category
        logger.debug(f"Registered category: {name}")

    @classmethod
    def get(cls, name: str) -> DatasetCategory | None:
        """Get a category by name."""
        result = cls._plugins.get(name)
        if result is None:
            return None
        # If it's a class, instantiate it
        if isinstance(result, type):
            return result()
        return result

    @classmethod
    def get_types_for_category(cls, category_name: str) -> list[str]:
        """Get all dataset type names for a category."""
        category = cls.get(category_name)
        if category is None:
            return []
        return category.dataset_types

    @classmethod
    def list_all_categories(cls) -> list[DatasetCategory]:
        """List all registered categories."""
        return [cls.get(name) for name in cls._plugins.keys() if cls.get(name) is not None]


def register_category(name: str):
    """Decorator to register a dataset category.

    Usage:
        @register_category("knowledge")
        class KnowledgeCategory(DatasetCategory):
            name = "knowledge"
            description = "Factual, encyclopedic content"
            dataset_types = ["facts", "glossary", "concept", "definition"]
    """
    def decorator(cls: type[DatasetCategory]) -> type[DatasetCategory]:
        CategoryRegistry.register(name, cls)
        return cls
    return decorator


# ============================================================================
# Dataset Types
# ============================================================================


@dataclass
class DatasetType:
    """A concrete dataset type with generator binding.

    A dataset type represents a specific output format that can be generated.
    It is bound to a specific generator that produces that type.
    """
    name: str
    description: str
    category: str  # Parent category name
    generator_name: str  # Which generator to use
    instruction_template: str = ""  # Jinja2 template for instruction
    supported_sources: list[str] = field(default_factory=list)  # Which sources can produce this type
    estimated_samples_per_doc: float = 2.0  # For planning
    min_content_length: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetTypeRegistry(PluginRegistry):
    """Registry for dataset type plugins."""

    _plugins: dict[str, DatasetType] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, dataset_type: DatasetType) -> None:
        """Register a dataset type."""
        # Store the instance directly
        cls._plugins[name] = dataset_type
        logger.debug(f"Registered dataset type: {name}")

    @classmethod
    def get(cls, name: str) -> DatasetType | None:
        """Get a dataset type by name."""
        result = cls._plugins.get(name)
        if result is None:
            return None
        if isinstance(result, type):
            return result()
        return result

    @classmethod
    def list_by_category(cls, category: str) -> list[str]:
        """List all dataset type names for a category."""
        return [
            name for name in cls._plugins.keys()
            if cls.get(name) and cls.get(name).category == category
        ]

    @classmethod
    def get_generator_for_type(cls, type_name: str) -> str | None:
        """Get the generator name for a dataset type."""
        dt = cls.get(type_name)
        if dt is None:
            return None
        return dt.generator_name

    @classmethod
    def list_all_types(cls) -> list[DatasetType]:
        """List all registered dataset types."""
        return [cls.get(name) for name in cls._plugins.keys() if cls.get(name) is not None]


def register_dataset_type(name: str, category: str, generator: str, **kwargs):
    """Decorator to register a dataset type.

    Usage:
        @register_dataset_type("facts", category="knowledge", generator="facts")
        class FactsType(DatasetType):
            name = "facts"
            description = "Factual statements from content"
            category = "knowledge"
            generator_name = "facts"
            estimated_samples_per_doc = 3.0
    """
    def decorator(cls: type[DatasetType]) -> type[DatasetType]:
        DatasetTypeRegistry.register(name, cls)
        return cls
    return decorator


# ============================================================================
# Built-in Categories and Dataset Types
# ============================================================================


def _register_builtin_categories():
    """Register built-in dataset categories."""
    # Knowledge category
    CategoryRegistry.register("knowledge", DatasetCategory(
        name="knowledge",
        description="Factual, encyclopedic content",
        dataset_types=["facts", "glossary", "concept", "definition"],
        priority=10,
    ))

    # QA category
    CategoryRegistry.register("qa", DatasetCategory(
        name="qa",
        description="Question answering datasets",
        dataset_types=["context_qa", "direct_qa", "yes_no"],
        priority=20,
    ))

    # Conversation category
    CategoryRegistry.register("conversation", DatasetCategory(
        name="conversation",
        description="Dialogue-based content",
        dataset_types=["chat", "dialogue"],
        priority=30,
    ))

    # Instruction category
    CategoryRegistry.register("instruction", DatasetCategory(
        name="instruction",
        description="Step-by-step tasks",
        dataset_types=["procedure", "howto"],
        priority=40,
    ))

    # Classification category
    CategoryRegistry.register("classification", DatasetCategory(
        name="classification",
        description="Categorization tasks",
        dataset_types=["topic_classify", "sentiment", "entity_classify"],
        priority=50,
    ))

    # Summary category
    CategoryRegistry.register("summary", DatasetCategory(
        name="summary",
        description="Summarization tasks",
        dataset_types=["executive_summary", "concise_summary"],
        priority=60,
    ))

    # Extraction category
    CategoryRegistry.register("extraction", DatasetCategory(
        name="extraction",
        description="Information extraction tasks",
        dataset_types=["key_value_extract"],
        priority=70,
    ))

    # RAG category
    CategoryRegistry.register("rag", DatasetCategory(
        name="rag",
        description="RAG-style QA",
        dataset_types=["citation_qa"],
        priority=80,
    ))


def _register_builtin_dataset_types():
    """Register built-in dataset types."""
    # Knowledge types
    DatasetTypeRegistry.register("facts", DatasetType(
        name="facts",
        description="Extract factual statements from content",
        category="knowledge",
        generator_name="facts",
        instruction_template="Extract factual statements from the following content.",
        estimated_samples_per_doc=3.0,
        min_content_length=100,
    ))

    DatasetTypeRegistry.register("glossary", DatasetType(
        name="glossary",
        description="Extract and define terms",
        category="knowledge",
        generator_name="glossary",
        instruction_template="Define the following terms.",
        estimated_samples_per_doc=2.0,
        min_content_length=50,
    ))

    DatasetTypeRegistry.register("concept", DatasetType(
        name="concept",
        description="Explain concepts in detail",
        category="knowledge",
        generator_name="concept",
        instruction_template="Explain this concept.",
        estimated_samples_per_doc=1.5,
        min_content_length=100,
    ))

    DatasetTypeRegistry.register("definition", DatasetType(
        name="definition",
        description="Provide clear definitions",
        category="knowledge",
        generator_name="definition",
        instruction_template="Provide a definition for the following.",
        estimated_samples_per_doc=1.0,
        min_content_length=30,
    ))

    # QA types
    DatasetTypeRegistry.register("context_qa", DatasetType(
        name="context_qa",
        description="Answer questions based on provided context",
        category="qa",
        generator_name="qa",
        instruction_template="Answer based on the given context.",
        estimated_samples_per_doc=4.0,
        min_content_length=200,
    ))

    DatasetTypeRegistry.register("direct_qa", DatasetType(
        name="direct_qa",
        description="Direct question answering without context",
        category="qa",
        generator_name="qa",
        instruction_template="Answer the question directly.",
        estimated_samples_per_doc=2.0,
        min_content_length=50,
    ))

    DatasetTypeRegistry.register("yes_no", DatasetType(
        name="yes_no",
        description="Yes/No question answering",
        category="qa",
        generator_name="qa",
        instruction_template="Answer yes or no.",
        estimated_samples_per_doc=1.0,
        min_content_length=30,
    ))

    # Conversation types
    DatasetTypeRegistry.register("chat", DatasetType(
        name="chat",
        description="Single/multi-turn chat conversations",
        category="conversation",
        generator_name="conversation",
        instruction_template="Continue the conversation naturally.",
        estimated_samples_per_doc=2.0,
        min_content_length=100,
    ))

    DatasetTypeRegistry.register("dialogue", DatasetType(
        name="dialogue",
        description="Roleplay dialogue scenarios",
        category="conversation",
        generator_name="conversation",
        instruction_template="Roleplay this dialogue scenario.",
        estimated_samples_per_doc=2.0,
        min_content_length=100,
    ))

    # Instruction types
    DatasetTypeRegistry.register("procedure", DatasetType(
        name="procedure",
        description="Step-by-step procedural instructions",
        category="instruction",
        generator_name="procedure",
        instruction_template="Explain the steps to accomplish this task.",
        estimated_samples_per_doc=1.5,
        min_content_length=100,
    ))

    DatasetTypeRegistry.register("howto", DatasetType(
        name="howto",
        description="How-to instructional content",
        category="instruction",
        generator_name="howto",
        instruction_template="How do I accomplish this?",
        estimated_samples_per_doc=2.0,
        min_content_length=80,
    ))

    # Classification types
    DatasetTypeRegistry.register("topic_classify", DatasetType(
        name="topic_classify",
        description="Classify content into topics",
        category="classification",
        generator_name="classification",
        instruction_template="Classify the topic of this content.",
        estimated_samples_per_doc=1.0,
        min_content_length=50,
    ))

    DatasetTypeRegistry.register("sentiment", DatasetType(
        name="sentiment",
        description="Determine sentiment of text",
        category="classification",
        generator_name="classification",
        instruction_template="What is the sentiment of this text?",
        estimated_samples_per_doc=1.0,
        min_content_length=30,
    ))

    DatasetTypeRegistry.register("entity_classify", DatasetType(
        name="entity_classify",
        description="Classify named entities",
        category="classification",
        generator_name="classification",
        instruction_template="Identify and classify the entities in this text.",
        estimated_samples_per_doc=2.0,
        min_content_length=100,
    ))

    # Summary types
    DatasetTypeRegistry.register("executive_summary", DatasetType(
        name="executive_summary",
        description="Executive-level summary",
        category="summary",
        generator_name="summary",
        instruction_template="Provide an executive summary.",
        estimated_samples_per_doc=1.0,
        min_content_length=300,
    ))

    DatasetTypeRegistry.register("concise_summary", DatasetType(
        name="concise_summary",
        description="Concise brief summary",
        category="summary",
        generator_name="summary",
        instruction_template="Summarize this concisely.",
        estimated_samples_per_doc=1.0,
        min_content_length=200,
    ))

    # Extraction types
    DatasetTypeRegistry.register("key_value_extract", DatasetType(
        name="key_value_extract",
        description="Extract key-value pairs",
        category="extraction",
        generator_name="extraction",
        instruction_template="Extract key-value pairs from this content.",
        estimated_samples_per_doc=3.0,
        min_content_length=50,
    ))

    # RAG types
    DatasetTypeRegistry.register("citation_qa", DatasetType(
        name="citation_qa",
        description="QA with citation support",
        category="rag",
        generator_name="qa",
        instruction_template="Answer with citations from the context.",
        estimated_samples_per_doc=2.0,
        min_content_length=300,
    ))


# Register built-ins on module import
_register_builtin_categories()
_register_builtin_dataset_types()
