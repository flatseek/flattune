"""Plugin registries for parsers, teachers, and generators.

This module provides a plugin-based architecture where:
- Parsers read and normalize data from various sources
- Teachers generate high-quality conversations using LLMs
- Generators produce datasets from knowledge graphs
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Type

logger = logging.getLogger(__name__)


# ============================================================================
# Base Plugin Interfaces
# ============================================================================


class SourceType(Enum):
    """Supported source types for teaching."""
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    YAML = "yaml"
    FLATSEEK = "flatseek"
    OPENAPI = "openapi"
    SWAGGER = "swagger"
    GRAPHQL = "graphql"
    MCP = "mcp"
    DATABASE = "database"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    SQLALCHEMY = "sqlalchemy"
    CLI = "cli"
    PYTHON = "python"
    WORKFLOW = "workflow"
    DIRECTORY = "directory"


@dataclass
class ParseResult:
    """Result from a parser containing extracted knowledge."""
    source: str
    source_type: SourceType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[tuple[str, str, str]] = field(default_factory=list)  # (subject, predicate, object)


class BaseParser(ABC):
    """Abstract base class for all parsers.

    Parsers are responsible for:
    - Reading data from a specific source type
    - Normalizing the data into a consistent format
    - Extracting entities and relationships when applicable
    - Never generating conversations (that's the teacher's job)
    """

    source_type: SourceType

    @abstractmethod
    def parse(self, source: str | Path) -> Iterator[ParseResult]:
        """Parse a source and yield knowledge fragments.

        Args:
            source: Path or URL to the source material.

        Yields:
            ParseResult objects containing extracted knowledge.
        """
        pass

    @abstractmethod
    def can_parse(self, source: str | Path) -> bool:
        """Check if this parser can handle the given source.

        Args:
            source: Path or URL to check.

        Returns:
            True if this parser can handle the source.
        """
        pass


class BaseTeacher(ABC):
    """Abstract base class for teacher models.

    Teachers are responsible for:
    - Generating high-quality conversations from knowledge
    - Maintaining conversation quality and consistency
    - Supporting both local and remote LLM providers
    """

    name: str
    supports_local: bool = False
    supports_remote: bool = False

    @abstractmethod
    def generate(
        self,
        knowledge: str | KnowledgeNode,
        sample_type: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a training sample from knowledge.

        Args:
            knowledge: Knowledge to generate from.
            sample_type: Type of sample (qa, summarization, tool_call, etc.)
            **kwargs: Additional generation parameters.

        Returns:
            Dictionary containing generated conversation.
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        knowledge_iter: Iterator[KnowledgeNode],
        sample_type: str,
        **kwargs,
    ) -> Iterator[dict[str, Any]]:
        """Generate multiple training samples.

        Args:
            knowledge_iter: Iterator of knowledge nodes.
            sample_type: Type of sample to generate.
            **kwargs: Additional generation parameters.

        Yields:
            Generated conversation dictionaries.
        """
        pass

    def supports(self, sample_type: str) -> bool:
        """Check if teacher supports a sample type.

        Args:
            sample_type: Type of sample to check.

        Returns:
            True if supported.
        """
        return True


@dataclass
class GeneratedSample:
    """A generated training sample with quality metadata."""
    conversation: list[dict[str, str]]
    sample_type: str
    source: str
    quality_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class BaseGenerator(ABC):
    """Abstract base class for dataset generators.

    Generators are responsible for:
    - Creating specific types of training samples from knowledge
    - Implementing diverse generation strategies
    - Producing multiple samples per knowledge fragment
    """

    name: str = "base"
    supported_types: list[str] = field(default_factory=list)

    def __init__(self, teacher: Any = None):
        """Initialize generator.

        Args:
            teacher: Teacher model for generation.
        """
        self.teacher = teacher

    @abstractmethod
    def generate(
        self,
        knowledge: str | KnowledgeNode,
        **kwargs,
    ) -> Iterator[GeneratedSample]:
        """Generate training samples from knowledge.

        Args:
            knowledge: Knowledge to generate from.
            **kwargs: Generation parameters.

        Yields:
            GeneratedSample objects.
        """
        pass

    def supports(self, sample_type: str) -> bool:
        """Check if generator supports a sample type.

        Args:
            sample_type: Type of sample to check.

        Returns:
            True if supported.
        """
        return sample_type in self.supported_types or not self.supported_types


# ============================================================================
# Registries
# ============================================================================


class PluginRegistry(ABC):
    """Base registry for plugin management."""

    _plugins: dict[str, Type] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, plugin_class: Type, alias: Optional[str] = None) -> None:
        """Register a plugin.

        Args:
            name: Plugin identifier.
            plugin_class: Plugin class to register.
            alias: Optional alias for the plugin.
        """
        # Note: Type check deferred to avoid issues with decorators
        # The subclass check is done lazily when the plugin is used
        cls._plugins[name] = plugin_class
        if alias:
            cls._aliases[alias] = name
        logger.debug(f"Registered {cls._base_type.__name__}: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
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
            raise ValueError(f"Unknown {cls._base_type.__name__}: {name}. Available: {available}")
        return plugin_class(**kwargs)

    @classmethod
    def list_plugins(cls) -> list[str]:
        """List all registered plugin names.

        Returns:
            List of plugin names.
        """
        return list(cls._plugins.keys())

    @classmethod
    def _get_base_type(cls) -> type:
        """Get the base type for this registry."""
        return cls._base_type


class ParserRegistry(PluginRegistry):
    """Registry for parser plugins."""

    _base_type = BaseParser
    _plugins: dict[str, Type] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, plugin_class: Type, alias: Optional[str] = None) -> None:
        """Register a parser plugin."""
        super().register(name, plugin_class, alias)

    @classmethod
    def get_for_source(cls, source: str | Path) -> Optional[Type[BaseParser]]:
        """Get appropriate parser for a source.

        Args:
            source: Path or URL to find parser for.

        Returns:
            Parser class that can handle the source, or None.
        """
        source_str = str(source)

        # First try all parsers' can_parse() method
        candidates = []
        for name, parser_class in cls._plugins.items():
            try:
                instance = parser_class()
                if instance.can_parse(source_str):
                    # Use priority if available, default to 50
                    priority = getattr(instance, 'priority', 50)
                    candidates.append((priority, name, parser_class))
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
        }

        if ext in ext_map:
            return cls.get(ext_map[ext])

        return None


class TeacherRegistry(PluginRegistry):
    """Registry for teacher model plugins."""

    _base_type = BaseTeacher
    _plugins: dict[str, Type] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, plugin_class: Type, alias: Optional[str] = None) -> None:
        """Register a teacher plugin."""
        super().register(name, plugin_class, alias)

    @classmethod
    def list_local_teachers(cls) -> list[str]:
        """List teachers that support local inference."""
        return [
            name for name, cls_type in cls._plugins.items()
            if getattr(cls_type, "supports_local", False)
        ]

    @classmethod
    def list_remote_teachers(cls) -> list[str]:
        """List teachers that support remote inference."""
        return [
            name for name, cls_type in cls._plugins.items()
            if getattr(cls_type, "supports_remote", False)
        ]


class GeneratorRegistry(PluginRegistry):
    """Registry for generator plugins."""

    _base_type = BaseGenerator
    _plugins: dict[str, Type] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def register(cls, name: str, plugin_class: Type, alias: Optional[str] = None) -> None:
        """Register a generator plugin."""
        super().register(name, plugin_class, alias)

    @classmethod
    def get_for_type(cls, sample_type: str) -> Optional[Type[BaseGenerator]]:
        """Get generator that supports a sample type.

        Args:
            sample_type: Type of sample to generate.

        Returns:
            Generator class or None.
        """
        for name, gen_class in cls._plugins.items():
            try:
                instance = gen_class()
                if instance.supports(sample_type):
                    return gen_class
            except Exception:
                pass
        return None


# ============================================================================
# Registration Decorators
# ============================================================================


def register_parser(name: str, alias: Optional[str] = None):
    """Decorator to register a parser plugin.

    Usage:
        @register_parser("markdown")
        class MarkdownParser(BaseParser):
            ...
    """
    def decorator(cls: Type[BaseParser]) -> Type[BaseParser]:
        ParserRegistry.register(name, cls, alias)
        return cls
    return decorator


def register_teacher(name: str, alias: Optional[str] = None):
    """Decorator to register a teacher plugin.

    Usage:
        @register_teacher("openai")
        class OpenAITeacher(BaseTeacher):
            ...
    """
    def decorator(cls: Type[BaseTeacher]) -> Type[BaseTeacher]:
        TeacherRegistry.register(name, cls, alias)
        return cls
    return decorator


def register_generator(name: str, alias: Optional[str] = None):
    """Decorator to register a generator plugin.

    Usage:
        @register_generator("qa")
        class QAGenerator(BaseGenerator):
            ...
    """
    def decorator(cls: Type[BaseGenerator]) -> Type[BaseGenerator]:
        GeneratorRegistry.register(name, cls, alias)
        return cls
    return decorator


# Import KnowledgeNode for type hints
from flattune.teach.knowledge_graph import KnowledgeNode
