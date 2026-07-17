"""Base generator class."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from flattune.teach.knowledge_graph import KnowledgeNode
from flattune.teach.registry import BaseGenerator

__all__ = ["BaseGenerator", "GeneratedSample"]


@dataclass
class GeneratedSample:
    """A generated training sample with quality metadata."""
    conversation: list[dict[str, str]]
    sample_type: str
    source: str
    quality_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
