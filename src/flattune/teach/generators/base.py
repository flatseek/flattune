"""Base generator class."""

from dataclasses import dataclass, field
from typing import Any

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
