"""Base parser class."""

from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flattune.teach.registry import BaseParser, SourceType


@dataclass
class ParseResult:
    """Result from a parser containing extracted knowledge."""

    source: str
    source_type: SourceType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[tuple[str, str, str]] = field(default_factory=list)


class BaseParser(BaseParser):
    """Abstract base class for parsers."""

    source_type: SourceType

    @abstractmethod
    def parse(self, source: str | Path) -> Iterator[ParseResult]:
        """Parse a source and yield knowledge fragments."""
        pass

    @abstractmethod
    def can_parse(self, source: str | Path) -> bool:
        """Check if this parser can handle the source."""
        pass
