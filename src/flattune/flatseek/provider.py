"""Base provider interface for FlatSeek integration."""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional


class FlatseekProvider(ABC):
    """Abstract base class for FlatSeek data providers.

    This defines the common interface that both library and CLI providers
    must implement, ensuring downstream components remain agnostic to
    the integration mode.
    """

    @abstractmethod
    def columns(self) -> list[str]:
        """Return available columns/fields in the index.

        Returns:
            List of column names available for querying.
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Search the index with a query.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            List of matching documents as dictionaries.
        """
        pass

    @abstractmethod
    def aggregate(self, query: str, aggs: dict[str, str]) -> dict[str, Any]:
        """Perform aggregation queries on the index.

        Args:
            query: The search query string.
            aggs: Dictionary of aggregation definitions.

        Returns:
            Dictionary of aggregation results.
        """
        pass

    @abstractmethod
    def export(self, query: str) -> Iterator[dict[str, Any]]:
        """Export documents matching the query.

        Args:
            query: The search query string.

        Yields:
            Matching documents as dictionaries.
        """
        pass

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return statistics about the index.

        Returns:
            Dictionary containing index statistics.
        """
        pass

    @abstractmethod
    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        """Stream documents matching the query without loading all into memory.

        Args:
            query: The search query string.

        Yields:
            Matching documents as dictionaries.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
