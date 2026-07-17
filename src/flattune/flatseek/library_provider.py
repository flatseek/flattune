"""Library-based FlatSeek provider using the FlatSeek Python API."""

from pathlib import Path
from typing import Any, Iterator, Optional

from flattune.flatseek.provider import FlatseekProvider


class LibraryProvider(FlatseekProvider):
    """FlatSeek provider using the embedded FlatSeek Python library.

    This is the preferred integration mode when FlatSeek is installed
    as a Python package, providing maximum performance and direct API access.

    Supports:
    - Local directory indexes
    - Local .fsk/.flatseek packed archive files
    - Remote HTTP/HTTPS .fsk files (via Range requests)
    - Encrypted .fsk files via encryption key
    """

    # Default batch size for export/stream operations
    _EXPORT_BATCH_SIZE = 1000

    def __init__(
        self,
        path: str,
        query: Optional[str] = None,
        encryption_key: Optional[str] = None,
    ):
        """Initialize the library provider.

        Args:
            path: Path to the FlatSeek index (directory, .fsk file, or HTTP URL).
            query: Default query to apply to all operations.
            encryption_key: Optional encryption key for encrypted .fsk files.
        """
        self.path = path
        self.default_query = query or "*"
        self.encryption_key = encryption_key

        # Import FlatSeek here to allow graceful fallback if not installed
        try:
            from flatseek.core.query_engine import QueryEngine
            from flatseek.flatseek_file import FlatseekFileStorageAdapter
        except ImportError:
            raise ImportError(
                "FlatSeek Python library is not installed. "
                "Install it with: pip install flatseek "
                "Or use CLI mode with: flattune --mode cli"
            )

        self._engine = self._create_engine(path, encryption_key)

    def _create_engine(self, path: str, encryption_key: Optional[str]):
        """Create a QueryEngine for the given path.

        Args:
            path: Path to the FlatSeek index (directory, .fsk file, or HTTP URL).
            encryption_key: Optional encryption key for encrypted .fsk files.

        Returns:
            A QueryEngine instance.
        """
        from flatseek.core.query_engine import QueryEngine
        from flatseek.flatseek_file import FlatseekFileStorageAdapter

        is_http_url = isinstance(path, str) and path.startswith(("http://", "https://"))

        # Determine if path is a packed archive (.fsk or .flatseek)
        if is_http_url:
            is_packed_archive = path.endswith((".fsk", ".flatseek"))
        else:
            path_obj = Path(path)
            is_packed_archive = path_obj.suffix in (".fsk", ".flatseek")

        if is_packed_archive:
            # For .fsk files (local or remote via HTTP), use FlatseekFileStorageAdapter
            adapter = FlatseekFileStorageAdapter(
                path,
                enc_key=encryption_key.encode() if encryption_key else None,
            )
            # QueryEngine with storage adapter for packed archives
            return QueryEngine(path, storage=adapter)
        else:
            # For directory indexes, use QueryEngine directly
            return QueryEngine(path)

    def columns(self) -> list[str]:
        """Return available columns/fields in the index.

        Returns:
            List of column names available for querying.
        """
        cols = self._engine.columns()
        if isinstance(cols, dict):
            return list(cols.keys())
        return cols if isinstance(cols, list) else []

    def search(self, query: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Search the index with a query.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            List of matching documents as dictionaries.
        """
        page_size = limit if limit is not None else 20
        result = self._engine.query(query, page=0, page_size=page_size)
        return result.get("results", [])

    def aggregate(self, query: str, aggs: dict[str, str]) -> dict[str, Any]:
        """Perform aggregation queries on the index.

        Args:
            query: The search query string.
            aggs: Dictionary of aggregation definitions.

        Returns:
            Dictionary of aggregation results.
        """
        return self._engine.aggregate(q=query, aggs=aggs)

    def export(self, query: str) -> Iterator[dict[str, Any]]:
        """Export documents matching the query using batched iteration.

        Uses pagination to stream documents without loading all into memory.

        Args:
            query: The search query string.

        Yields:
            Matching documents as dictionaries.
        """
        page = 0
        page_size = self._EXPORT_BATCH_SIZE
        while True:
            result = self._engine.query(query, page=page, page_size=page_size)
            docs = result.get("results", [])
            if not docs:
                break
            for doc in docs:
                yield doc
            total = result.get("total", 0)
            if (page + 1) * page_size >= total:
                break
            page += 1

    def stats(self) -> dict[str, Any]:
        """Return statistics about the index.

        Returns:
            Dictionary containing index statistics.
        """
        return self._engine.stats

    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        """Stream documents matching the query without loading all into memory.

        Uses pagination to efficiently iterate over large result sets.

        Args:
            query: The search query string.

        Yields:
            Matching documents as dictionaries.
        """
        yield from self.export(query)

    def __repr__(self) -> str:
        return f"LibraryProvider(path={self.path!r})"