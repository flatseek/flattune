"""CLI-based FlatSeek provider using the flatseek executable."""

import json
import subprocess
from collections.abc import Iterator
from typing import Any

from flattune.flatseek.provider import FlatseekProvider


class CLIProvider(FlatseekProvider):
    """FlatSeek provider using the flatseek CLI executable.

    This is the fallback integration mode when FlatSeek is not available
    as a Python package. It spawns the flatseek CLI process and parses output.
    """

    def __init__(
        self,
        path: str,
        query: str | None = None,
        encryption_key: str | None = None,
    ):
        """Initialize the CLI provider.

        Args:
            path: Path to the FlatSeek index (directory or .fsk file).
            query: Default query to apply to all operations.
            encryption_key: Optional encryption key for encrypted indexes.
        """
        self.path = path
        self.default_query = query or "*"
        self.encryption_key = encryption_key

        # Verify flatseek CLI is available
        if not self._is_flatseek_available():
            raise RuntimeError(
                "flatseek CLI is not available. "
                "Install FlatSeek or use library mode with FlatSeek Python package."
            )

    def _is_flatseek_available(self) -> bool:
        """Check if the flatseek CLI is available."""
        try:
            result = subprocess.run(
                ["flatseek", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _run_command(self, args: list[str]) -> str:
        """Run a flatseek CLI command and return output.

        Args:
            args: Command-line arguments for flatseek.

        Returns:
            Raw stdout output from the command.

        Raises:
            RuntimeError: If the command fails.
        """
        cmd = ["flatseek"] + args

        # Add passphrase if encryption key is provided
        if self.encryption_key:
            cmd.extend(["--passphrase", self.encryption_key])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            if result.returncode != 0:
                raise RuntimeError(f"flatseek CLI failed: {result.stderr}")
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError("flatseek CLI command timed out")

    def columns(self) -> list[str]:
        """Return available columns/fields in the index.

        Note: flatseek CLI doesn't have a direct columns command.
        We extract columns from the stats output.
        """
        stats = self.stats()
        # Stats output contains column_map with column information
        column_map = stats.get("column_map", {})
        if isinstance(column_map, dict):
            return list(column_map.keys())
        # Fallback: try to get columns from first result if available
        return []

    def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Search the index with a query.

        Uses: flatseek search <data_dir> [query] [-n page_size]
        """
        args = [self.path]
        if query:
            args.append(query)
        args.extend(["-n", str(limit) if limit is not None else "20"])

        output = self._run_command(["search"] + args)
        # Search output is human-readable text, need to parse it
        # Format: "score\\tJSON doc" per line
        results = []
        for line in output.strip().split("\n"):
            if line.strip():
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    try:
                        doc = json.loads(parts[1])
                        results.append(doc)
                    except json.JSONDecodeError:
                        pass
        return results

    def aggregate(self, query: str, aggs: dict[str, str]) -> dict[str, Any]:
        """Perform aggregation queries on the index.

        Note: flatseek CLI doesn't support aggregations directly.
        This is a simplified implementation that may not provide full aggregation support.
        """
        # For CLI mode, we simulate aggregations by searching and processing results
        # This is a limitation of CLI mode - full aggregation requires library mode
        results = self.search(query, limit=1000)

        aggs_result = {}
        for agg_name, agg_def in aggs.items():
            if isinstance(agg_def, dict):
                agg_type = (
                    agg_def.get("terms") or agg_def.get("stats") or agg_def.get("cardinality")
                )
                if isinstance(agg_type, dict):
                    field = agg_type.get("field")
                    if agg_type.get("terms") and field:
                        # Count by field values
                        counts: dict[str, int] = {}
                        for doc in results:
                            value = doc.get(field, "missing")
                            counts[value] = counts.get(value, 0) + 1
                        aggs_result[agg_name] = {
                            "buckets": [{"key": k, "doc_count": v} for k, v in counts.items()]
                        }
                    elif agg_type.get("stats") and field:
                        # Compute stats
                        values = [
                            float(doc[field]) for doc in results if doc.get(field) is not None
                        ]
                        if values:
                            aggs_result[agg_name] = {
                                "count": len(values),
                                "min": min(values),
                                "max": max(values),
                                "avg": sum(values) / len(values),
                                "sum": sum(values),
                            }
        return aggs_result

    def export(self, query: str) -> Iterator[dict[str, Any]]:
        """Export documents matching the query.

        Uses: flatseek export <data_dir> -q <query> [-n limit]
        """
        args = [self.path, "-q", query]

        output = self._run_command(["export"] + args)
        for line in output.strip().split("\n"):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass

    def stats(self) -> dict[str, Any]:
        """Return statistics about the index.

        Uses: flatseek stats <data_dir>
        """
        output = self._run_command(["stats", self.path])
        # Stats output is typically JSON
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            # May be human-readable, return as-is
            return {"raw": output}

    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        """Stream documents matching the query without loading all into memory.

        Uses pagination with export command to stream documents.
        """
        # Export command supports streaming via -n limit per batch
        batch_size = 1000
        page = 0

        while True:
            args = [self.path, "-q", query, "-n", str(batch_size), "-p", str(page)]
            output = self._run_command(["export"] + args)

            docs = []
            for line in output.strip().split("\n"):
                if line.strip():
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

            if not docs:
                break

            yield from docs

            if len(docs) < batch_size:
                break
            page += 1

    def __repr__(self) -> str:
        return f"CLIProvider(path={self.path!r})"
