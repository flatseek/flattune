"""Automatic FlatSeek provider selection based on available backends."""

from typing import Optional

from flattune.flatseek.provider import FlatseekProvider
from flattune.flatseek.library_provider import LibraryProvider
from flattune.flatseek.cli_provider import CLIProvider
from flattune.config import FlatseekMode, FlatseekSource


def create_provider(
    path: str,
    mode: FlatseekMode = FlatseekMode.AUTO,
    source: FlatseekSource = FlatseekSource.DIRECTORY,
    query: Optional[str] = None,
    encryption_key: Optional[str] = None,
) -> FlatseekProvider:
    """Create the appropriate FlatSeek provider based on available backends.

    Priority:
    1. Explicit configuration in YAML (mode parameter)
    2. Embedded Python library (LibraryProvider)
    3. FlatSeek CLI (CLIProvider)
    4. Error with clear installation guidance

    Args:
        path: Path to the FlatSeek index (directory or .fsk file).
        mode: Integration mode (auto, library, or cli).
        source: Data source type (directory, .fsk, or http).
        query: Default query to apply to all operations.
        encryption_key: Optional encryption key for encrypted indexes.

    Returns:
        An appropriate FlatseekProvider implementation.

    Raises:
        ImportError: If neither library nor CLI mode is available.
        RuntimeError: If the specified mode is not available.
    """
    if mode == FlatseekMode.LIBRARY:
        return LibraryProvider(
            path=path,
            query=query,
            encryption_key=encryption_key,
        )

    if mode == FlatseekMode.CLI:
        return CLIProvider(
            path=path,
            query=query,
            encryption_key=encryption_key,
        )

    # AUTO mode: try library first, then CLI
    if mode == FlatseekMode.AUTO:
        # Try library mode first
        try:
            return LibraryProvider(
                path=path,
                query=query,
                encryption_key=encryption_key,
            )
        except ImportError:
            pass

        # Fall back to CLI mode
        try:
            return CLIProvider(
                path=path,
                query=query,
                encryption_key=encryption_key,
            )
        except RuntimeError:
            pass

        # Neither mode is available
        raise ImportError(
            "FlatSeek is not available. Please either:\n"
            "1. Install the FlatSeek Python library: pip install flatseek\n"
            "2. Install the flatseek CLI and ensure it's in your PATH\n"
        )

    raise ValueError(f"Unknown FlatSeek mode: {mode}")
