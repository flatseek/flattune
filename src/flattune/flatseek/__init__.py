"""FlatSeek integration for FlatTune."""

from flattune.flatseek.auto import create_provider
from flattune.flatseek.cli_provider import CLIProvider
from flattune.flatseek.library_provider import LibraryProvider
from flattune.flatseek.provider import FlatseekProvider

__all__ = ["FlatseekProvider", "LibraryProvider", "CLIProvider", "create_provider"]
