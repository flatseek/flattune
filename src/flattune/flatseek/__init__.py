"""FlatSeek integration for FlatTune."""

from flattune.flatseek.provider import FlatseekProvider
from flattune.flatseek.library_provider import LibraryProvider
from flattune.flatseek.cli_provider import CLIProvider
from flattune.flatseek.auto import create_provider

__all__ = ["FlatseekProvider", "LibraryProvider", "CLIProvider", "create_provider"]
