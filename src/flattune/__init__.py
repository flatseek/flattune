"""FlatTune - The missing bridge between Flatseek and modern LLM fine-tuning frameworks."""

__version__ = "0.1.0"
__author__ = "FlatTune Contributors"

from flattune.config import FlatTuneConfig, FlatseekConfig, ModelConfig, DatasetConfig
from flattune.config import TrainConfig, ExportConfig, BenchmarkConfig
from flattune.config import FlatseekMode, FlatseekSource, ModelSource, DatasetType
from flattune.config import TrainBackend, ExportFormat, BenchmarkBackend

from flattune.flatseek import FlatseekProvider, LibraryProvider, CLIProvider
from flattune.flatseek.auto import create_provider

from flattune.dataset import DatasetBuilder, DatasetSplitter

from flattune.trainer import TrainerBase, TrainerFactory

from flattune.exporter import ModelExporter

from flattune.lmstudio import LMStudioClient
from flattune.ollama import OllamaClient

from flattune.benchmark import BenchmarkRunner, BenchmarkReporter

from flattune.utils import (
    get_logger,
    setup_logging,
    create_run_folder,
    get_run_folder,
    compute_hash,
    get_git_info,
    get_timestamp,
    set_seed,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "FlatTuneConfig",
    "FlatseekConfig",
    "ModelConfig",
    "DatasetConfig",
    "TrainConfig",
    "ExportConfig",
    "BenchmarkConfig",
    # Enums
    "FlatseekMode",
    "FlatseekSource",
    "ModelSource",
    "DatasetType",
    "TrainBackend",
    "ExportFormat",
    "BenchmarkBackend",
    # FlatSeek
    "FlatseekProvider",
    "LibraryProvider",
    "CLIProvider",
    "create_provider",
    # Dataset
    "DatasetBuilder",
    "DatasetSplitter",
    # Trainer
    "TrainerBase",
    "TrainerFactory",
    # Exporter
    "ModelExporter",
    # Clients
    "LMStudioClient",
    "OllamaClient",
    # Benchmark
    "BenchmarkRunner",
    "BenchmarkReporter",
    # Utils
    "get_logger",
    "setup_logging",
    "create_run_folder",
    "get_run_folder",
    "compute_hash",
    "get_git_info",
    "get_timestamp",
    "set_seed",
]
