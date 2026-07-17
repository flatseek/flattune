"""FlatTune - The missing bridge between Flatseek and modern LLM fine-tuning frameworks."""

__version__ = "0.1.0"
__author__ = "FlatTune Contributors"

from flattune.benchmark import BenchmarkReporter, BenchmarkRunner
from flattune.config import (
    BenchmarkBackend,
    BenchmarkConfig,
    DatasetConfig,
    DatasetType,
    ExportConfig,
    ExportFormat,
    FlatseekConfig,
    FlatseekMode,
    FlatseekSource,
    FlatTuneConfig,
    ModelConfig,
    ModelSource,
    TrainBackend,
    TrainConfig,
)
from flattune.dataset import DatasetBuilder, DatasetSplitter
from flattune.exporter import ModelExporter
from flattune.flatseek import CLIProvider, FlatseekProvider, LibraryProvider
from flattune.flatseek.auto import create_provider
from flattune.lmstudio import LMStudioClient
from flattune.ollama import OllamaClient
from flattune.trainer import TrainerBase, TrainerFactory
from flattune.utils import (
    compute_hash,
    create_run_folder,
    get_git_info,
    get_logger,
    get_run_folder,
    get_timestamp,
    set_seed,
    setup_logging,
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
