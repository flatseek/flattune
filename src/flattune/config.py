"""Configuration management for FlatTune using YAML with dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml


class FlatseekMode(Enum):
    """Flatseek integration mode."""
    AUTO = "auto"
    LIBRARY = "library"
    CLI = "cli"


class FlatseekSource(Enum):
    """Flatseek data source type."""
    DIRECTORY = "directory"
    FSK = ".fsk"
    HTTP = "http"


class ModelSource(Enum):
    """Model source type."""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    GGUF = "gguf"
    MLX = "mlx"


class DatasetType(Enum):
    """Dataset generation type."""
    INSTRUCTION = "instruction"
    CHAT = "chat"
    QA = "qa"
    CLASSIFICATION = "classification"
    SUMMARY = "summary"
    EXTRACTION = "extraction"
    RAG = "rag"


class TrainBackend(Enum):
    """Training backend."""
    UNSLOTH = "unsloth"
    TRANSFORMERS = "transformers"
    AXOLOTL = "axolotl"
    MLX_LM = "mlx-lm"


class ExportFormat(Enum):
    """Export format."""
    GGUF = "gguf"
    MLX = "mlx"
    HF_MERGED = "hf-merged"
    HF_ADAPTER = "hf-adapter"
    SAFETENSORS = "safetensors"


class QuantizationType(Enum):
    """Quantization types for GGUF export."""
    Q2_K = "q2_k"
    Q3_K = "q3_k"
    Q3_K_M = "q3_k_m"
    Q4_K = "q4_k"
    Q4_K_M = "q4_k_m"
    Q5_K = "q5_k"
    Q5_K_M = "q5_k_m"
    Q6_K = "q6_k"
    Q8_0 = "q8_0"
    F16 = "f16"
    F32 = "f32"


class BenchmarkBackend(Enum):
    """Benchmark backend."""
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"
    TRANSFORMERS = "transformers"


@dataclass
class FlatseekConfig:
    """Configuration for Flatseek integration."""
    mode: FlatseekMode = FlatseekMode.AUTO
    source: FlatseekSource = FlatseekSource.DIRECTORY
    path: Optional[str] = None
    query: Optional[str] = None
    encryption_key: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value if isinstance(self.mode, FlatseekMode) else self.mode,
            "source": self.source.value if isinstance(self.source, FlatseekSource) else self.source,
            "path": self.path,
            "query": self.query,
            "encryption_key": self.encryption_key,
        }


@dataclass
class ModelConfig:
    """Configuration for the model."""
    source: ModelSource = ModelSource.LOCAL
    repo: Optional[str] = None
    path: str = "~/.lmstudio/models/prism-ml/Bonsai-1.7B-gguf/Bonsai-1.7B-Q1_0.gguf"
    adapter_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source.value if isinstance(self.source, ModelSource) else self.source,
            "repo": self.repo,
            "path": self.path,
            "adapter_path": self.adapter_path,
        }


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    type: DatasetType = DatasetType.INSTRUCTION
    query: str = "*"           # Flatseek query to filter records
    tags: Optional[list[str]] = None  # Filter by tags/categories (e.g., ["python", "rust"])
    field_mapping: Optional[dict[str, str]] = None  # Map index fields to generator input
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_samples: Optional[int] = None
    min_length: int = 10
    max_length: int = 2048
    generators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value if isinstance(self.type, DatasetType) else self.type,
            "query": self.query,
            "tags": self.tags,
            "field_mapping": self.field_mapping,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "max_samples": self.max_samples,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "generators": self.generators,
        }


@dataclass
class TrainConfig:
    """Configuration for training."""
    backend: TrainBackend = TrainBackend.UNSLOTH
    epochs: int = 3
    lr: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 2
    max_seq_length: int = 2048
    max_samples: Optional[int] = None
    gradient_accumulation: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    use_qwen3_format: bool = False

    def to_dict(self) -> dict:
        return {
            "backend": self.backend.value if isinstance(self.backend, TrainBackend) else self.backend,
            "epochs": self.epochs,
            "lr": self.lr,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "gradient_accumulation": self.gradient_accumulation,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_qwen3_format": self.use_qwen3_format,
        }


@dataclass
class ExportConfig:
    """Configuration for export."""
    format: ExportFormat = ExportFormat.GGUF
    quantization: str = "Q4_K_M"
    output_dir: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "format": self.format.value if isinstance(self.format, ExportFormat) else self.format,
            "quantization": self.quantization,
            "output_dir": self.output_dir,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    backend: BenchmarkBackend = BenchmarkBackend.LMSTUDIO
    prompt_file: Optional[str] = None
    num_runs: int = 10
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    def to_dict(self) -> dict:
        return {
            "backend": self.backend.value if isinstance(self.backend, BenchmarkBackend) else self.backend,
            "prompt_file": self.prompt_file,
            "num_runs": self.num_runs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass
class FlatTuneConfig:
    """Main configuration for FlatTune."""
    name: str = "flattune-run"
    description: Optional[str] = None
    flatseek: FlatseekConfig = field(default_factory=FlatseekConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> FlatTuneConfig:
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty YAML file: {path}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> FlatTuneConfig:
        """Create configuration from a dictionary."""
        flatseek_data = data.get("flatseek", {})
        if isinstance(flatseek_data.get("mode"), str):
            flatseek_data["mode"] = FlatseekMode(flatseek_data["mode"])
        if isinstance(flatseek_data.get("source"), str):
            flatseek_data["source"] = FlatseekSource(flatseek_data["source"])

        model_data = data.get("model", {})
        if isinstance(model_data.get("source"), str):
            model_data["source"] = ModelSource(model_data["source"])

        dataset_data = data.get("dataset", {})
        if isinstance(dataset_data.get("type"), str):
            dataset_data["type"] = DatasetType(dataset_data["type"])

        train_data = data.get("train", {})
        if isinstance(train_data.get("backend"), str):
            train_data["backend"] = TrainBackend(train_data["backend"])
        # Convert numeric values from strings
        if "lr" in train_data:
            train_data["lr"] = float(train_data["lr"])
        if "lora_rank" in train_data:
            train_data["lora_rank"] = int(train_data["lora_rank"])
        if "lora_alpha" in train_data:
            train_data["lora_alpha"] = int(train_data["lora_alpha"])

        export_data = data.get("export", {})
        if isinstance(export_data.get("format"), str):
            export_data["format"] = ExportFormat(export_data["format"])

        benchmark_data = data.get("benchmark", {})
        if isinstance(benchmark_data.get("backend"), str):
            benchmark_data["backend"] = BenchmarkBackend(benchmark_data["backend"])

        return cls(
            name=data.get("name", "flattune-run"),
            description=data.get("description"),
            flatseek=FlatseekConfig(**flatseek_data) if flatseek_data else FlatseekConfig(),
            model=ModelConfig(**model_data) if model_data else ModelConfig(),
            dataset=DatasetConfig(**dataset_data) if dataset_data else DatasetConfig(),
            train=TrainConfig(**train_data) if train_data else TrainConfig(),
            export=ExportConfig(**export_data) if export_data else ExportConfig(),
            benchmark=BenchmarkConfig(**benchmark_data) if benchmark_data else BenchmarkConfig(),
            output_dir=data.get("output_dir", "outputs"),
        )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "flatseek": self.flatseek.to_dict(),
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "train": self.train.to_dict(),
            "export": self.export.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "output_dir": self.output_dir,
        }

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
