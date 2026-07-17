"""Tests for configuration management."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from flattune.config import (
    BenchmarkBackend,
    BenchmarkConfig,
    DatasetConfig,
    DatasetType,
    ExportConfig,
    ExportFormat,
    FlatTuneConfig,
    FlatseekConfig,
    FlatseekMode,
    FlatseekSource,
    ModelConfig,
    ModelSource,
    QuantizationType,
    TrainBackend,
    TrainConfig,
)


class TestEnums:
    """Test configuration enums."""

    def test_flatseek_mode_values(self):
        assert FlatseekMode.AUTO.value == "auto"
        assert FlatseekMode.LIBRARY.value == "library"
        assert FlatseekMode.CLI.value == "cli"

    def test_flatseek_source_values(self):
        assert FlatseekSource.DIRECTORY.value == "directory"
        assert FlatseekSource.FSK.value == ".fsk"
        assert FlatseekSource.HTTP.value == "http"

    def test_model_source_values(self):
        assert ModelSource.HUGGINGFACE.value == "huggingface"
        assert ModelSource.LOCAL.value == "local"
        assert ModelSource.GGUF.value == "gguf"
        assert ModelSource.MLX.value == "mlx"

    def test_dataset_type_values(self):
        assert DatasetType.INSTRUCTION.value == "instruction"
        assert DatasetType.CHAT.value == "chat"
        assert DatasetType.QA.value == "qa"
        assert DatasetType.CLASSIFICATION.value == "classification"
        assert DatasetType.SUMMARY.value == "summary"
        assert DatasetType.EXTRACTION.value == "extraction"
        assert DatasetType.RAG.value == "rag"

    def test_train_backend_values(self):
        assert TrainBackend.UNSLOTH.value == "unsloth"
        assert TrainBackend.TRANSFORMERS.value == "transformers"
        assert TrainBackend.AXOLOTL.value == "axolotl"
        assert TrainBackend.MLX_LM.value == "mlx-lm"

    def test_export_format_values(self):
        assert ExportFormat.GGUF.value == "gguf"
        assert ExportFormat.MLX.value == "mlx"
        assert ExportFormat.HF_MERGED.value == "hf-merged"
        assert ExportFormat.HF_ADAPTER.value == "hf-adapter"
        assert ExportFormat.SAFETENSORS.value == "safetensors"

    def test_quantization_type_values(self):
        assert QuantizationType.Q4_K_M.value == "q4_k_m"
        assert QuantizationType.Q8_0.value == "q8_0"
        assert QuantizationType.F16.value == "f16"

    def test_benchmark_backend_values(self):
        assert BenchmarkBackend.LMSTUDIO.value == "lmstudio"
        assert BenchmarkBackend.OLLAMA.value == "ollama"
        assert BenchmarkBackend.TRANSFORMERS.value == "transformers"


class TestFlatseekConfig:
    """Test FlatseekConfig dataclass."""

    def test_default_values(self):
        config = FlatseekConfig()
        assert config.mode == FlatseekMode.AUTO
        assert config.source == FlatseekSource.DIRECTORY
        assert config.path is None
        assert config.query is None
        assert config.encryption_key is None

    def test_custom_values(self):
        config = FlatseekConfig(
            mode=FlatseekMode.LIBRARY,
            source=FlatseekSource.FSK,
            path="/path/to/data.fsk",
            query="tag:python",
            encryption_key="secret",
        )
        assert config.mode == FlatseekMode.LIBRARY
        assert config.source == FlatseekSource.FSK
        assert config.path == "/path/to/data.fsk"
        assert config.query == "tag:python"
        assert config.encryption_key == "secret"

    def test_to_dict(self):
        config = FlatseekConfig(
            mode=FlatseekMode.AUTO,
            path="/path/to/data.fsk",
        )
        d = config.to_dict()
        assert d["mode"] == "auto"
        assert d["source"] == "directory"
        assert d["path"] == "/path/to/data.fsk"


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.source == ModelSource.LOCAL
        assert config.repo is None
        assert config.adapter_path is None

    def test_to_dict(self):
        config = ModelConfig(
            source=ModelSource.HUGGINGFACE,
            repo="Qwen/Qwen3-4B",
        )
        d = config.to_dict()
        assert d["source"] == "huggingface"
        assert d["repo"] == "Qwen/Qwen3-4B"


class TestDatasetConfig:
    """Test DatasetConfig dataclass."""

    def test_default_values(self):
        config = DatasetConfig()
        assert config.type == DatasetType.INSTRUCTION
        assert config.query == "*"
        assert config.tags is None
        assert config.train_split == 0.8
        assert config.val_split == 0.1
        assert config.test_split == 0.1
        assert config.max_samples is None
        assert config.generators == []

    def test_to_dict(self):
        config = DatasetConfig(
            type=DatasetType.QA,
            query="tag:python",
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            generators=["qa", "summary"],
        )
        d = config.to_dict()
        assert d["type"] == "qa"
        assert d["query"] == "tag:python"
        assert d["train_split"] == 0.7
        assert d["generators"] == ["qa", "summary"]


class TestTrainConfig:
    """Test TrainConfig dataclass."""

    def test_default_values(self):
        config = TrainConfig()
        assert config.backend == TrainBackend.UNSLOTH
        assert config.epochs == 3
        assert config.lr == 2e-4
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.batch_size == 2
        assert config.seed == 42

    def test_to_dict(self):
        config = TrainConfig(
            backend=TrainBackend.TRANSFORMERS,
            epochs=5,
            lr=1e-4,
            lora_rank=8,
        )
        d = config.to_dict()
        assert d["backend"] == "transformers"
        assert d["epochs"] == 5
        assert d["lr"] == 1e-4
        assert d["lora_rank"] == 8


class TestExportConfig:
    """Test ExportConfig dataclass."""

    def test_default_values(self):
        config = ExportConfig()
        assert config.format == ExportFormat.GGUF
        assert config.quantization == "Q4_K_M"
        assert config.output_dir is None

    def test_to_dict(self):
        config = ExportConfig(
            format=ExportFormat.MLX,
            quantization="Q8_0",
            output_dir="/output",
        )
        d = config.to_dict()
        assert d["format"] == "mlx"
        assert d["quantization"] == "Q8_0"


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_default_values(self):
        config = BenchmarkConfig()
        assert config.backend == BenchmarkBackend.LMSTUDIO
        assert config.num_runs == 10
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_to_dict(self):
        config = BenchmarkConfig(
            backend=BenchmarkBackend.OLLAMA,
            num_runs=5,
            max_tokens=256,
        )
        d = config.to_dict()
        assert d["backend"] == "ollama"
        assert d["num_runs"] == 5
        assert d["max_tokens"] == 256


class TestFlatTuneConfig:
    """Test FlatTuneConfig dataclass."""

    def test_default_values(self):
        config = FlatTuneConfig()
        assert config.name == "flattune-run"
        assert config.description is None
        assert isinstance(config.flatseek, FlatseekConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.dataset, DatasetConfig)
        assert isinstance(config.train, TrainConfig)
        assert isinstance(config.export, ExportConfig)
        assert isinstance(config.benchmark, BenchmarkConfig)

    def test_from_yaml(self, temp_dir: Path):
        yaml_content = """
name: test-project
description: Test description

flatseek:
  mode: library
  path: ./data.fsk
  query: tag:python

dataset:
  type: qa
  generators:
    - qa
    - summary

train:
  backend: transformers
  epochs: 5
  lr: 1e-4
"""
        yaml_file = temp_dir / "test.yml"
        yaml_file.write_text(yaml_content)

        config = FlatTuneConfig.from_yaml(str(yaml_file))

        assert config.name == "test-project"
        assert config.description == "Test description"
        assert config.flatseek.mode == FlatseekMode.LIBRARY
        assert config.flatseek.path == "./data.fsk"
        assert config.dataset.type == DatasetType.QA
        assert config.train.backend == TrainBackend.TRANSFORMERS
        assert config.train.epochs == 5

    def test_from_yaml_with_string_enums(self, temp_dir: Path):
        """Test that string enum values are properly converted."""
        yaml_content = """
name: test-project
flatseek:
  mode: auto
  source: .fsk
dataset:
  type: qa
train:
  backend: unsloth
export:
  format: gguf
benchmark:
  backend: lmstudio
"""
        yaml_file = temp_dir / "test.yml"
        yaml_file.write_text(yaml_content)

        config = FlatTuneConfig.from_yaml(str(yaml_file))

        assert config.flatseek.mode == FlatseekMode.AUTO
        assert config.flatseek.source == FlatseekSource.FSK
        assert config.dataset.type == DatasetType.QA
        assert config.train.backend == TrainBackend.UNSLOTH
        assert config.export.format == ExportFormat.GGUF
        assert config.benchmark.backend == BenchmarkBackend.LMSTUDIO

    def test_to_dict(self):
        config = FlatTuneConfig(
            name="test",
            description="Test project",
        )
        d = config.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test project"
        assert "flatseek" in d
        assert "model" in d
        assert "dataset" in d
        assert "train" in d
        assert "export" in d
        assert "benchmark" in d

    def test_to_yaml(self, temp_dir: Path):
        config = FlatTuneConfig(
            name="test",
            flatseek=FlatseekConfig(path="/data.fsk"),
            train=TrainConfig(epochs=5),
        )
        yaml_file = temp_dir / "output.yml"

        config.to_yaml(str(yaml_file))

        assert yaml_file.exists()
        loaded = yaml.safe_load(yaml_file.read_text())
        assert loaded["name"] == "test"
        assert loaded["flatseek"]["path"] == "/data.fsk"
        assert loaded["train"]["epochs"] == 5

    def test_to_yaml_roundtrip(self, temp_dir: Path):
        """Test that config can be saved and loaded back."""
        original = FlatTuneConfig(
            name="roundtrip-test",
            description="Testing roundtrip",
            flatseek=FlatseekConfig(mode=FlatseekMode.CLI, path="/test.fsk"),
            dataset=DatasetConfig(type=DatasetType.SUMMARY, generators=["summary"]),
            train=TrainConfig(backend=TrainBackend.TRANSFORMERS, epochs=10, lr=5e-5),
            export=ExportConfig(format=ExportFormat.MLX, quantization="Q8_0"),
            benchmark=BenchmarkConfig(backend=BenchmarkBackend.OLLAMA, num_runs=20),
        )
        yaml_file = temp_dir / "roundtrip.yml"

        original.to_yaml(str(yaml_file))
        loaded = FlatTuneConfig.from_yaml(str(yaml_file))

        assert loaded.name == original.name
        assert loaded.description == original.description
        assert loaded.flatseek.mode == original.flatseek.mode
        assert loaded.flatseek.path == original.flatseek.path
        assert loaded.dataset.type == original.dataset.type
        assert loaded.train.backend == original.train.backend
        assert loaded.train.epochs == original.train.epochs
        assert loaded.export.format == original.export.format
        assert loaded.benchmark.backend == original.benchmark.backend
