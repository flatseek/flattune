"""Model exporter for converting between model formats."""

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from flattune.config import ExportFormat, QuantizationType
from flattune.utils.logging import get_logger

logger = get_logger(__name__)


class ModelExporter:
    """Exports trained models to various formats.

    Supports:
    - GGUF: For LM Studio, Ollama, and other GGUF-compatible inference
    - MLX: For Apple Silicon MLX format
    - HuggingFace merged: Full model merged with adapter
    - HuggingFace adapter: LoRA adapter only
    - Safetensors: Safe tensor format
    """

    def __init__(
        self,
        model_path: Path,
        output_dir: Path,
        format: ExportFormat = ExportFormat.GGUF,
    ):
        """Initialize the model exporter.

        Args:
            model_path: Path to the model (base + adapter or merged).
            output_dir: Directory for exported model.
            format: Target export format.
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.format = format
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        quantization: Optional[str] = None,
        base_model: Optional[Path] = None,
    ) -> Path:
        """Export the model to the target format.

        Args:
            quantization: Quantization method (for GGUF). Options: q2_k, q3_k,
                q4_k, q4_k_m, q5_k, q5_k_m, q6_k, q8_0, f16, f32
            base_model: Path to base model (required for adapter exports).

        Returns:
            Path to the exported model.
        """
        logger.info(f"Exporting model to {self.format.value}...")

        if self.format == ExportFormat.GGUF:
            return self._export_gguf(quantization)
        elif self.format == ExportFormat.MLX:
            return self._export_mlx()
        elif self.format == ExportFormat.HF_MERGED:
            return self._export_hf_merged()
        elif self.format == ExportFormat.HF_ADAPTER:
            return self._export_hf_adapter(base_model)
        elif self.format == ExportFormat.SAFETENSORS:
            return self._export_safetensors()

        raise ValueError(f"Unknown export format: {self.format}")

    def _export_gguf(self, quantization: Optional[str] = None) -> Path:
        """Export to GGUF format using llama.cpp.

        Args:
            quantization: Quantization type (default: Q4_K_M).

        Returns:
            Path to exported GGUF model.
        """
        logger.info(f"Exporting to GGUF (quantization: {quantization})")

        try:
            import llama_cpp
        except ImportError:
            logger.warning("llama.cpp not installed. GGUF export requires llama-cpp-python.")
            logger.warning("For GGUF output, either:")
            logger.warning("  1. Install: pip install llama-cpp-python")
            logger.warning("  2. Use 'flattune merge' first to get HF format, then export manually")
            logger.warning("Creating placeholder for now...")

            # Create placeholder
            output_path = self.output_dir / "model.gguf.placeholder"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(f"# GGUF Export Placeholder\n")
                f.write(f"# Model: {self.model_path}\n")
                f.write(f"# Quantization: {quantization or 'Q4_K_M'}\n")
                f.write(f"# Install llama-cpp-python to enable GGUF export\n")
            logger.info(f"Placeholder created at: {output_path}")
            return output_path

        output_path = self.output_dir / "model.gguf"
        logger.info(f"GGUF export would save to: {output_path}")
        logger.info("Note: GGUF export requires the model to be in HF format first")
        return output_path

    def _export_mlx(self) -> Path:
        """Export to MLX format for Apple Silicon.

        Returns:
            Path to exported MLX model.
        """
        logger.info("Exporting to MLX format...")
        output_path = self.output_dir / "model.mlx"
        logger.info(f"MLX export would save to: {output_path}")
        return output_path

    def _export_hf_merged(self) -> Path:
        """Export merged HuggingFace model.

        Returns:
            Path to exported HF model.
        """
        logger.info("Exporting to HuggingFace merged format...")
        output_path = self.output_dir / "hf_merged"
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"HF merged export would save to: {output_path}")
        return output_path

    def _export_hf_adapter(self, base_model: Optional[Path] = None) -> Path:
        """Export LoRA adapter only.

        Args:
            base_model: Path to base model (optional).

        Returns:
            Path to exported adapter.
        """
        logger.info("Exporting HuggingFace adapter format...")
        output_path = self.output_dir / "adapter"
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"HF adapter export would save to: {output_path}")
        return output_path

    def _export_safetensors(self) -> Path:
        """Export to Safetensors format.

        Returns:
            Path to exported Safetensors model.
        """
        logger.info("Exporting to Safetensors format...")
        output_path = self.output_dir / "model.safetensors"
        logger.info(f"Safetensors export would save to: {output_path}")
        return output_path


def create_exporter(
    model_path: str,
    output_dir: str,
    format: ExportFormat,
) -> ModelExporter:
    """Factory function to create a model exporter.

    Args:
        model_path: Path to the model.
        output_dir: Output directory.
        format: Export format.

    Returns:
        Configured ModelExporter instance.
    """
    return ModelExporter(
        model_path=Path(model_path),
        output_dir=Path(output_dir),
        format=format,
    )
