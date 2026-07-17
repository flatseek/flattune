"""Base trainer interface for all training backends."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class TrainerBase(ABC):
    """Abstract base class for all training backends.

    Defines the interface that all trainers (Unsloth, Axolotl, MLX-LM, etc.)
    must implement.

    Supports two data modes:
    1. Dataset path - load from saved JSONL file
    2. Streaming - directly stream from FlatSeek query (no intermediate file)
    """

    def __init__(
        self,
        model_path: str,
        dataset_path: str | None = None,
        output_dir: Path | None = None,
        config: Any | None = None,
        flatseek_provider: Any | None = None,
        flatseek_query: str | None = None,
    ):
        """Initialize the trainer.

        Args:
            model_path: Path to the base model.
            dataset_path: Path to training dataset (if not streaming).
            output_dir: Directory to save checkpoints and outputs.
            config: Training configuration.
            flatseek_provider: FlatSeek provider for streaming (optional).
            flatseek_query: Query to stream from FlatSeek (optional).
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir) if output_dir else Path("checkpoints")
        self.config = config
        self.flatseek_provider = flatseek_provider
        self.flatseek_query = flatseek_query or "*"

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def stream_documents(self) -> Iterator[dict]:
        """Stream documents from FlatSeek or file.

        Yields:
            Document dictionaries.
        """
        if self.flatseek_provider:
            # Stream directly from FlatSeek
            yield from self.flatseek_provider.stream(self.flatseek_query)
        elif self.dataset_path:
            # Load from file
            with open(self.dataset_path) as f:
                for line in f:
                    if line.strip():
                        import json
                        yield json.loads(line.strip())
        else:
            raise ValueError("Either flatseek_provider or dataset_path must be provided")

    @abstractmethod
    def train(self) -> dict[str, Any]:
        """Run the training process.

        Returns:
            Dictionary containing training metrics and artifacts.
        """
        pass

    @abstractmethod
    def merge(self) -> Path:
        """Merge the LoRA adapter with the base model.

        Returns:
            Path to the merged model.
        """
        pass

    @abstractmethod
    def export(self, format: str) -> Path:
        """Export the model to the specified format.

        Args:
            format: Target format (gguf, mlx, hf).

        Returns:
            Path to the exported model.
        """
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Evaluate the trained model.

        Returns:
            Dictionary of evaluation metrics.
        """
        pass

    def get_checkpoints(self) -> list[Path]:
        """Get list of saved checkpoints.

        Returns:
            List of checkpoint file paths.
        """
        if not self.output_dir.exists():
            return []
        checkpoints = list(self.output_dir.glob("*.pt"))
        checkpoints.extend(self.output_dir.glob("*.safetensors"))
        checkpoints.extend(self.output_dir.glob("*.bin"))
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime)

    def get_latest_checkpoint(self) -> Path | None:
        """Get the path to the latest checkpoint.

        Returns:
            Path to the latest checkpoint or None if no checkpoints exist.
        """
        checkpoints = self.get_checkpoints()
        return checkpoints[-1] if checkpoints else None
